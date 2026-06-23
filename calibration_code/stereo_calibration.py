# -*- coding: utf-8 -*-
"""
相机-投影仪双目标定模块
-----------------------------
功能：
1. 使用 cv2.stereoCalibrate 联合优化相机和投影仪的内外参数
2. 求解相机到投影仪的旋转矩阵 R 和平移向量 T
3. 支持固定内参（CALIB_FIX_INTRINSIC）模式和联合优化模式
4. 计算双目标定全局重投影误差

参考 MATLAB Stereo Camera Calibrator Toolbox
"""

import numpy as np
import cv2


class StereoCalibration:
    """
    相机-投影仪双目标定器

    将投影仪视为第二台相机进行标定，求解两台虚拟相机之间的位姿关系
    """

    def __init__(self, config):
        """
        初始化双目标定器

        参数：
            config: CalibrationConfig 配置实例
        """
        self.config = config

        # 双目标定结果
        self.ret = None                 # 全局重投影误差的 RMS
        self.camera_mtx = None          # 相机内参
        self.camera_dist = None         # 相机畸变
        self.projector_mtx = None       # 投影仪内参
        self.projector_dist = None      # 投影仪畸变
        self.R = None                   # 投影仪坐标系相对于相机坐标系的旋转矩阵
        self.T = None                   # 投影仪坐标系相对于相机坐标系的平移向量
        self.E = None                   # 本质矩阵
        self.F = None                   # 基础矩阵

        # 标定时使用的数据
        self.object_points = []         # 世界三维点集
        self.camera_points = []         # 相机二维角点
        self.projector_points = []      # 投影仪二维角点
        self.image_size = None          # 相机图像尺寸

        # 每幅图像的重投影误差
        self.per_pose_error = {}

    # ==================== 数据准备 ====================

    def prepare_data(self, obj_points_list, camera_points_list,
                     projector_points_list, pose_names, image_size,
                     camera_mtx=None, camera_dist=None,
                     projector_mtx=None, projector_dist=None):
        """
        准备双目标定数据

        参数：
            obj_points_list: list of np.ndarray, 每组位姿的世界三维坐标
            camera_points_list: list of np.ndarray, 相机检测角点
            projector_points_list: list of np.ndarray, 投影仪解码角点
            pose_names: list of str, 位姿名称
            image_size: tuple (width, height)
            camera_mtx: 相机内参矩阵（3x3），若提供则使用固定内参
            camera_dist: 相机畸变系数
            projector_mtx: 投影仪内参矩阵
            projector_dist: 投影仪畸变系数
        """
        self.object_points = []
        self.camera_points = []
        self.projector_points = []
        self.pose_names = []

        for objp, camp, projp, name in zip(
                obj_points_list, camera_points_list, projector_points_list, pose_names):
            if objp is None or camp is None or projp is None:
                continue
            self.object_points.append(np.asarray(objp, dtype=np.float32))
            self.camera_points.append(np.asarray(camp, dtype=np.float32))
            self.projector_points.append(np.asarray(projp, dtype=np.float32))
            self.pose_names.append(name)

        self.image_size = image_size
        self.camera_mtx = camera_mtx
        self.camera_dist = camera_dist
        self.projector_mtx = projector_mtx
        self.projector_dist = projector_dist

        if len(self.object_points) < 3:
            raise RuntimeError(f"双目标定有效位姿数量不足（当前 {len(self.object_points)}，至少需要 3 幅）")

    # ==================== 执行双目标定 ====================

    def calibrate(self, fix_intrinsic=True):
        """
        执行相机-投影仪双目标定

        标志位说明：
            CALIB_FIX_INTRINSIC: 固定两台设备的内参，只优化外参 R, T
            CALIB_USE_INTRINSIC_GUESS: 使用传入的内参作为初始猜测值进行优化
            CALIB_SAME_FOCAL_LENGTH: 约束 fx 和 fy 相同
            CALIB_ZERO_TANGENT_DIST: 切向畸变设为0 (p1=p2=0)
            CALIB_RATIONAL_MODEL: 使用8参数畸变模型

        参数：
            fix_intrinsic: bool, 是否固定内参（推荐使用，因单目标定已求得较优内参）

        返回：
            success: bool
        """
        # 标定迭代终止条件
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                    self.config.stereo_criteria[1],
                    self.config.stereo_criteria[2])

        # 构造 flags
        flags = 0
        if fix_intrinsic and (self.camera_mtx is not None and self.projector_mtx is not None):
            # 固定内参，只优化 R, T
            flags |= cv2.CALIB_FIX_INTRINSIC
        else:
            # 使用传入的内参作为初始值，与外参一起优化
            if self.camera_mtx is not None:
                flags |= cv2.CALIB_USE_INTRINSIC_GUESS

        try:
            self.ret, self.camera_mtx, self.camera_dist, \
                self.projector_mtx, self.projector_dist, \
                self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
                    self.object_points,
                    self.camera_points,
                    self.projector_points,
                    self.camera_mtx, self.camera_dist,
                    self.projector_mtx, self.projector_dist,
                    self.image_size,
                    criteria=criteria,
                    flags=flags
                )
        except cv2.error as e:
            raise RuntimeError(f"双目标定失败: {str(e)}")

        # 计算每幅图像的重投影误差
        self._compute_per_pose_error()

        return True

    def _compute_per_pose_error(self):
        """
        计算并记录每位姿的双目标定重投影误差

        思路：
            对每个位姿，用 cv2.solvePnP 分别估计其相对于相机/投影仪的外参
            然后将世界点投影回两幅图像，计算与观测点的平均误差
        """
        self.per_pose_error = {}

        for i, (objp, camp, projp, name) in enumerate(zip(
                self.object_points, self.camera_points, self.projector_points, self.pose_names)):

            # 使用当前的外参估计（rvec/tvec 由 stereoCalibrate 内部估计但不返回）
            # 这里用 solvePnP 单独估计每个位姿的相机和投影仪外参
            try:
                # 相机侧
                ret_cam, rvec_cam, tvec_cam = cv2.solvePnP(
                    objp, camp, self.camera_mtx, self.camera_dist
                )
                # 投影仪侧
                ret_proj, rvec_proj, tvec_proj = cv2.solvePnP(
                    objp, projp, self.projector_mtx, self.projector_dist
                )
                if not ret_cam or not ret_proj:
                    continue

                # 投影回相机
                proj_cam, _ = cv2.projectPoints(objp, rvec_cam, tvec_cam,
                                                self.camera_mtx, self.camera_dist)
                err_cam = cv2.norm(camp, proj_cam, cv2.NORM_L2) / len(camp)

                # 投影回投影仪
                proj_proj, _ = cv2.projectPoints(objp, rvec_proj, tvec_proj,
                                                 self.projector_mtx, self.projector_dist)
                err_proj = cv2.norm(projp, proj_proj, cv2.NORM_L2) / len(projp)

                # 总误差
                err_total = (err_cam + err_proj) / 2.0
                self.per_pose_error[name] = {
                    'camera_error': err_cam,
                    'projector_error': err_proj,
                    'total_error': err_total
                }
            except Exception as e:
                self.per_pose_error[name] = {
                    'camera_error': -1.0,
                    'projector_error': -1.0,
                    'total_error': -1.0,
                    'note': f'solvePnP 失败: {str(e)}'
                }

    # ==================== 结果评估 ====================

    def get_mean_reprojection_error(self):
        """获取平均重投影误差（所有位姿 total_error 的均值）"""
        if not self.per_pose_error:
            return float(self.ret) if self.ret is not None else -1.0
        errors = [v['total_error'] for v in self.per_pose_error.values() if v['total_error'] >= 0]
        if not errors:
            return float(self.ret) if self.ret is not None else -1.0
        return float(np.mean(errors))

    # ==================== 结果获取 ====================

    def get_calibration_result(self):
        """
        获取完整双目标定结果

        返回：
            result: dict
                {
                    'camera_mtx': 相机内参矩阵 3x3,
                    'camera_dist': 相机畸变系数,
                    'projector_mtx': 投影仪内参矩阵 3x3,
                    'projector_dist': 投影仪畸变系数,
                    'R': 相机->投影仪的旋转矩阵 3x3,
                    'T': 相机->投影仪的平移向量 3x1,
                    'E': 本质矩阵,
                    'F': 基础矩阵,
                    'rms': cv2.stereoCalibrate 返回的 RMS 误差,
                    'per_pose_error': 每位姿的误差字典,
                    'mean_error': 平均重投影误差,
                    'valid_poses': 有效位姿名称列表,
                    'pose_count': 有效位姿数量
                }
        """
        return {
            'camera_mtx': self.camera_mtx,
            'camera_dist': self.camera_dist,
            'projector_mtx': self.projector_mtx,
            'projector_dist': self.projector_dist,
            'R': self.R,
            'T': self.T,
            'E': self.E,
            'F': self.F,
            'rms': float(self.ret) if self.ret is not None else -1.0,
            'per_pose_error': self.per_pose_error,
            'mean_error': self.get_mean_reprojection_error(),
            'valid_poses': self.pose_names,
            'pose_count': len(self.pose_names)
        }

    def print_result(self):
        """打印双目标定结果"""
        print("========== 相机-投影仪双目标定结果 ==========")
        print(f"参与标定的位姿数: {len(self.pose_names)}")
        print(f"全局 RMS 重投影误差: {self.ret:.4f} 像素")
        print(f"平均重投影误差: {self.get_mean_reprojection_error():.4f} 像素")

        if self.R is not None:
            print(f"\n相机 -> 投影仪 旋转矩阵 R:")
            print(self.R)
        if self.T is not None:
            print(f"\n相机 -> 投影仪 平移向量 T:")
            print(self.T.ravel())

        # 打印每位姿的误差
        if self.per_pose_error:
            print(f"\n位姿详细误差:")
            for name, err in self.per_pose_error.items():
                if err['total_error'] >= 0:
                    print(f"  {name}: 相机 {err['camera_error']:.3f}px, "
                          f"投影仪 {err['projector_error']:.3f}px, "
                          f"总平均 {err['total_error']:.3f}px")
        print("===============================================")
