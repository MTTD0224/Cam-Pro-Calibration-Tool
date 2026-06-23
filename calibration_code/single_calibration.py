# -*- coding: utf-8 -*-
"""
单目标定模块
-----------------------------
功能：
1. 基于张氏标定法（Zhang's Method）进行单目相机/投影仪标定
2. 使用 cv2.calibrateCamera 求解内参矩阵、畸变系数、外参
3. 统计每幅图像的重投影误差，并过滤超阈值图像
4. 评估标定精度（平均重投影误差）

参考 MATLAB Camera Calibrator Toolbox 的流程
"""

import numpy as np
import cv2


class SingleCalibration:
    """单目标定器，用于相机或投影仪的单目标定"""

    def __init__(self, config, camera_name="camera"):
        """
        初始化单目标定器

        参数：
            config: CalibrationConfig 配置实例
            camera_name: 字符串，标识此次标定的设备名（'camera' 或 'projector'）
        """
        self.config = config
        self.camera_name = camera_name

        # 标定结果（初始化为 None）
        self.ret = None           # 标定返回值（rms 重投影误差的平方根）
        self.mtx = None           # 内参矩阵 (3x3)
        self.dist = None          # 畸变系数 (1x5) 或 (1x8, 1x12, 1x14)
        self.rvecs = []           # 每张图像的旋转向量列表
        self.tvecs = []           # 每张图像的平移向量列表
        self.reprojection_errors = []  # 每张图像的重投影误差

        # 用于标定的输入数据
        self.object_points = []   # 世界坐标点集（每组位姿一份）
        self.image_points = []    # 图像二维点集（每组位姿一份）
        self.image_size = None    # 图像尺寸 (width, height)

        # 位姿有效性记录
        self.valid_pose_names = []   # 成功参与标定的位姿名称
        self.filtered_pose_names = []  # 被过滤（误差过大）的位姿名称
        self.per_pose_error = {}     # 每个位姿的重投影误差

    # ==================== 数据准备 ====================

    def prepare_data(self, obj_points_list, img_points_list, pose_names, image_size):
        """
        准备标定所需的数据

        参数：
            obj_points_list: list of np.ndarray, 每组位姿的世界三维坐标
            img_points_list: list of np.ndarray, 每组位姿的图像二维坐标
            pose_names: list of str, 每个位姿的名称
            image_size: tuple (width, height), 图像尺寸
        """
        self.object_points = []
        self.image_points = []
        self.valid_pose_names = []

        for objp, imgp, name in zip(obj_points_list, img_points_list, pose_names):
            if objp is None or imgp is None:
                continue
            self.object_points.append(np.asarray(objp, dtype=np.float32))
            self.image_points.append(np.asarray(imgp, dtype=np.float32))
            self.valid_pose_names.append(name)

        self.image_size = image_size

        if len(self.object_points) < 3:
            raise RuntimeError(f"有效位姿数量不足（当前 {len(self.object_points)}，至少需要 3 幅）")

    # ==================== 执行标定 ====================

    def calibrate(self, fix_k3=False):
        """
        执行单目标定（cv2.calibrateCamera）

        内参矩阵格式:
            [[ fx   0   cx ],
             [  0   fy  cy ],
             [  0    0    1 ]]

        畸变系数格式 (默认5参数):
            [k1, k2, p1, p2, k3]

        参数：
            fix_k3: bool, 是否固定 k3=0（对于普通镜头可以设置为True加速）

        返回：
            success: bool, 标定是否成功
        """
        if len(self.object_points) == 0:
            raise RuntimeError("尚未准备标定数据，请先调用 prepare_data()")

        # 标定迭代终止条件
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                    self.config.calib_criteria[1],
                    self.config.calib_criteria[2])

        # 构造 flags
        flags = 0
        if fix_k3:
            flags |= cv2.CALIB_FIX_K3

        try:
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.object_points,
                self.image_points,
                self.image_size,
                None, None,
                criteria=criteria,
                flags=flags
            )
        except cv2.error as e:
            raise RuntimeError(f"OpenCV 标定失败: {str(e)}")

        # 计算每张图像的重投影误差
        self._compute_per_pose_error()

        return True

    def _compute_per_pose_error(self):
        """
        计算并记录每张图像的重投影误差

        公式：
            image_points_proj = cv2.projectPoints(obj_points, rvec, tvec, mtx, dist)
            error = mean( || image_points - image_points_proj ||_2 )
        """
        self.reprojection_errors = []
        self.per_pose_error = {}

        for i, (objp, imgp, name) in enumerate(zip(
                self.object_points, self.image_points, self.valid_pose_names)):
            # 将世界点投影回图像
            imgp_proj, _ = cv2.projectPoints(
                objp, self.rvecs[i], self.tvecs[i], self.mtx, self.dist)

            # 计算欧氏距离
            error = cv2.norm(imgp, imgp_proj, cv2.NORM_L2) / len(imgp_proj)
            self.reprojection_errors.append(error)
            self.per_pose_error[name] = error

    # ==================== 误差过滤 ====================

    def filter_by_reprojection_error(self, threshold=None):
        """
        按重投影误差过滤图像，过滤掉误差过大的图像，并重新进行标定

        参数：
            threshold: float, 过滤阈值（像素），None 则使用配置中的值

        返回：
            filtered: list of str, 被过滤掉的位姿名称列表
        """
        if threshold is None:
            threshold = self.config.reprojection_threshold

        if len(self.reprojection_errors) == 0:
            return []

        # 收集需要保留的位姿
        keep_obj_points = []
        keep_img_points = []
        keep_names = []
        filtered = []

        for i, name in enumerate(self.valid_pose_names):
            if self.reprojection_errors[i] > threshold:
                filtered.append(name)
            else:
                keep_obj_points.append(self.object_points[i])
                keep_img_points.append(self.image_points[i])
                keep_names.append(name)

        if len(keep_obj_points) < 3:
            print(f"[警告] 过滤后剩余位姿数量不足（当前 {len(keep_obj_points)} < 3），跳过过滤")
            self.filtered_pose_names = []
            return []

        # 重新标定
        self.object_points = keep_obj_points
        self.image_points = keep_img_points
        self.valid_pose_names = keep_names
        self.filtered_pose_names = filtered

        # 重新执行标定
        self.calibrate()

        return filtered

    # ==================== 精度评估 ====================

    def get_mean_reprojection_error(self):
        """获取标定的平均重投影误差（所有保留图像的均值）"""
        if len(self.reprojection_errors) == 0:
            return -1.0
        return float(np.mean(self.reprojection_errors))

    def get_max_reprojection_error(self):
        """获取最大重投影误差"""
        if len(self.reprojection_errors) == 0:
            return -1.0
        return float(np.max(self.reprojection_errors))

    def get_error_statistics(self):
        """获取完整的误差统计信息"""
        if len(self.reprojection_errors) == 0:
            return {}
        errors = np.array(self.reprojection_errors)
        return {
            'mean': float(np.mean(errors)),
            'std': float(np.std(errors)),
            'min': float(np.min(errors)),
            'max': float(np.max(errors)),
            'median': float(np.median(errors)),
            'count': int(len(errors)),
            'total_rms': float(self.ret) if self.ret is not None else -1.0
        }

    # ==================== 结果获取 ====================

    def get_calibration_result(self):
        """
        获取完整的标定结果

        返回：
            result: dict
                {
                    'mtx': 内参矩阵 (3x3),
                    'dist': 畸变系数向量,
                    'mean_error': 平均重投影误差,
                    'rvecs': 旋转向量列表,
                    'tvecs': 平移向量列表,
                    'image_size': 图像尺寸 (w, h),
                    'per_pose_error': 每位姿误差字典,
                    'statistics': 误差统计信息
                }
        """
        return {
            'mtx': self.mtx,
            'dist': self.dist,
            'mean_error': self.get_mean_reprojection_error(),
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
            'image_size': self.image_size,
            'per_pose_error': self.per_pose_error,
            'statistics': self.get_error_statistics(),
            'valid_poses': self.valid_pose_names,
            'filtered_poses': self.filtered_pose_names,
            'name': self.camera_name
        }

    def print_result(self):
        """打印标定结果（类似MATLAB输出格式）"""
        print(f"========== {self.camera_name} 单目标定结果 ==========")
        print(f"图像尺寸: {self.image_size}")
        print(f"参与标定的位姿数: {len(self.valid_pose_names)}")
        if self.filtered_pose_names:
            print(f"被过滤的位姿: {self.filtered_pose_names}")

        if self.mtx is not None:
            print("\n内参矩阵:")
            print(f"  fx = {self.mtx[0, 0]:.4f},  fy = {self.mtx[1, 1]:.4f}")
            print(f"  cx = {self.mtx[0, 2]:.4f},  cy = {self.mtx[1, 2]:.4f}")
            print(f"  完整矩阵:\n{self.mtx}")

        if self.dist is not None:
            print(f"\n畸变系数 (k1, k2, p1, p2, k3, ...):\n{self.dist}")

        stats = self.get_error_statistics()
        if stats:
            print(f"\n重投影误差统计:")
            print(f"  总RMS误差: {stats['total_rms']:.4f} 像素")
            print(f"  平均误差: {stats['mean']:.4f} 像素")
            print(f"  标准差: {stats['std']:.4f} 像素")
            print(f"  最小值: {stats['min']:.4f} 像素")
            print(f"  最大值: {stats['max']:.4f} 像素")
        print("==================================================")
