# -*- coding: utf-8 -*-
"""
相机-投影仪标定工作流整合模块
-----------------------------
将图像加载、角点检测、格雷码解码、单目标定、双目标定、
参数保存等功能整合为一个完整的工作流

为GUI和命令行调用提供统一接口
"""

import os
import numpy as np
import cv2
from config import CalibrationConfig, get_default_config

from image_loader import ImageLoader
from chessboard_detector import ChessboardDetector
from graycode_decoder import GraycodeDecoder
from single_calibration import SingleCalibration
from stereo_calibration import StereoCalibration
from params_saver import ParamSaver


class CalibrationPipeline:
    """
    相机-投影仪联合标定工作流

    使用示例:
        pipeline = CalibrationPipeline(config)
        result = pipeline.run(root_dir, output_dir)
    """

    def __init__(self, config=None, log_callback=None, progress_callback=None):
        """
        初始化标定工作流

        参数:
            config: CalibrationConfig 配置，None 则使用默认
            log_callback: function(msg) 日志回调
            progress_callback: function(percent, msg) 进度回调
        """
        self.config = config if config is not None else get_default_config()
        self.log = log_callback if log_callback is not None else self._default_log
        self.progress = progress_callback if progress_callback is not None else self._default_progress

        # 各模块实例
        self.loader = ImageLoader(self.config)
        self.detector = ChessboardDetector(self.config)
        self.decoder = GraycodeDecoder(self.config)
        self.camera_calib = SingleCalibration(self.config, "camera")
        self.projector_calib = SingleCalibration(self.config, "projector")
        self.stereo_calib = StereoCalibration(self.config)
        self.saver = ParamSaver()

        # 中间结果
        self.all_camera_corners = {}   # pose_name -> (ret, corners)
        self.all_projector_corners = {}  # pose_name -> (corners, valid_flags)
        self.object_points_template = None
        self.stereo_result = None
        self._pose_list = []

    def _default_log(self, msg):
        print(msg)

    def _default_progress(self, percent, msg):
        print(f"[{int(percent):3d}% - {msg}")

    # ==================== 1. 加载图像 ====================

    def step_load_images(self, root_dir):
        """步骤1：加载根目录下所有位姿图像"""
        self.log("[步骤 1/6] 正在加载标定图像...")
        self.loader.load_all_poses(root_dir, progress_callback=None)
        summary = self.loader.get_summary()
        self.log(f"    共加载 {summary['total']} 组位姿，"
                f"每组 {summary['images_per_pose']} 张图像")
        invalid_list = self.loader.get_invalid_poses()
        if invalid_list:
            self.log(f"    警告：{len(invalid_list)} 组位姿图像数量不匹配，已跳过")
        self.progress(10, "图像加载完成")
        return True

    # ==================== 2. 相机角点检测 ====================

    def step_detect_corners(self):
        """步骤2：检测棋盘格角点（用于相机标定）"""
        self.log("[步骤 2/6] 正在检测棋盘格角点...")
        corner_results = {}
        pose_names = self.loader.pose_names
        total = len(pose_names)
        success = 0

        for i, pose_name in enumerate(pose_names):
            white_img = self.loader.get_white_image(pose_name)
            if white_img is None:
                corner_results[pose_name] = (False, None)
                continue
            ret, corners = self.detector.detect_corners(white_img)
            corner_results[pose_name] = (ret, corners)
            if ret:
                success += 1
            self.progress(10 + 10 * (i + 1) / max(total, 1),
                          f"角点检测: {i + 1}/{total}")

        self.all_camera_corners = corner_results
        self.log(f"    角点检测完成：{success}/{total} 组成功")
        if success == 0:
            raise RuntimeError("所有图像的角点检测都失败了，请检查图像")
        self.progress(20, "角点检测完成")
        return True

    # ==================== 3. 相机单目标定 ====================

    def step_camera_calibration(self):
        """步骤3：相机单目标定"""
        self.log("[步骤 3/6] 正在执行相机单目标定...")
        objp_template = self.detector.generate_object_points()
        self.object_points_template = objp_template

        obj_points_list = []
        img_points_list = []
        pose_names = []
        image_size = None

        for pose_name, (ret, corners) in self.all_camera_corners.items():
            if not ret:
                continue
            obj_points_list.append(objp_template.copy())
            img_points_list.append(corners)
            pose_names.append(pose_name)
            white = self.loader.get_white_image(pose_name)
            if white is not None and image_size is None:
                h, w = white.shape[:2]
                image_size = (w, h)

        if len(obj_points_list) == 0:
            raise RuntimeError("无有效角点数据，无法进行相机标定")

        self.camera_calib.prepare_data(obj_points_list, img_points_list,
                                    pose_names, image_size)
        self.camera_calib.calibrate()
        filtered = self.camera_calib.filter_by_reprojection_error()
        stats = self.camera_calib.get_error_statistics()
        self.log(f"    相机平均重投影误差: {stats['mean']:.4f} 像素")
        self.log(f"    参与标定的位姿数: {stats['count']}")
        if filtered:
            self.log(f"    被过滤位姿: {filtered}")
        self.progress(35, "相机标定完成")
        return True

    # ==================== 4. 格雷码解码（投影仪角点） ====================

    def step_decode_graycode(self):
        """步骤4：格雷码解码，得到投影仪角点"""
        self.log("[步骤 4/6] 正在执行格雷码解码...")
        pose_names = self.loader.pose_names
        total = len(pose_names)
        success = 0

        self.all_projector_corners = {}

        # 只对相机角点检测成功的位姿进行解码
        valid_pose_count = 0
        total_valid = 0

        for i, pose_name in enumerate(pose_names):
            ret, corners = self.all_camera_corners.get(pose_name, (False, None))
            if not ret:
                continue
            total_valid += 1

        for i, pose_name in enumerate(pose_names):
            ret, corners = self.all_camera_corners.get(pose_name, (False, None))
            if not ret:
                continue

            try:
                compensated = self.loader.apply_light_compensation(pose_name)
                decode_result = self.decoder.decode_pose(compensated)
                proj_corners, valid_flags = self.decoder.lookup_corners(
                    decode_result['proj_col'],
                    decode_result['proj_row'],
                    corners)
                if proj_corners is None or not np.any(valid_flags):
                    self.all_projector_corners[pose_name] = (None, valid_flags)
                    continue
                # 如果有一些角点解码失败，则该位姿被标记为失败
                valid_ratio = float(np.sum(valid_flags)) / max(len(valid_flags), 1)
                if valid_ratio < 0.5:
                    self.all_projector_corners[pose_name] = (None, valid_flags)
                    self.log(f"    {pose_name} 解码有效率: {valid_ratio:.2%}，跳过")
                    continue

                self.all_projector_corners[pose_name] = (proj_corners, valid_flags)
                success += 1
            except Exception as e:
                self.log(f"    {pose_name} 解码失败: {str(e)}")
                self.all_projector_corners[pose_name] = (None, np.zeros(0, dtype=bool))

            pose_count = i + 1
            self.progress(35 + 20 * pose_count / max(total_valid, 1),
                          f"格雷码解码: {pose_count}/{total}")

        self.log(f"    格雷码解码完成：{success}/{total_valid} 组成功")
        if success == 0:
            raise RuntimeError("所有组的格雷码解码均失败，请检查格雷码图像")
        self.progress(55, "格雷码解码完成")
        return True

    # ==================== 5. 投影仪单目标定 ====================

    def step_projector_calibration(self):
        """步骤5：投影仪单目标定"""
        self.log("[步骤 5/6] 正在执行投影仪单目标定...")
        obj_points_list = []
        proj_points_list = []
        pose_names = []
        proj_w, proj_h = self.config.projector_size

        for pose_name, (proj_corners, valid_flags) in self.all_projector_corners.items():
            if proj_corners is None:
                continue
            # 仅保留同时有相机角点和投影仪角点都有效的位姿
            ret, cam_corners = self.all_camera_corners.get(pose_name, (False, None))
            if not ret:
                continue
            obj_points_list.append(self.object_points_template.copy())
            proj_points_list.append(proj_corners)
            pose_names.append(pose_name)

        if len(obj_points_list) < 3:
            raise RuntimeError(f"投影仪标定数据不足（当前 {len(obj_points_list)} < 3")

        self.projector_calib.prepare_data(
            obj_points_list, proj_points_list, pose_names, (proj_w, proj_h))
        self.projector_calib.calibrate()
        filtered = self.projector_calib.filter_by_reprojection_error()
        stats = self.projector_calib.get_error_statistics()
        self.log(f"    投影仪平均重投影误差: {stats['mean']:.4f} 像素")
        self.log(f"    参与标定的位姿数: {stats['count']}")
        if filtered:
            self.log(f"    被过滤位姿: {filtered}")
        self.progress(70, "投影仪标定完成")
        return True

    # ==================== 6. 双目标定 ====================

    def step_stereo_calibration(self):
        """步骤6：相机-投影仪双目标定"""
        self.log("[步骤 6/6] 正在执行相机-投影仪双目标定...")

        # 获取投影仪标定后剩余的有效位姿
        proj_valid = set(self.projector_calib.valid_pose_names)

        obj_points_list = []
        camera_points_list = []
        projector_points_list = []
        pose_names = []
        image_size = None

        for pose_name in proj_valid:
            ret, cam_corners = self.all_camera_corners.get(pose_name, (False, None))
            proj_corners, _ = self.all_projector_corners.get(pose_name, (None, None))
            if not ret or cam_corners is None or proj_corners is None:
                continue
            obj_points_list.append(self.object_points_template.copy())
            camera_points_list.append(cam_corners)
            projector_points_list.append(proj_corners)
            pose_names.append(pose_name)
            if image_size is None:
                white = self.loader.get_white_image(pose_name)
                if white is not None:
                    h, w = white.shape[:2]
                    image_size = (w, h)

        if len(obj_points_list) < 3:
            raise RuntimeError(f"双目标定数据不足（当前 {len(obj_points_list)} < 3")

        self.stereo_calib.prepare_data(
            obj_points_list, camera_points_list, projector_points_list,
            pose_names, image_size,
            self.camera_calib.mtx, self.camera_calib.dist,
            self.projector_calib.mtx, self.projector_calib.dist)

        self.stereo_calib.calibrate(fix_intrinsic=True)
        result = self.stereo_calib.get_calibration_result()
        self.stereo_result = result
        self.log(f"    双目标定全局 RMS 误差: {result['rms']:.4f} 像素")
        self.log(f"    平均重投影误差: {result['mean_error']:.4f} 像素")
        self.log(f"    参与标定位姿数: {result['pose_count']}")
        self.progress(90, "双目标定完成")
        return True

    # ==================== 保存结果 ====================

    def save_results(self, output_dir):
        """保存标定结果"""
        self.log("[保存] 正在保存标定 npy 文件...")
        self.saver.set_output_dir(output_dir)
        self.saver.save_all_params(self.stereo_result)
        self.log(f"    已保存到: {output_dir}")
        for f in self.saver.saved_files:
            self.log(f"      - {os.path.basename(f)}")
        self.progress(100, "标定完成")
        return True

    # ==================== 主运行 ====================

    def run(self, root_dir, output_dir):
        """运行完整标定流程"""
        try:
            self.log("=========== 开始相机-投影仪联合标定 ===========")
            self.log(f"标定根目录: {root_dir}")
            self.log(f"输出目录: {output_dir}")
            self.log(f"棋盘格: {self.config.pattern_size} 内角点, "
                    f"{self.config.square_size} mm/格")
            self.log(f"格雷码位数: {self.config.graycode_bits}")
            self.log(f"投影仪分辨率: {self.config.projector_size}")
            self.log(f"误差过滤阈值: {self.config.reprojection_threshold} 像素")
            self.log("===========================================")

            if not os.path.exists(root_dir):
                raise FileNotFoundError(f"根目录不存在: {root_dir}")

            self.step_load_images(root_dir)
            self.step_detect_corners()
            self.step_camera_calibration()
            self.step_decode_graycode()
            self.step_projector_calibration()
            self.step_stereo_calibration()
            self.save_results(output_dir)

            self.log("=========== 标定完成 ===========")
            return self.stereo_result
        except Exception as e:
            self.log(f"[错误] 标定失败: {str(e)}")
            raise


# ==================== 便捷函数 ====================

def run_calibration(root_dir, output_dir, config=None, log_cb=None, prog_cb=None):
    """
    便捷函数：执行完整标定流程

    :param root_dir: 标定图像根目录
    :param output_dir: 输出目录
    :param config: CalibrationConfig 实例（可选）
    :param log_cb: 日志回调
    :param prog_cb: 进度回调
    :return: 双目标定结果字典
    """
    pipeline = CalibrationPipeline(config, log_cb, prog_cb)
    return pipeline.run(root_dir, output_dir)


if __name__ == "__main__":
    # 命令行模式
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="相机-投影仪联合标定工具")
    parser.add_argument("--root", "-i", required=True, help="标定图像根目录")
    parser.add_argument("--output", "-o", required=True, help="输出目录")
    parser.add_argument("--cols", type=int, default=11, help="棋盘格内点列数")
    parser.add_argument("--rows", type=int, default=8, help="棋盘格内点行数")
    parser.add_argument("--square", type=float, default=10.0, help="方格边长(mm)")
    parser.add_argument("--bits", type=int, default=11, help="格雷码位数")
    parser.add_argument("--proj-w", type=int, default=1920, help="投影仪宽度")
    parser.add_argument("--proj-h", type=int, default=1080, help="投影仪高度")
    parser.add_argument("--threshold", type=float, default=0.3, help="重投影误差阈值")
    args = parser.parse_args()

    cfg = CalibrationConfig()
    cfg.pattern_size = (args.cols, args.rows)
    cfg.square_size = args.square
    cfg.graycode_bits = args.bits
    cfg.projector_size = (args.proj_w, args.proj_h)
    cfg.reprojection_threshold = args.threshold

    run_calibration(args.root, args.output, cfg)
