# -*- coding: utf-8 -*-
"""
棋盘格角点检测模块
-----------------------------
功能：
1. 在棋盘格内角点检测（cv2.findChessboardCorners）
2. 亚像素精细化角点（cv2.cornerSubPix）
3. 生成世界坐标系下的三维坐标点集
4. 角点检测可视化
"""

import numpy as np
import cv2


class ChessboardDetector:
    """棋盘格角点检测器"""

    def __init__(self, config):
        """
        初始化检测器

        参数：
            config: CalibrationConfig 配置实例
        """
        self.config = config

    # ==================== 世界坐标生成 ====================

    def generate_object_points(self):
        """
        生成棋盘格在世界坐标系下的三维坐标点集

        参考MATLAB标定工具箱的做法：
        棋盘格放置于Z=0平面，原点位于左上角第一个角点，
        X轴沿水平方向(向右)，Y轴沿垂直方向(向下)

        返回值：
            objp: np.ndarray, shape=(n_points, 3), dtype=float32
                  每一行是一个角点的 (X, Y, 0.0) 坐标
        """
        # 获取棋盘格参数
        cols, rows = self.config.pattern_size
        square_size = self.config.square_size

        # 生成网格坐标
        # 使用 mgrid 生成规则的坐标矩阵
        # 参考MATLAB: [X, Y] = meshgrid(0:cols-1, 0:rows-1)
        X, Y = np.mgrid[0:cols, 0:rows]

        # 转换为点列表 [[x1,y1], [x2,y2], ...]
        # np.mgrid[0:cols, 0:rows] 的维度：
        # 第一维度 (rows, cols)，第二维度 (rows, cols)
        # 需要转置后展平为 (cols*rows, 2)
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

        return objp

    # ==================== 角点检测 ====================

    def detect_corners(self, image):
        """
        检测单张棋盘格图像的内角点

        对大图像自动下采样加速角点检测，然后将结果映射回原始分辨率
        再做亚像素精细化。

        参数：
            image: 灰度图 (HxW np.uint8)

        返回值：
            ret: bool, 是否成功检测
            corners: np.ndarray, shape=(n_points, 1, 2), dtype=float32, 亚像素精细角点
                     失败则返回 None
        """
        if image is None:
            return False, None

        cols, rows = self.config.pattern_size
        pattern_size_tuple = (cols, rows)

        # 亚像素角点精细迭代终止条件
        subpix_iter = self.config.subpix_criteria[1]
        subpix_eps = self.config.subpix_criteria[2]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    subpix_iter, subpix_eps)

        # 棋盘格角点检测标志位
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FILTER_QUADS)

        h, w = image.shape[:2]
        max_w = getattr(self.config, 'downsample_max_width', 0)

        img_for_detect = image
        scale = 1.0

        # 大图像下采样加速
        if max_w > 0 and w > max_w:
            scale = float(max_w) / float(w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            img_for_detect = cv2.resize(image, (new_w, new_h),
                                       interpolation=cv2.INTER_AREA)

        # 1. 粗略角点检测（在可能下采样后的图像上）
        ret, corners = cv2.findChessboardCorners(img_for_detect,
                                                pattern_size_tuple,
                                                flags=flags)

        if not ret:
            return False, None

        # 2. 若图像被下采样，将角点映射回原始分辨率
        if scale != 1.0:
            corners = corners / float(scale)

        # 3. 亚像素角点精细化（在原始图像上）
        corners_refined = cv2.cornerSubPix(
            image,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        return True, corners_refined

    # ==================== 可视化 ====================

    def draw_corners(self, image, corners, ret=True):
        """
        在棋盘格图像上绘制检测到的角点

        参数：
            image: 原始图像（灰度或彩色）
            corners: 检测到的角点数组
            ret: 是否检测成功

        返回值：
            result_img: 绘制完成的彩色图像
        """
        # 转换为彩色图以便彩色绘制
        if len(image.shape) == 2:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            result = image.copy()

        cols, rows = self.config.pattern_size
        pattern_size_tuple = (cols, rows)

        cv2.drawChessboardCorners(result, pattern_size_tuple, corners, ret)

        return result

    # ==================== 批量检测 ====================

    def detect_all_poses(self, image_loader, progress_callback=None):
        """
        对所有有效位姿进行棋盘格角点检测

        参数：
            image_loader: ImageLoader 实例，包含已加载的图像数据
            progress_callback: 进度回调函数

        返回值：
            all_corners: dict {pose_name -> (ret, corners)}
        """
        all_corners = {}
        pose_names = image_loader.pose_names
        total = len(pose_names)

        for i, pose_name in enumerate(pose_names):
            # 获取全白棋盘格图像
            white_img = image_loader.get_white_image(pose_name)

            if white_img is None:
                all_corners[pose_name] = (False, None)
            else:
                ret, corners = self.detect_corners(white_img)
                all_corners[pose_name] = (ret, corners)

            if progress_callback is not None:
                try:
                    status = "成功" if all_corners[pose_name][0] else "失败"
                    progress_callback(i + 1, total, f"检测 {pose_name}: {status}")
                except Exception:
                    pass

        return all_corners
