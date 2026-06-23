# -*- coding: utf-8 -*-
"""
标定参数配置模块
-----------------------------
对标定过程中的关键参数进行集中管理，包括：
1. 棋盘格内角点行列数（pattern_size）
2. 棋盘格单个方格物理尺寸（square_size，单位mm）
3. 格雷码位数（graycode_bits）
4. 格雷码图像总数（graycode_total_imgs）
5. 重投影误差过滤阈值（reprojection_threshold）
6. 投影仪分辨率（projector_size）
"""

import os


class CalibrationConfig:
    """标定参数配置类，统一管理所有可配置参数"""

    PATTERN_CHESSBOARD = 'chessboard'
    PATTERN_CIRCLES_GRID = 'circles_grid'

    def __init__(self):
        # ---------- 标定板类型 ----------
        self.pattern_type = CalibrationConfig.PATTERN_CHESSBOARD

        # ---------- 棋盘格参数 ----------
        self.pattern_size = (11, 8)
        self.square_size = 10.0

        # ---------- 圆形网格参数 ----------
        self.circle_pattern_size = (10, 11)
        self.circle_spacing = 9.0

        # ---------- 格雷码参数 ----------
        self.graycode_bits = 11
        self.graycode_total_imgs = 4 * self.graycode_bits + 1

        # ---------- 投影仪参数 ----------
        self.projector_size = (1920, 1080)

        # ---------- 标定算法参数 ----------
        self.reprojection_threshold = 1.0

        # ---------- 性能优化 ----------
        self.downsample_max_width = 2000

        # ---------- 迭代终止条件 ----------
        self.subpix_criteria = (None, 30, 0.001)
        self.calib_criteria = (None, 50, 1e-5)
        self.stereo_criteria = (None, 50, 1e-5)

    @property
    def bits(self):
        """获取格雷码位数"""
        return self.graycode_bits

    @bits.setter
    def bits(self, value):
        """设置格雷码位数，同时更新图像总数"""
        self.graycode_bits = int(value)
        self.graycode_total_imgs = 4 * self.graycode_bits + 1

    def get_graycode_indices(self):
        """
        根据当前格雷码位数返回各类型图像的索引区间

        返回值：
            {
                'black': 0,                                 # 全黑背景图索引
                'h_positive': (1, bits),                    # 水平正格雷码
                'h_negative': (bits + 1, 2 * bits),         # 水平反格雷码
                'v_positive': (2 * bits + 1, 3 * bits),     # 垂直正格雷码
                'v_negative': (3 * bits + 1, 4 * bits),     # 垂直反格雷码
                'white': 4 * bits                           # 全白/棋盘格图
            }
        """
        N = self.graycode_bits
        return {
            'black': 0,
            'h_positive': (1, N),
            'h_negative': (N + 1, 2 * N),
            'v_positive': (2 * N + 1, 3 * N),
            'v_negative': (3 * N + 1, 4 * N),
            'white': 4 * N
        }

    def print_config(self):
        """打印当前配置信息"""
        print("========== 标定参数配置 ==========")
        print(f"标定板类型: {self.pattern_type}")
        if self.pattern_type == CalibrationConfig.PATTERN_CHESSBOARD:
            print(f"棋盘格内角点数目: {self.pattern_size} (列 x 行)")
            print(f"棋盘格方格物理尺寸: {self.square_size} mm")
        elif self.pattern_type == CalibrationConfig.PATTERN_CIRCLES_GRID:
            print(f"圆形网格圆心数目: {self.circle_pattern_size} (列 x 行)")
            print(f"圆心间距: {self.circle_spacing} mm")
        print(f"格雷码位数: {self.graycode_bits}")
        print(f"每组位姿图像总数: {self.graycode_total_imgs} 张")
        print(f"投影仪分辨率: {self.projector_size}")
        print(f"重投影误差过滤阈值: {self.reprojection_threshold} 像素")
        print("=================================")


# 全局配置实例
_global_config = None


def get_default_config():
    """获取全局默认配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = CalibrationConfig()
    return _global_config


# 图像文件扩展名列表（按优先级）
IMAGE_EXTENSIONS = ['.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff']
