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

    def __init__(self):
        # ---------- 棋盘格参数 ----------
        # 棋盘格内角点数目（列，行），参考MATLAB标定工具箱默认参数
        self.pattern_size = (11, 8)          # (columns, rows) = (11, 8) 内角点
        # 单个棋盘格物理尺寸（单位：毫米 mm）
        self.square_size = 10.0               # 方格边长 15mm

        # ---------- 格雷码参数 ----------
        # 格雷码位数（建议值：8-12，根据投影仪分辨率选择）
        # 1920x1080 分辨率时，垂直方向1920=2^10.9，使用11位水平格雷码
        # 水平方向1080=2^10.07，使用11位垂直格雷码
        self.graycode_bits = 11               # 格雷码位数

        # 每姿图像总张数：
        #   0                   : 全黑背景图
        #   1 ~ bits            : 水平正格雷码
        #   bits+1 ~ 2*bits    : 水平反格雷码
        #   2*bits+1 ~ 3*bits  : 垂直正格雷码
        #   3*bits+1 ~ 4*bits  : 垂直反格雷码（同时，最后一张作为全白棋盘格图）
        # 共计 4 * bits + 1 张图像
        self.graycode_total_imgs = 4 * self.graycode_bits + 1

        # 投影仪物理分辨率（宽，高），像素单位
        self.projector_size = (1920, 1080)    # 1080P投影仪分辨率

        # ---------- 标定算法参数 ----------
        # 重投影误差过滤阈值，超过此阈值的图像视为无效标定图像
        self.reprojection_threshold = 1.0     # 单位：像素

        # ---------- 性能优化 ----------
        # 大图像下采样宽高阈值。若图像长边>此阈值，将按比例下采样以加速
        # 0 表示不下采样；建议 1500~2500
        self.downsample_max_width = 2000

        # 亚像素角点迭代终止条件
        # 参考MATLAB的criteria，最大迭代30次，收敛阈值0.001像素
        self.subpix_criteria = (None, 30, 0.001)

        # 单目标定的迭代终止条件
        self.calib_criteria = (
            None,
            50,
            1e-5
        )

        # 双目标定迭代终止条件
        self.stereo_criteria = (
            None,
            50,
            1e-5
        )

        # ---------- 图像文件路径规则 ----------
        # 每组位姿下的图像文件命名规则：
        # 0.bmp          : 全黑背景图（black）
        # 1 ~ N.bmp      : 水平正格雷码（horizontal_positive）
        # N+1 ~ 2N.bmp   : 水平反格雷码（horizontal_negative）
        # 2N+1 ~ 3N.bmp  : 垂直正格雷码（vertical_positive）
        # 3N+1 ~ 4N.bmp  : 垂直反格雷码（vertical_negative）
        # (4N+1).bmp     : 全白棋盘格图（white / chessboard）
        # 其中 N = graycode_bits

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
        print(f"棋盘格内角点数目: {self.pattern_size} (列 x 行)")
        print(f"棋盘格方格物理尺寸: {self.square_size} mm")
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
