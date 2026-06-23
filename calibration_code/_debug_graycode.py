# -*- coding: utf-8 -*-
"""格雷码解码诊断脚本 - 分析为什么解码有效率低"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

from config import CalibrationConfig
from image_loader import ImageLoader
from graycode_decoder import GraycodeDecoder

root = r"d:\2 Program Code\CameraCalibration\calibration_img"

cfg = CalibrationConfig()
cfg.pattern_size = (11, 8)
cfg.square_size = 10.0
cfg.graycode_bits = 11
cfg.projector_size = (1920, 1080)

print("=" * 70)
print("格雷码解码诊断")
print("=" * 70)

# 1. 扫描并加载图像
loader = ImageLoader(cfg)
loader.load_all_poses(root)
print(f"发现位姿: {loader.pose_names}")
print(f"每姿图像数: {cfg.graycode_total_imgs} (索引 0-{cfg.graycode_total_imgs - 1})")
print()

# 2. 检查图像索引
indices = cfg.get_graycode_indices()
print("索引分布:")
print(f"  black = {indices['black']}")
print(f"  h_positive = {indices['h_positive']}")
print(f"  h_negative = {indices['h_negative']}")
print(f"  v_positive = {indices['v_positive']}")
print(f"  v_negative = {indices['v_negative']}")
print(f"  white = {indices['white']}")
print()

# 3. 针对前3个位姿做详细分析
decoder = GraycodeDecoder(cfg)
for pose_name in loader.pose_names[:3]:
    print("-" * 60)
    print(f"位姿: {pose_name}")

    # 加载原始图像
    black = loader.get_black_image(pose_name)
    white = loader.get_white_image(pose_name)
    h_pos, h_neg = loader.get_graycode_images(pose_name, 'horizontal')
    v_pos, v_neg = loader.get_graycode_images(pose_name, 'vertical')

    print(f"  black 图像可用: {black is not None}, 形状: {black.shape if black is not None else 'N/A'}, 均值: {black.mean() if black is not None else 'N/A':.1f}")
    print(f"  white 图像可用: {white is not None}, 形状: {white.shape if white is not None else 'N/A'}, 均值: {white.mean() if white is not None else 'N/A':.1f}")
    print(f"  h_pos 张数: {len([x for x in h_pos if x is not None])}/{len(h_pos)}")
    print(f"  h_neg 张数: {len([x for x in h_neg if x is not None])}/{len(h_neg)}")
    print(f"  v_pos 张数: {len([x for x in v_pos if x is not None])}/{len(v_pos)}")
    print(f"  v_neg 张数: {len([x for x in v_neg if x is not None])}/{len(v_neg)}")

    # 检查 white-black 的差值（理论上应较大）
    if black is not None and white is not None:
        diff = white.astype(np.float32) - black.astype(np.float32)
        print(f"  white-black 差值: 均值={diff.mean():.2f}, 中位数={np.median(diff):.2f}, 最小值={diff.min():.2f}, 最大值={diff.max():.2f}")

    # 检查水平正/反格雷码的差值（逐位）
    if all(x is not None for x in h_pos) and all(x is not None for x in h_neg):
        bits = cfg.graycode_bits
        print(f"  水平格雷码逐位差值统计 (正-反):")
        for i in range(min(bits, 5)):  # 只打印前5位
            diff = h_pos[i].astype(np.float32) - h_neg[i].astype(np.float32)
            # 统计正负比例
            pos_ratio = np.sum(diff > 0) / diff.size
            neg_ratio = np.sum(diff < 0) / diff.size
            near_zero = np.sum(np.abs(diff) < 10) / diff.size
            print(f"    位 {i + 1}/{bits}: 均值={diff.mean():+.2f}, 绝对值均值={np.abs(diff).mean():.2f}, 正占比={pos_ratio:.2%}, 负占比={neg_ratio:.2%}, 接近零(<10)占比={near_zero:.2%}")

    # 尝试解码（带各种阈值）
    print()
    print(f"  --- 尝试不同解码阈值 ---")
    compensated = loader.apply_light_compensation(pose_name)

    # 手动做解码，测试不同阈值
    bits = cfg.graycode_bits
    # 水平
    h_pos_c = compensated['h_positive']
    h_neg_c = compensated['h_negative']
    if h_pos_c[0] is not None and h_neg_c[0] is not None:
        h, w = h_pos_c[0].shape
        # 计算所有位的差分矩阵
        diff_tensor = np.zeros((bits, h, w), dtype=np.float32)
        for i in range(bits):
            if h_pos_c[i] is not None and h_neg_c[i] is not None:
                diff_tensor[i] = h_pos_c[i].astype(np.float32) - h_neg_c[i].astype(np.float32)

        abs_sum = np.sum(np.abs(diff_tensor), axis=0)  # (h, w)

        # 尝试不同阈值
        for t_val in [50, 100, 200, 500, 1000, bits * 10, bits * 20, bits * 50]:
            valid = np.sum(abs_sum >= t_val)
            ratio = valid / abs_sum.size
            print(f"    水平 | 差分和阈值={t_val:.0f}: 有效像素={valid/1e6:.2f}M ({ratio:.2%})")

        print(f"    水平 | 差分和统计: 均值={abs_sum.mean():.1f}, 中位数={np.median(abs_sum):.1f}, 5%分位={np.percentile(abs_sum, 5):.1f}, 10%分位={np.percentile(abs_sum, 10):.1f}")

    # 尝试解码完整
    try:
        decode_result = decoder.decode_pose(compensated)
        total = decode_result['total_pixels']
        valid = decode_result['valid_pixels']
        print(f"  当前解码器有效率: {valid}/{total} = {valid / total:.2%}")

        # 检查解码出来的值是否合理
        proj_col = decode_result['proj_col']
        proj_row = decode_result['proj_row']
        valid_mask = decode_result['valid_mask']
        if np.any(valid_mask):
            valid_cols = proj_col[valid_mask]
            valid_rows = proj_row[valid_mask]
            print(f"  有效投影列范围: [{valid_cols.min():.1f}, {valid_cols.max():.1f}] (投影仪宽={cfg.projector_size[0]})")
            print(f"  有效投影行范围: [{valid_rows.min():.1f}, {valid_rows.max():.1f}] (投影仪高={cfg.projector_size[1]})")
    except Exception as e:
        print(f"  解码失败: {e}")

    print()

print("=" * 70)
print("诊断完成")
print("=" * 70)
