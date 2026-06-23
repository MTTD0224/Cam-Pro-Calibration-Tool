# -*- coding: utf-8 -*-
"""验证格雷码解码值是否需要缩放"""
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

loader = ImageLoader(cfg)
loader.load_all_poses(root)
decoder = GraycodeDecoder(cfg)

# 只分析第一个位姿
pose_name = loader.pose_names[0]
compensated = loader.apply_light_compensation(pose_name)

# 手动解码 - 不用投影仪范围过滤
h_pos = compensated['h_positive']
h_neg = compensated['h_negative']
v_pos = compensated['v_positive']
v_neg = compensated['v_negative']

bits = cfg.graycode_bits
proj_w, proj_h = cfg.projector_size

def decode_raw(positive_imgs, negative_imgs):
    h = positive_imgs[0].shape[0]
    w = positive_imgs[0].shape[1]

    diff_tensor = np.zeros((bits, h, w), dtype=np.float32)
    for i in range(bits):
        diff_tensor[i] = positive_imgs[i].astype(np.float32) - negative_imgs[i].astype(np.float32)

    # 格雷码 -> 二进制
    binary_matrix = (diff_tensor > 0).astype(np.int32)
    for i in range(1, bits):
        binary_matrix[i] = binary_matrix[i - 1] ^ binary_matrix[i]

    # 二进制 -> 十进制
    weights = np.array([1 << (bits - i - 1) for i in range(bits)], dtype=np.int32)
    weights = weights.reshape(bits, 1, 1)
    decoded = np.sum(binary_matrix * weights, axis=0).astype(np.float32)
    return decoded

proj_col_raw = decode_raw(h_pos, h_neg)
proj_row_raw = decode_raw(v_pos, v_neg)

print("=" * 60)
print("原始解码值分布统计")
print("=" * 60)

print(f"\n水平解码值统计:")
print(f"  范围: [{proj_col_raw.min():.0f}, {proj_col_raw.max():.0f}] (理论: 0 - {2**bits - 1})")
print(f"  均值: {proj_col_raw.mean():.1f}")
print(f"  < 0 的像素: {np.sum(proj_col_raw < 0)}")
print(f"  >= {proj_w} 的像素: {np.sum(proj_col_raw >= proj_w)} ({100 * np.sum(proj_col_raw >= proj_w) / proj_col_raw.size:.2f}%)")
print(f"  >= {2**bits} 的像素: {np.sum(proj_col_raw >= 2**bits)}")

print(f"\n垂直解码值统计:")
print(f"  范围: [{proj_row_raw.min():.0f}, {proj_row_raw.max():.0f}] (理论: 0 - {2**bits - 1})")
print(f"  均值: {proj_row_raw.mean():.1f}")
print(f"  < 0 的像素: {np.sum(proj_row_raw < 0)}")
print(f"  >= {proj_h} 的像素: {np.sum(proj_row_raw >= proj_h)} ({100 * np.sum(proj_row_raw >= proj_h) / proj_row_raw.size:.2f}%)")

# 方案A: 直接用0-1919, 0-1079 范围
valid_a = np.logical_and(
    np.logical_and(proj_col_raw >= 0, proj_col_raw < proj_w),
    np.logical_and(proj_row_raw >= 0, proj_row_raw < proj_h)
)
print(f"\n方案 A (原始值直接当像素坐标): 有效率 = {np.sum(valid_a) / valid_a.size:.2%}")

# 方案B: 缩放解码值到投影仪分辨率
proj_col_scaled = proj_col_raw * proj_w / (2**bits)
proj_row_scaled = proj_row_raw * proj_h / (2**bits)
valid_b = np.logical_and(
    np.logical_and(proj_col_scaled >= 0, proj_col_scaled < proj_w),
    np.logical_and(proj_row_scaled >= 0, proj_row_scaled < proj_h)
)
print(f"方案 B (缩放 {proj_w}/{2**bits} = {proj_w / (2**bits):.6f}): 有效率 = {np.sum(valid_b) / valid_b.size:.2%}")
print(f"  水平缩放后: [{proj_col_scaled.min():.1f}, {proj_col_scaled.max():.1f}]")
print(f"  垂直缩放后: [{proj_row_scaled.min():.1f}, {proj_row_scaled.max():.1f}]")

# 方案C: 用2^N值作为投影仪像素，但需要确保小于等于分辨率
proj_col_c = proj_col_raw  # 不缩放
proj_row_c = proj_row_raw
valid_c = np.logical_and(
    np.logical_and(proj_col_c >= 0, proj_col_c < 2**bits),
    np.logical_and(proj_row_c >= 0, proj_row_c < 2**bits)
)
# 但我们也需要有效的角点映射
print(f"\n方案 C (不缩放, 仅0-{2**bits-1}): 有效率 = {np.sum(valid_c) / valid_c.size:.2%}")

# 检查角点位置对应的值
print(f"\n=== 摄像机检测到的角点对应的解码值 ===")
from chessboard_detector import ChessboardDetector
detector = ChessboardDetector(cfg)
white = loader.get_white_image(pose_name)
ret, corners = detector.detect_corners(white)
if ret:
    print(f"检测到 {len(corners)} 个角点")
    for idx in range(min(len(corners), 10)):  # 只看前10个角点
        col = int(round(corners[idx, 0, 0]))
        row = int(round(corners[idx, 0, 1]))
        if 0 <= row < proj_col_raw.shape[0] and 0 <= col < proj_col_raw.shape[1]:
            col_val_a = proj_col_raw[row, col]
            row_val_a = proj_row_raw[row, col]
            # 双线性插值
            col_val_b = decoder._bilinear_lookup(proj_col_raw, corners[idx, 0, 0], corners[idx, 0, 1])
            row_val_b = decoder._bilinear_lookup(proj_row_raw, corners[idx, 0, 0], corners[idx, 0, 1])
            print(f"  角点 {idx}: 相机({corners[idx,0,0]:.1f}, {corners[idx, 0, 1]:.1f})  ->  原始解码值({col_val_b:.1f}, {row_val_b:.1f}), 缩放后({col_val_b * proj_w / 2**bits:.1f}, {row_val_b * proj_h / 2**bits:.1f})")

print("\n=== 结论 ===")
print("从原始解码值来看，需要将解码值从 0-{} 缩放到投影仪像素范围 0-{} 和 0-{}".format(2**bits - 1, proj_w, proj_h))
