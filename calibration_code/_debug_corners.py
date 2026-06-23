# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np

# 检查一组位姿的棋盘格角点
folder = r"d:\2 Program Code\CameraCalibration\calibration_img\1"
# 索引 44 是全白/棋盘格图像（共 0-44=45 张）
img_path = os.path.join(folder, "44.bmp")
print(f"读取图像: {img_path}")
print(f"存在: {os.path.exists(img_path)}")

if not os.path.exists(img_path):
    # 也尝试 target.bmp
    img_path2 = os.path.join(folder, "target.bmp")
    print(f"尝试 target.bmp: {os.path.exists(img_path2)}")
    img_path = img_path2

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print(f"图像尺寸: {img.shape if img is not None else 'None'}")
if img is not None:
    print(f"像素范围: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")

# 尝试不同的角点数目
for pattern in [(11,8),(12,9),(10,7),(9,6),(8,5)]:
    ret, corners = cv2.findChessboardCorners(img, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FILTER_QUADS)
    print(f"pattern={pattern}: ret={ret}, corners={'found' if ret else 'none'}")

# 尝试 target.bmp
target = os.path.join(folder, "target.bmp")
if os.path.exists(target):
    print()
    print(f"读取 target.bmp...")
    img2 = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    print(f"尺寸: {img2.shape if img2 is not None else 'None'}")
    if img2 is not None:
        print(f"像素范围: min={img2.min()}, max={img2.max()}, mean={img2.mean():.1f}")
        for pattern in [(11,8),(12,9),(10,7),(9,6),(8,5)]:
            ret, corners = cv2.findChessboardCorners(img2, pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FILTER_QUADS)
            print(f"  pattern={pattern}: ret={ret}")
