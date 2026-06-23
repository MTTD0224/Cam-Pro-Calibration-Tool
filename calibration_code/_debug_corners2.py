# -*- coding: utf-8 -*-
import os, sys
import cv2
import numpy as np

folder = r"d:\2 Program Code\CameraCalibration\calibration_img\1"
img_path = os.path.join(folder, "44.bmp")
print(f"读取: {img_path}")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
print(f"尺寸: {img.shape if img is not None else 'None'}")
if img is None:
    print("图像读取失败")
    sys.exit(1)

# 缩小加快处理
scale = 4
h, w = img.shape
small = cv2.resize(img, (w // scale, h // scale))
print(f"缩小后: {small.shape}, mean={small.mean():.1f}, min={small.min()}, max={small.max()}")

# 保存一个小截图用于可视化
cv2.imwrite("d:/2 Program Code/CameraCalibration/calibration_code/_chess_preview.png", small)

# 在小图上尝试检测
print("\n在缩小的图像上尝试检测:")
for pattern in [(11,8),(12,9),(10,7),(9,6),(8,5)]:
    ret, corners = cv2.findChessboardCorners(small, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FILTER_QUADS)
    print(f"  pattern={pattern}: ret={ret}")

# 尝试 0.bmp
print("\n检查 0.bmp (全黑):")
img0 = cv2.imread(os.path.join(folder, "0.bmp"), cv2.IMREAD_GRAYSCALE)
print(f"尺寸: {img0.shape if img0 is not None else 'None'}, mean={img0.mean():.1f}")

# 尝试 target.bmp
target = os.path.join(folder, "target.bmp")
if os.path.exists(target):
    print(f"\n检查 target.bmp:")
    img_t = cv2.imread(target, cv2.IMREAD_GRAYSCALE)
    print(f"尺寸: {img_t.shape if img_t is not None else 'None'}, mean={img_t.mean():.1f}")
    small_t = cv2.resize(img_t, (w // scale, h // scale))
    cv2.imwrite("d:/2 Program Code/CameraCalibration/calibration_code/_target_preview.png", small_t)
    for pattern in [(11,8),(12,9),(10,7)]:
        ret, corners = cv2.findChessboardCorners(small_t, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FILTER_QUADS)
        print(f"  pattern={pattern}: ret={ret}")
