# -*- coding: utf-8 -*-
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from config import CalibrationConfig
from calibration_pipeline import run_calibration

root = r"d:\2 Program Code\CameraCalibration\calibration_img"
out = r"d:\2 Program Code\CameraCalibration\calibration_code\test_output"

cfg = CalibrationConfig()
cfg.pattern_size = (11, 8)
cfg.square_size = 10.0
cfg.graycode_bits = 11
cfg.projector_size = (1920, 1080)
cfg.reprojection_threshold = 1.0  # 使用较宽松的阈值

print("开始测试完整标定工作流...")
print(f"图像根目录: {root}")
print(f"输出目录: {out}")
print(f"格雷码位数: {cfg.graycode_bits}, 每姿图像数: {cfg.graycode_total_imgs}")
print()

try:
    result = run_calibration(root, out, cfg)
    print()
    print("=== 标定结果总结 ===")
    print(f"全局 RMS: {result.get('rms', -1):.4f}")
    print(f"平均误差: {result.get('mean_error', -1):.4f}")
    print(f"有效位姿: {result.get('pose_count', 0)}")
    cm = result.get('camera_mtx')
    if cm is not None:
        print(f"相机 fx={cm[0,0]:.2f}, fy={cm[1,1]:.2f}, cx={cm[0,2]:.2f}, cy={cm[1,2]:.2f}")
    pm = result.get('projector_mtx')
    if pm is not None:
        print(f"投影 fx={pm[0,0]:.2f}, fy={pm[1,1]:.2f}, cx={pm[0,2]:.2f}, cy={pm[1,2]:.2f}")
    print("参数文件保存到:", out)
    for f in os.listdir(out):
        print(f"  - {f}")
    sys.exit(0)
except Exception as e:
    print(f"[错误] 标定失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
