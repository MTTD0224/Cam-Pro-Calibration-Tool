# -*- coding: utf-8 -*-
"""
相机-投影仪联合标定工具 主程序入口
=======================================
功能：
  1. 启动 PyQt5 图形界面（默认方式）
  2. 支持命令行模式（--cli 参数）

使用方法：
  启动 GUI:     python main.py
  命令行标定:   python main.py --cli --root <图像根目录> --output <输出目录>
  查看帮助:     python main.py -h
"""

import os
import sys
import argparse


def _check_dependencies():
    """检查依赖库是否安装"""
    missing = []
    required = [
        ("PyQt5", "PyQt5"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
    ]
    for import_name, pkg_name in required:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        print("[警告] 以下依赖库未安装，可能影响功能:")
        for m in missing:
            print(f"    - {m}  (pip install {m})")
    return missing


def run_gui():
    """启动图形界面"""
    from calibration_gui import run_gui
    print("[信息] 启动相机-投影仪联合标定工具 GUI...")
    run_gui()


def run_cli(args):
    """命令行标定模式"""
    from calibration_pipeline import run_calibration
    from config import CalibrationConfig

    if not os.path.isdir(args.root):
        print(f"[错误] 图像根目录不存在: {args.root}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    cfg = CalibrationConfig()
    cfg.pattern_size = (args.cols, args.rows)
    cfg.square_size = args.square
    cfg.graycode_bits = args.bits
    cfg.projector_size = (args.proj_w, args.proj_h)
    cfg.reprojection_threshold = args.threshold

    print("=========== 标定参数 ===========")
    print(f"棋盘格内角点: {cfg.pattern_size} (列 x 行)")
    print(f"方格边长: {cfg.square_size} mm")
    print(f"格雷码位数: {cfg.graycode_bits}")
    print(f"投影仪分辨率: {cfg.projector_size}")
    print(f"重投影误差过滤阈值: {cfg.reprojection_threshold} 像素")
    print(f"图像根目录: {args.root}")
    print(f"输出目录: {args.output}")
    print("==================================")

    result = run_calibration(args.root, args.output, cfg)

    print("\n=========== 标定完成 ===========")
    if result:
        print(f"全局 RMS 误差: {result.get('rms', -1):.4f} 像素")
        print(f"有效位姿数: {result.get('pose_count', 0)}")
        cm = result.get('camera_mtx')
        if cm is not None:
            print(f"相机内参 fx={cm[0,0]:.4f}, fy={cm[1,1]:.4f}, "
                  f"cx={cm[0,2]:.4f}, cy={cm[1,2]:.4f}")
        pm = result.get('projector_mtx')
        if pm is not None:
            print(f"投影仪内参 fx={pm[0,0]:.4f}, fy={pm[1,1]:.4f}, "
                  f"cx={pm[0,2]:.4f}, cy={pm[1,2]:.4f}")
        print(f"参数文件已保存至: {args.output}")
    else:
        print("[警告] 标定结果为空，请检查日志")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="相机-投影仪联合标定工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用示例：
  1) 启动图形界面：
      python main.py

  2) 命令行模式标定：
      python main.py --cli --root ./calibration_img --output ./output
                      --cols 11 --rows 8 --square 15 --bits 11
"""
    )
    parser.add_argument("--cli", action="store_true",
                      help="启用命令行模式（无 GUI）")
    parser.add_argument("--root", "-i", help="标定图像根目录（--cli 模式必需）")
    parser.add_argument("--output", "-o", help="参数输出目录（--cli 模式必需）")
    parser.add_argument("--cols", type=int, default=11, help="棋盘格内点列数（默认 11）")
    parser.add_argument("--rows", type=int, default=8, help="棋盘格内点行数（默认 8）")
    parser.add_argument("--square", type=float, default=10.0, help="方格边长（mm，默认 10）")
    parser.add_argument("--bits", type=int, default=11, help="格雷码位数（默认 11）")
    parser.add_argument("--proj-w", type=int, default=1920, help="投影仪宽度（像素，默认 1920）")
    parser.add_argument("--proj-h", type=int, default=1080, help="投影仪高度（像素，默认 1080）")
    parser.add_argument("--threshold", type=float, default=0.3,
                      help="重投影误差过滤阈值（像素，默认 0.3）")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 检查依赖库
    _check_dependencies()

    if args.cli:
        # 命令行模式
        if not args.root or not args.output:
            print("[错误] --cli 模式下必须同时指定 --root 和 --output")
            parser.print_help()
            sys.exit(1)
        run_cli(args)
    else:
        # 默认启动 GUI
        run_gui()


if __name__ == "__main__":
    main()
