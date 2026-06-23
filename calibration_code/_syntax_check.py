# -*- coding: utf-8 -*-
"""全模块语法检查脚本"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

modules = [
    'config.py',
    'image_loader.py',
    'chessboard_detector.py',
    'graycode_decoder.py',
    'single_calibration.py',
    'stereo_calibration.py',
    'params_saver.py',
    'calibration_pipeline.py',
    'calibration_gui.py',
    'main.py'
]

error_count = 0
for m in modules:
    try:
        source = open(m, 'r', encoding='utf-8').read()
        compile(source, m, 'exec')
        print(f"[OK] {m}")
    except SyntaxError as e:
        print(f"[ERR] {m}: {e}")
        error_count += 1
    except Exception as e:
        print(f"[FAIL] {m}: {e}")
        error_count += 1

print()
if error_count == 0:
    print("所有模块语法检查通过！")
    sys.exit(0)
else:
    print(f"发现 {error_count} 个语法错误")
    sys.exit(1)
