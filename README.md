# 相机 - 投影仪联合标定工具（格雷码结构光）

一个对标 MATLAB 双目相机标定工具箱，使用 Python + PyQt5 + OpenCV 实现的桌面端结构光相机-投影仪联合标定软件。

---

## 1. 功能概览

- ✅ **相机单目标定**：棋盘格/圆形网格角点自动检测 + OpenCV `calibrateCamera` 求解内参、畸变系数
- ✅ **格雷码批量解码**：互补格雷码（正/反格雷码）消除环境光干扰，输出相机像素→投影仪像素的对应映射
- ✅ **投影仪虚拟单目标定**：复用世界三维坐标点，把投影仪当作逆向相机进行单目标定
- ✅ **相机-投影仪双目标定**：`cv2.stereoCalibrate` 优化全局参数，求解旋转矩阵 `R` + 平移向量 `T`
- ✅ **参数保存**：一键输出 `camera_mtx.npy` / `camera_dist.npy` / `projector_mtx.npy` / `projector_dist.npy` / `result_T.npy`
- ✅ **标定板类型切换**：支持棋盘格（Chessboard）和对称式圆形网格（Circles Grid）两种标定板
- ✅ **GUI + 命令行**：既可以图形界面点选操作，也可以用脚本批量运行
- ✅ **异常捕获**：路径不存在、图像读取失败、角点失败、格雷码无有效点、标定奇异报错均有弹窗或日志提示

---

## 2. 依赖安装

```bash
pip install numpy opencv-python PyQt5
```

> `matplotlib` 为可选依赖（用于自定义可视化）。

---

## 3. 目录结构

```
calibration_code/
├── main.py                   # 程序入口（GUI / CLI）
├── calibration_gui.py        # PyQt5 主界面
├── calibration_pipeline.py   # 整合各模块的完整标定工作流
├── config.py                 # 统一参数配置（含标定板类型）
├── image_loader.py           # 图像加载 + 光照补偿
├── chessboard_detector.py    # 标定板角点检测（棋盘格 + 圆形网格）
├── graycode_decoder.py       # 格雷码解码（互补格雷码）
├── single_calibration.py     # 单目标定（相机/投影仪）
├── stereo_calibration.py     # 相机-投影仪双目标定
├── params_saver.py           # 保存/加载 npy 参数文件
└── README.md                 # 本说明书
```

---

## 4. 图像采集与命名规范

**每组位姿** 对应一个子文件夹。把所有位姿文件夹放在同一个根目录下。命名规则按个人习惯即可，只要是整数（1, 2, 3, …）或字母顺序排列均可。

### 4.1 每个位姿文件夹内部的图像命名规则

```
<根目录>/
    ├── 1/
    │   ├── 0.bmp       ← 全黑背景图（投影全黑）
    │   ├── 1.bmp       ┐
    │   ├── ...         │ 水平正格雷码 (N 张，N=bits)
    │   ├── N.bmp       ┘
    │   ├── N+1.bmp     ┐
    │   ├── ...         │ 水平反格雷码 (N 张)
    │   ├── 2N.bmp      ┘
    │   ├── 2N+1.bmp    ┐
    │   ├── ...         │ 垂直正格雷码 (N 张)
    │   ├── 3N.bmp      ┘
    │   ├── 3N+1.bmp    ┐
    │   ├── ...         │ 垂直反格雷码 (N 张)
    │   └── 4N.bmp      ┘  ← 最后一张同时作为"全白投影/标定板图"
    ├── 2/
    ├── 3/
    └── ...
```

**总图像数 = 4 × bits + 1**（最后一张反格雷码同时用作全白标定板图，复用同一文件）。

以 `bits=11` 为例：每组位姿共 **45 张**图像（0-44）。

### 4.2 支持的图像格式

`.bmp`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`。建议使用无损格式（`bmp`/`png`）以避免压缩噪声影响格雷码解码。

### 4.3 标定板规格

#### 棋盘格标定板

- 内角点数：例如 11×8（列数 × 行数）
- 每个方格物理尺寸：例如 15 mm
- 请确保同一标定过程中方格尺寸不要改变
- 标定板要平整，拍摄时光照均匀

#### 对称式圆形网格标定板

- 圆心数：例如 10×11（列数 × 行数）
- 相邻圆心间距：例如 9 mm
- 圆点颜色：白底黑色圆 或 **黑底白色圆**（推荐，blobColor=255）
- 建议圆点直径约为圆心间距的 0.6~0.7 倍

### 4.4 建议的位姿数

至少 **10~15 组不同位姿**，覆盖不同距离、不同倾角，避免纯平行/纯正面姿态。

---

## 5. 使用方式

### 5.1 图形界面（GUI）

```bash
cd calibration_code
python main.py
```

操作步骤：

1. **图像根目录**：点击"浏览"选择标定图像所在父目录
2. **输出目录**：选择保存 npy 参数文件的目录
3. **标定参数**：
   - **标定板类型**：选择「棋盘格」或「圆形网格」
   - 棋盘格：列数/行数、方格边长(mm)
   - 圆形网格：列数/行数、圆心间距(mm)
   - 通用：格雷码位数、投影仪分辨率、误差阈值
4. **预览角点**（可选）：点击"① 预览角点检测"快速确认角点检测是否正常
5. **开始标定**：点击"② 开始完整标定"，进度条实时刷新，底部日志显示细节
6. **查看结果**：切换到"标定结果"、"误差详情"标签页查看参数与误差
7. **保存 npy**：自动在输出目录生成 5 个 npy 文件 + 1 个 JSON 描述

### 5.2 命令行（CLI）

#### 棋盘格模式

```bash
python main.py --cli --root <图像根目录> --output <输出目录> \
               --pattern chessboard --cols 11 --rows 8 --square 15.0 \
               --bits 11 --proj-w 1920 --proj-h 1080 --threshold 1.0
```

#### 圆形网格模式

```bash
python main.py --cli --root <图像根目录> --output <输出目录> \
               --pattern circles --cols 10 --rows 11 --spacing 9.0 \
               --bits 11 --proj-w 1920 --proj-h 1080 --threshold 1.0
```

#### 常用参数

| 参数 | 含义 | 默认 |
|------|------|------|
| `--cli` | 使用命令行模式（无 GUI） | False |
| `--root` | 标定图像根目录（必需） | - |
| `--output` | 输出目录（必需） | - |
| `--pattern` | 标定板类型：`chessboard` 或 `circles` | chessboard |
| `--cols` | 棋盘格内角点列数 / 圆形网格列数 | 11 |
| `--rows` | 棋盘格内角点行数 / 圆形网格行数 | 8 |
| `--square` | 方格边长（mm，仅棋盘格） | 15.0 |
| `--spacing` | 圆形网格圆心间距（mm，仅圆形网格） | 9.0 |
| `--bits` | 格雷码位数 | 11 |
| `--proj-w/--proj-h` | 投影仪分辨率（宽/高，像素） | 1920/1080 |
| `--threshold` | 重投影误差过滤阈值（像素） | 1.0 |

---

## 6. 参数文件格式与加载示例

标定完成后将在输出目录生成 **6 个文件**（含 JSON 描述）：

```
output/
├── camera_mtx.npy       # 相机内参矩阵 (3x3)
├── camera_dist.npy      # 相机畸变系数 (1x5 或 1x...)
├── projector_mtx.npy    # 投影仪内参矩阵 (3x3)
├── projector_dist.npy   # 投影仪畸变系数
├── result_T.npy         # 相机→投影仪外参 4x4 变换矩阵
└── calibration_info.json# 可读性摘要（fx/fy/cx/cy/R/T ...）
```

内参矩阵格式：

```
[ fx  0  cx ]
[ 0   fy cy ]
[ 0   0   1 ]
```

畸变系数顺序（OpenCV 默认）：`[k1, k2, p1, p2, k3]`

`result_T.npy` 存储的是 4x4 变换矩阵，把"相机坐标系点 P_cam"转换到"投影仪坐标系 P_proj"：

```
[ R11 R12 R13 Tx ]
[ R21 R22 R23 Ty ]
[ R31 R32 R33 Tz ]
[ 0   0   0   1  ]
```

### 6.1 加载示例

```python
import numpy as np

# ------- 方式 A：手动加载 -------
camera_mtx = np.load("camera_mtx.npy")
camera_dist = np.load("camera_dist.npy")
projector_mtx = np.load("projector_mtx.npy")
projector_dist = np.load("projector_dist.npy")
transform = np.load("result_T.npy")  # shape (4, 4)

R = transform[:3, :3]
T = transform[:3, 3:4]

print("相机内参:\n", camera_mtx)
print("相机畸变:\n", camera_dist)
print("旋转矩阵 R:\n", R)
print("平移向量 T:\n", T.ravel())


# ------- 方式 B：使用 params_saver 辅助函数 -------
import sys
sys.path.append(".")
from params_saver import load_all_params, load_result_T

params = load_all_params(dir_path=".")
print("投影仪焦距 fx =", params['projector_mtx'][0, 0])
print("投影仪焦距 fy =", params['projector_mtx'][1, 1])

R, T = load_result_T("result_T.npy")
print("R=\n", R)
print("T=\n", T.ravel())


# ------- 应用：把相机像素反投影到投影仪像素 -------
# 示例点：相机像素 (u, v) = (1000, 500)，假设深度 z = 500 mm
u, v, z = 1000, 500, 500.0
# 归一化平面坐标
x = (u - camera_mtx[0, 2]) / camera_mtx[0, 0] * z
y = (v - camera_mtx[1, 2]) / camera_mtx[1, 1] * z
P_cam = np.array([[x], [y], [z], [1.0]])
P_proj = transform @ P_cam  # 投影仪坐标系下三维点
# 再用投影仪内参反投影到投影仪像素
up = projector_mtx[0, 0] * P_proj[0, 0] / P_proj[2, 0] + projector_mtx[0, 2]
vp = projector_mtx[1, 1] * P_proj[1, 0] / P_proj[2, 0] + projector_mtx[1, 2]
print(f"相机({u},{v}) -> 投影仪({up:.1f},{vp:.1f})")
```

---

## 7. 核心算法要点（对标 MATLAB）

| 步骤 | MATLAB 工具箱 | 本实现 |
|------|--------------|--------|
| 棋盘格角点检测 | `detectCheckerboardPoints` | `cv2.findChessboardCorners` + `cornerSubPix` |
| 圆形网格圆心检测 | `detectCirclesGrid` | `cv2.findCirclesGrid` + `SimpleBlobDetector` |
| 世界坐标生成 | `generateCheckerboardPoints` / `generateCirclePoints` | `np.mgrid` 生成规则网格（行优先），适配两种标定板 |
| 单目标定 | `estimateCameraParameters` | `cv2.calibrateCamera` |
| 畸变模型 | Brown-Conrady（k1, k2, p1, p2, k3）| OpenCV 默认 5 参数模型（同 MATLAB） |
| 格雷码解码 | 需自行编写 | 正/反格雷码差分 → 格雷码→二进制→十进制，差分值绝对值阈值过滤阴影与过曝 |
| 格雷码缩放 | 需自行编写 | 将解码值 (0~2^N-1) 缩放到投影仪实际分辨率 (0~width-1, 0~height-1) |
| 双目标定 | `estimateStereoCalibration` | `cv2.stereoCalibrate`，固定内参优化 `R`, `T` |

---

## 8. 位姿过滤逻辑

标定流程中，位姿会经过 **4 层过滤**，逐步剔除质量差的数据：

| 层 | 过滤条件 | 判定规则 |
|----|----------|----------|
| 1 | 角点检测失败 | `cv2.findChessboardCorners` / `findCirclesGrid` 返回 `False` |
| 2 | 重投影误差过大 | 误差 > `--threshold`（默认 1.0 像素），过滤后重新标定 |
| 3 | 格雷码解码有效率低 | 有效角点比例 < 50% |
| 4 | 交叉验证 | 只有同时有相机+投影仪有效角点的位姿参与双目标定 |

**关键参数**：
- `--threshold`：重投影误差过滤阈值（默认 1.0 像素）
- 解码有效率阈值：50%（固定）
- 最小有效位姿数：3（标定算法要求）

---

## 9. 常见问题

- **角点检测失败**：图像过曝/过暗，或标定板贴得不够平整。提高光照对比度。
- **圆形网格检测失败**：检查圆点颜色是否与配置一致（`blobColor=255` 检测白色圆点）。调整 `minArea/maxArea` 过滤噪点。
- **格雷码解码率低**：投影仪与相机未同步触发、环境光太强、表面反射不均。建议关灯、用哑光标定板、提高格雷码位数到 11~12。
- **重投影误差过大（>2px）**：增加位姿组数、增加格雷码位数、降低误差过滤阈值。
- **程序无法启动**：请确认已安装 `PyQt5`、`opencv-python`、`numpy`。
- **大图像运行慢**：修改 `config.py` 中的 `downsample_max_width` 到适当值（如 1500~2500）即可自动下采样加速。默认 2000。

---

## 10. 输出示例（calibration_info.json）

```json
{
  "calibration_summary": {
    "global_rms_error_pixels": 0.31,
    "mean_reprojection_error_pixels": 0.29,
    "valid_pose_count": 10,
    "valid_poses": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
  },
  "camera_intrinsics": {"fx": 14237.41, "fy": 14214.53, "cx": 2078.52, "cy": 477.77},
  "projector_intrinsics": {"fx": 960.12, "fy": 958.40, "cx": 959.49, "cy": 539.49},
  "extrinsics_camera_to_projector": {
    "R_matrix": [[...], [...], [...]],
    "T_vector": [[...], [...], [...]],
    "description": "将相机坐标系中的点 P_cam 转换到投影仪坐标系：P_proj = R * P_cam + T"
  }
}
```
