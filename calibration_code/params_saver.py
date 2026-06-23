# -*- coding: utf-8 -*-
"""
标定参数保存与加载模块
-----------------------------
功能：
1. 将相机/投影仪的内参、畸变系数、R/T矩阵保存为 npy 文件
2. 支持多种 result_T.npy 的保存格式（展平拼接 / 字典 / 二维数组）
3. 提供便捷的参数加载函数
4. 生成标定信息汇总的 JSON 文本文件

输出文件（5 个 npy 文件）：
    camera_mtx.npy    - 相机内参矩阵 (3x3)
    camera_dist.npy   - 相机畸变系数 (1x5 等)
    projector_mtx.npy - 投影仪内参矩阵 (3x3)
    projector_dist.npy- 投影仪畸变系数 (1x5 等)
    result_T.npy      - 相机->投影仪的旋转矩阵 R + 平移向量 T
"""

import os
import json
import numpy as np


class ParamSaver:
    """标定参数保存器"""

    def __init__(self, output_dir=None):
        """
        初始化参数保存器

        参数：
            output_dir: 输出目录路径（若不存在将自动创建）
        """
        self.output_dir = output_dir
        self.saved_files = []

    # ==================== 目录设置 ====================

    def set_output_dir(self, output_dir):
        """设置输出目录"""
        self.output_dir = output_dir

    def ensure_dir(self, dir_path=None):
        """确保目录存在，不存在则创建"""
        if dir_path is None:
            dir_path = self.output_dir
        if dir_path is None:
            raise RuntimeError("未设置输出目录")
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    # ==================== 保存核心 ====================

    def save_single_npy(self, array, filename, output_dir=None):
        """
        保存单个 numpy 数组为 npy 文件

        参数：
            array: np.ndarray, 要保存的数组
            filename: str, 文件名（如 'camera_mtx.npy'）
            output_dir: str, 输出目录，None 则使用默认

        返回：
            full_path: str, 保存的完整路径
        """
        if array is None:
            raise RuntimeError(f"保存 {filename} 失败：数组为 None")

        if output_dir is None:
            output_dir = self.output_dir

        self.ensure_dir(output_dir)
        full_path = os.path.join(output_dir, filename)

        # 确保保存为 float64 以便兼容大多数应用场景
        arr_to_save = np.asarray(array, dtype=np.float64)
        np.save(full_path, arr_to_save)

        self.saved_files.append(full_path)
        return full_path

    def save_all_params(self, stereo_result, output_dir=None):
        """
        一次性保存所有 5 个标定 npy 文件

        参数：
            stereo_result: dict, 来自 StereoCalibration.get_calibration_result() 的结果
            output_dir: str, 输出目录，None 则使用默认

        返回：
            file_list: list of str, 已保存的文件完整路径列表
        """
        if output_dir is not None:
            self.output_dir = output_dir

        if self.output_dir is None:
            raise RuntimeError("未设置输出目录")

        self.ensure_dir()
        self.saved_files = []

        # 1. 相机内参
        self.save_single_npy(stereo_result['camera_mtx'], 'camera_mtx.npy')

        # 2. 相机畸变
        self.save_single_npy(stereo_result['camera_dist'], 'camera_dist.npy')

        # 3. 投影仪内参
        self.save_single_npy(stereo_result['projector_mtx'], 'projector_mtx.npy')

        # 4. 投影仪畸变
        self.save_single_npy(stereo_result['projector_dist'], 'projector_dist.npy')

        # 5. 旋转矩阵 R + 平移向量 T 合并保存
        R = np.asarray(stereo_result['R'], dtype=np.float64)
        T = np.asarray(stereo_result['T'], dtype=np.float64)
        # 构造 4x4 变换矩阵（方便直接使用）
        # [R  T]
        # [0  1]
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = R.reshape(3, 3)
        transform[:3, 3:4] = T.reshape(3, 1)
        self.save_single_npy(transform, 'result_T.npy')

        # 额外信息：保存 JSON 描述文本
        self.save_info_json(stereo_result)

        return self.saved_files

    # ==================== 信息文件 ====================

    def save_info_json(self, stereo_result, output_dir=None):
        """
        保存标定参数的 JSON 描述文本，方便人工查看

        参数：
            stereo_result: dict, 来自 StereoCalibration.get_calibration_result()
        """
        if output_dir is None:
            output_dir = self.output_dir

        info = {
            'calibration_summary': {
                'global_rms_error_pixels': float(stereo_result.get('rms', -1)),
                'mean_reprojection_error_pixels': float(stereo_result.get('mean_error', -1)),
                'valid_pose_count': int(stereo_result.get('pose_count', 0)),
                'valid_poses': stereo_result.get('valid_poses', []),
            },
            'camera_intrinsics': {
                'fx': float(stereo_result['camera_mtx'][0, 0]),
                'fy': float(stereo_result['camera_mtx'][1, 1]),
                'cx': float(stereo_result['camera_mtx'][0, 2]),
                'cy': float(stereo_result['camera_mtx'][1, 2]),
            },
            'camera_distortion': {
                'coefficients': stereo_result['camera_dist'].ravel().tolist(),
                'description': '[k1, k2, p1, p2, k3, ...]'
            },
            'projector_intrinsics': {
                'fx': float(stereo_result['projector_mtx'][0, 0]),
                'fy': float(stereo_result['projector_mtx'][1, 1]),
                'cx': float(stereo_result['projector_mtx'][0, 2]),
                'cy': float(stereo_result['projector_mtx'][1, 2]),
            },
            'projector_distortion': {
                'coefficients': stereo_result['projector_dist'].ravel().tolist(),
                'description': '[k1, k2, p1, p2, k3, ...]'
            },
            'extrinsics_camera_to_projector': {
                'R_matrix': stereo_result['R'].tolist(),
                'T_vector': stereo_result['T'].ravel().tolist(),
                'description': '将相机坐标系中的点 X_cam 转换到投影仪坐标系：X_proj = R * X_cam + T'
            },
            'files': {
                'camera_mtx': 'camera_mtx.npy',
                'camera_dist': 'camera_dist.npy',
                'projector_mtx': 'projector_mtx.npy',
                'projector_dist': 'projector_dist.npy',
                'result_T': 'result_T.npy (4x4 transform matrix [R T; 0 1])'
            }
        }

        json_path = os.path.join(output_dir, 'calibration_info.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        self.saved_files.append(json_path)
        return json_path


# ==================== 便捷加载函数 ====================

def load_camera_mtx(path='camera_mtx.npy'):
    """加载相机内参矩阵 (3x3)"""
    return np.load(path)


def load_camera_dist(path='camera_dist.npy'):
    """加载相机畸变系数向量"""
    return np.load(path)


def load_projector_mtx(path='projector_mtx.npy'):
    """加载投影仪内参矩阵 (3x3)"""
    return np.load(path)


def load_projector_dist(path='projector_dist.npy'):
    """加载投影仪畸变系数向量"""
    return np.load(path)


def load_result_T(path='result_T.npy', format_mode='matrix'):
    """
    加载相机->投影仪的 R, T 矩阵

    参数：
        path: npy 文件路径
        format_mode:
            'matrix' - 默认，返回 4x4 变换矩阵中的 R(3x3) 和 T(3x1)
            'flatten' - 文件保存的是一维展平拼接 [R_ravel, T_ravel]
            'dict' - 文件是 dict {'R': ..., 'T': ...}

    返回：
        R: 3x3 旋转矩阵
        T: 3x1 平移向量
    """
    data = np.load(path, allow_pickle=True)

    if format_mode == 'dict':
        # 字典格式
        if isinstance(data, dict):
            R = np.asarray(data['R'], dtype=np.float64).reshape(3, 3)
            T = np.asarray(data['T'], dtype=np.float64).reshape(3, 1)
        else:
            # 可能是 object 数组包装的 dict
            item = data.item()
            R = np.asarray(item['R'], dtype=np.float64).reshape(3, 3)
            T = np.asarray(item['T'], dtype=np.float64).reshape(3, 1)

    elif format_mode == 'flatten':
        # 一维展平拼接格式 [R11, R12, R13, R21, ..., R33, Tx, Ty, Tz]
        flat = data.ravel()
        R = flat[:9].reshape(3, 3)
        T = flat[9:12].reshape(3, 1)

    else:  # matrix
        # 4x4 变换矩阵
        arr = np.asarray(data, dtype=np.float64)
        if arr.shape == (4, 4):
            R = arr[:3, :3]
            T = arr[:3, 3:4]
        elif arr.shape == (3, 4):
            R = arr[:3, :3]
            T = arr[:3, 3:4]
        elif arr.size == 12:
            # 自动检测：展平数据前9个 R，后3个 T
            flat = arr.ravel()
            R = flat[:9].reshape(3, 3)
            T = flat[9:12].reshape(3, 1)
        else:
            raise ValueError(f"无法识别 result_T.npy 的数据形状 {arr.shape}")

    return R, T


def load_all_params(dir_path='.', separate_RT=False):
    """
    一键加载所有 5 个标定文件

    参数：
        dir_path: 标定参数所在目录
        separate_RT: True 时将 result_T 拆分为 R 和 T 单独返回

    返回：
        params: dict
            {
                'camera_mtx': np.ndarray (3x3),
                'camera_dist': np.ndarray,
                'projector_mtx': np.ndarray (3x3),
                'projector_dist': np.ndarray,
                'result_T': np.ndarray (4x4),  # 若 separate_RT=False
                'R': np.ndarray (3x3),         # 若 separate_RT=True
                'T': np.ndarray (3x1),         # 若 separate_RT=True
            }
    """
    params = {
        'camera_mtx': np.load(os.path.join(dir_path, 'camera_mtx.npy')),
        'camera_dist': np.load(os.path.join(dir_path, 'camera_dist.npy')),
        'projector_mtx': np.load(os.path.join(dir_path, 'projector_mtx.npy')),
        'projector_dist': np.load(os.path.join(dir_path, 'projector_dist.npy')),
    }

    RT_path = os.path.join(dir_path, 'result_T.npy')
    if separate_RT:
        R, T = load_result_T(RT_path)
        params['R'] = R
        params['T'] = T
    else:
        params['result_T'] = np.load(RT_path)

    return params
