# -*- coding: utf-8 -*-
"""
图像加载与预处理模块
-----------------------------
功能：
1. 遍历标定根目录下的所有位姿文件夹
2. 加载每组位姿下的所有标定图像
3. 图像格式统一转换为灰度图
4. 光照补偿预处理（基于全白/全黑图）
"""

import os
import glob
import numpy as np
import cv2

from config import IMAGE_EXTENSIONS


class ImageLoader:
    """
    图像加载器

    负责：
    1. 发现根目录下所有的子文件夹（每个子文件夹代表一个位姿）
    2. 按命名规则读取每组位姿下的所有格雷码/棋盘格图像
    3. 图像格式标准化（灰度化、类型转换）
    """

    def __init__(self, config):
        """
        初始化图像加载器

        参数：
            config: CalibrationConfig 实例，包含所有配置参数
        """
        self.config = config
        # 存储所有加载的图像数据
        # 结构：{ 'folder_name': {
        #                  'images': dict(index -> np.ndarray),
        #                  'folder_path': str,
        #                  'valid': bool
        #              } }
        self.poses_data = {}
        # 存储位姿文件夹的有序列表
        self.pose_names = []

    # ==================== 路径扫描 ====================

    def scan_root_dir(self, root_dir):
        """
        扫描标定根目录，发现所有位姿子文件夹

        目录结构示例：
            root_dir/
                1/
                    0.bmp   (全黑)
                    1.bmp   (水平格雷码第1幅)
                    ...
                    44.bmp  (全白棋盘格)
                2/
                    ...
                ...

        参数：
            root_dir: 根目录路径（绝对路径或相对路径）

        返回：
            pose_names: 位姿文件夹名称有序列表
        """
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"标定根目录不存在: {root_dir}")

        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"路径不是一个有效目录: {root_dir}")

        # 获取所有子文件夹
        sub_dirs = []
        for item in os.listdir(root_dir):
            item_path = os.path.join(root_dir, item)
            if os.path.isdir(item_path):
                sub_dirs.append((item, item_path))

        # 按文件夹名称排序（数字优先）
        def sort_key(name_path):
            name, _ = name_path
            try:
                return (0, int(name))
            except ValueError:
                return (1, name.lower())

        sub_dirs.sort(key=sort_key)

        self.pose_names = [name for name, _ in sub_dirs]

        if len(self.pose_names) == 0:
            raise RuntimeError("根目录下未发现任何位姿子文件夹，请检查目录结构")

        return self.pose_names

    # ==================== 图像读取 ====================

    def _find_image_file(self, folder_path, index):
        """
        在指定文件夹中查找指定索引的图像文件（按扩展名优先级搜索）

        参数：
            folder_path: 文件夹路径
            index: 图像索引（0, 1, 2, ...）

        返回：
            完整文件路径，如果未找到返回 None
        """
        for ext in IMAGE_EXTENSIONS:
            filename = f"{index}{ext}"
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath):
                return filepath
        return None

    def load_pose_images(self, folder_path):
        """
        加载单个位姿文件夹内的所有标定图像

        参数：
            folder_path: 位姿文件夹完整路径

        返回：
            images_dict: { index(int) -> image(np.ndarray, uint8) }
            loaded_count: 成功加载的图像数量
        """
        images_dict = {}
        total = self.config.graycode_total_imgs

        for idx in range(total):
            filepath = self._find_image_file(folder_path, idx)
            if filepath is None:
                # 未找到指定索引的图像，跳过
                continue
            # 读取图像（以灰度模式读取）
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[警告] 无法读取图像: {filepath}")
                continue
            images_dict[idx] = img

        return images_dict, len(images_dict)

    def load_all_poses(self, root_dir, progress_callback=None):
        """
        加载根目录下所有位姿文件夹内的所有图像

        参数：
            root_dir: 根目录路径
            progress_callback: 进度回调函数 progress_callback(current, total, message)

        返回：
            poses_data: dict，键为文件夹名称，值为字典
                        {
                            'images': dict(index -> np.ndarray),
                            'folder_path': str,
                            'valid': bool,
                            'gray_bits': int
                        }
        """
        # 1. 扫描根目录
        self.scan_root_dir(root_dir)

        # 2. 遍历加载
        total_poses = len(self.pose_names)
        self.poses_data = {}

        for i, pose_name in enumerate(self.pose_names):
            folder_path = os.path.join(root_dir, pose_name)
            try:
                images, count = self.load_pose_images(folder_path)
                valid = count == self.config.graycode_total_imgs
                self.poses_data[pose_name] = {
                    'images': images,
                    'folder_path': folder_path,
                    'valid': valid,
                    'loaded_count': count
                }
            except Exception as e:
                print(f"[错误] 加载位姿 {pose_name} 失败: {str(e)}")
                self.poses_data[pose_name] = {
                    'images': {},
                    'folder_path': folder_path,
                    'valid': False,
                    'loaded_count': 0
                }

            # 进度汇报
            if progress_callback is not None:
                msg = f"已加载 {pose_name}: {count}/{self.config.graycode_total_imgs} 张图像"
                try:
                    progress_callback(i + 1, total_poses, msg)
                except Exception:
                    pass

        return self.poses_data

    # ==================== 图像获取 ====================

    def get_pose_data(self, pose_name):
        """
        获取指定位姿名称对应的图像数据

        参数：
            pose_name: 位姿文件夹名称

        返回：
            该位姿的字典数据 {'images', 'folder_path', 'valid'}
        """
        if pose_name not in self.poses_data:
            raise KeyError(f"未发现位姿 {pose_name}")
        return self.poses_data[pose_name]

    def get_image(self, pose_name, index):
        """
        获取指定位姿、指定索引的图像

        参数：
            pose_name: 位姿名称
            index: 图像索引

        返回：
            np.ndarray 图像数据，不存在则返回 None
        """
        data = self.get_pose_data(pose_name)
        return data['images'].get(index, None)

    def get_white_image(self, pose_name):
        """获取全白棋盘格图像（索引 4N+1）"""
        idx = self.config.get_graycode_indices()['white']
        return self.get_image(pose_name, idx)

    def get_black_image(self, pose_name):
        """获取全黑背景图像（索引 0）"""
        return self.get_image(pose_name, 0)

    def get_graycode_images(self, pose_name, direction='horizontal'):
        """
        获取格雷码图像序列

        参数：
            pose_name: 位姿名称
            direction: 'horizontal' 或 'vertical'

        返回：
            positive_imgs: list of np.ndarray，正格雷码图像列表（按位从高位到低位）
            negative_imgs: list of np.ndarray，反格雷码图像列表
        """
        indices = self.config.get_graycode_indices()
        if direction == 'horizontal':
            pos_range = indices['h_positive']
            neg_range = indices['h_negative']
        elif direction == 'vertical':
            pos_range = indices['v_positive']
            neg_range = indices['v_negative']
        else:
            raise ValueError(f"未知方向参数: {direction}")

        pos_start, pos_end = pos_range
        neg_start, neg_end = neg_range

        positive_imgs = []
        negative_imgs = []

        for i in range(pos_start, pos_end + 1):
            img = self.get_image(pose_name, i)
            positive_imgs.append(img)

        for j in range(neg_start, neg_end + 1):
            img = self.get_image(pose_name, j)
            negative_imgs.append(img)

        return positive_imgs, negative_imgs

    # ==================== 光照补偿 ====================

    def apply_light_compensation(self, pose_name):
        """
        对指定位姿下的所有格雷码图像进行光照补偿处理

        原理：
        I_compensated = (I_positive - I_black) / (I_white - I_black)

        其中 I_black 为全黑背景图（环境光干扰），I_white 为全白投影图（最大光照）

        参数：
            pose_name: 位姿文件夹名称

        返回：
            compensated: dict，包含四个方向的补偿后图像
                {
                    'h_positive': [gray_compensated, ...],
                    'h_negative': [...],
                    'v_positive': [...],
                    'v_negative': [...]
                }
        """
        black_img = self.get_black_image(pose_name)
        white_img = self.get_white_image(pose_name)

        if black_img is None or white_img is None:
            raise RuntimeError(f"位姿 {pose_name} 缺少全黑/全白图像，无法进行光照补偿")

        # 归一化到float类型，防止uint8溢出
        black = black_img.astype(np.float32)
        white = white_img.astype(np.float32)

        # 计算差值（I_white - I_black），防止除零
        diff = white - black
        diff[diff < 1.0] = 1.0

        # 获取四个方向的格雷码图像
        h_pos, h_neg = self.get_graycode_images(pose_name, 'horizontal')
        v_pos, v_neg = self.get_graycode_images(pose_name, 'vertical')

        def compensate(img_list):
            compensated_list = []
            for img in img_list:
                if img is None:
                    compensated_list.append(None)
                    continue
                img_f = img.astype(np.float32)
                # 光照补偿公式
                comp = (img_f - black) / diff * 255.0
                comp = np.clip(comp, 0, 255)
                compensated_list.append(comp.astype(np.uint8))
            return compensated_list

        return {
            'h_positive': compensate(h_pos),
            'h_negative': compensate(h_neg),
            'v_positive': compensate(v_pos),
            'v_negative': compensate(v_neg)
        }

    # ==================== 状态查询 ====================

    def get_valid_poses(self):
        """获取所有有效位姿名称列表"""
        return [name for name, data in self.poses_data.items() if data['valid']]

    def get_invalid_poses(self):
        """获取所有无效位姿名称列表"""
        return [name for name, data in self.poses_data.items() if not data['valid']]

    def get_summary(self):
        """返回统计信息字典"""
        total = len(self.poses_data)
        valid = len(self.get_valid_poses())
        invalid = total - valid
        return {
            'total': total,
            'valid': valid,
            'invalid': invalid,
            'images_per_pose': self.config.graycode_total_imgs
        }

    def print_summary(self):
        """打印加载统计信息"""
        summary = self.get_summary()
        print("========== 图像加载统计 ==========")
        print(f"总位姿数: {summary['total']}")
        print(f"有效位姿数: {summary['valid']}")
        print(f"无效位姿数: {summary['invalid']}")
        print(f"每点位姿图像数: {summary['images_per_pose']}")
        if self.pose_names:
            print("位姿列表:")
            for name in self.pose_names:
                data = self.poses_data[name]
                status = "✓" if data['valid'] else "✗"
                print(f"  {status} {name}: {data['loaded_count']}/{summary['images_per_pose']} 张")
        print("==================================")
