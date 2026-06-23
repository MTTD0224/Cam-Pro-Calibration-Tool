# -*- coding: utf-8 -*-
"""
格雷码解码模块
-----------------------------
功能：
1. 将一组格雷码图像序列解码为投影仪像素坐标
2. 采用互补格雷码（正/反格雷码相减）消除环境光干扰
3. 将格雷码转换为二进制，再转换为十进制坐标值
4. 根据相机角点查表匹配对应投影仪像素坐标
5. 过滤阴影/过曝/解码失败的无效像素

核心原理（参考MATLAB结构光标定）：
- 对每个相机像素 (u, v)，收集其在N张正格雷码和N张反格雷码中的亮度值
- 二值化判断：I_positive - I_negative > 0 则为 1，否则为 0
- 得到N位格雷码序列 G = [g1, g2, ..., gN]
- 格雷码转二进制：b1 = g1, bi = bi-1 XOR gi（i >= 2）
- 二进制转十进制：pixel_index = sum(bi * 2^(N-i))
"""

import numpy as np
import cv2


class GraycodeDecoder:
    """格雷码解码器，实现从图像到投影仪像素坐标的完整解码"""

    def __init__(self, config):
        """
        初始化格雷码解码器

        参数：
            config: CalibrationConfig 配置实例
        """
        self.config = config

    # ==================== 格雷码 <-> 二进制转换 ====================

    @staticmethod
    def graycode_to_binary(gray_bits):
        """
        格雷码序列转二进制序列

        原理：
            binary[0] = gray[0]
            binary[i] = binary[i-1] XOR gray[i]  (i >= 1)

        参数：
            gray_bits: list or np.ndarray of 0/1

        返回：
            binary_bits: np.ndarray of 0/1
        """
        bits = np.asarray(gray_bits, dtype=np.int32)
        n = len(bits)
        binary = np.zeros(n, dtype=np.int32)
        binary[0] = bits[0]
        for i in range(1, n):
            binary[i] = binary[i - 1] ^ bits[i]
        return binary

    @staticmethod
    def binary_to_decimal(binary_bits):
        """
        二进制位序列转十进制数值

        参数：
            binary_bits: list or np.ndarray of 0/1, 高位在前

        返回：
            value: int, 十进制值
        """
        bits = np.asarray(binary_bits, dtype=np.int32)
        n = len(bits)
        # 构造权值向量 [2^(n-1), 2^(n-2), ..., 2^0]
        weights = np.array([1 << (n - i - 1) for i in range(n)], dtype=np.int32)
        return int(np.sum(bits * weights))

    @staticmethod
    def graycode_to_decimal(gray_bits):
        """格雷码序列直接转十进制数值"""
        binary = GraycodeDecoder.graycode_to_binary(gray_bits)
        return GraycodeDecoder.binary_to_decimal(binary)

    # ==================== 单像素解码 ====================

    def decode_single_pixel(self, u, v, positive_imgs, negative_imgs):
        """
        解码单个相机像素对应的投影仪列/行坐标

        原理（互补格雷码法）：
            对每张正格雷码 I_pos 和反格雷码 I_neg:
                diff = I_pos[u,v] - I_neg[u,v]
                if diff > 0: bit = 1
                else:         bit = 0
            得到 N 位格雷码序列，再转换为十进制

        参数：
            u, v: 相机像素坐标 (行, 列)
            positive_imgs: list of np.ndarray, 正格雷码图像列表（按位从高位到低位）
            negative_imgs: list of np.ndarray, 反格雷码图像列表

        返回：
            success: bool, 是否解码成功
            value: int, 解码得到的投影仪坐标（列或行）
        """
        bits = self.config.graycode_bits
        if len(positive_imgs) != bits or len(negative_imgs) != bits:
            return False, -1

        gray_bits = np.zeros(bits, dtype=np.int32)

        for i in range(bits):
            p_img = positive_imgs[i]
            n_img = negative_imgs[i]
            if p_img is None or n_img is None:
                return False, -1

            # 亮度差二值化判断
            diff = int(p_img[u, v]) - int(n_img[u, v])
            if diff > 0:
                gray_bits[i] = 1
            else:
                gray_bits[i] = 0

        # 格雷码 -> 二进制 -> 十进制
        value = self.graycode_to_decimal(gray_bits)

        # 检查结果是否在投影仪分辨率范围内
        return True, value

    # ==================== 全图解码：向量化实现 ====================

    def decode_full_map(self, positive_imgs, negative_imgs):
        """
        向量化解码：一次得到整个相机图像上每个像素对应的投影仪坐标

        使用 numpy 数组运算大幅加速。对大图像自动下采样加速，
        解码完成后映射回原始坐标。

        参数：
            positive_imgs: list of np.ndarray (uint8), 正格雷码图像列表
            negative_imgs: list of np.ndarray (uint8), 反格雷码图像列表

        返回：
            proj_map: np.ndarray (H, W), dtype=float32, 每个像素的解码结果
                      无效像素设置为 NaN
            valid_mask: np.ndarray (H, W), dtype=bool, 有效像素掩码
        """
        bits = self.config.graycode_bits

        if len(positive_imgs) != bits or len(negative_imgs) != bits:
            raise RuntimeError(f"格雷码图像数量不匹配: 需要{bits}张正/反格雷码")

        # 获取图像尺寸
        h, w = positive_imgs[0].shape

        # 检查是否需要下采样加速
        max_w = getattr(self.config, 'downsample_max_width', 0)
        scale = 1.0
        if max_w > 0 and w > max_w:
            scale = float(max_w) / float(w)
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            # 对所有格雷码图进行下采样
            pos_down = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        if img is not None else None for img in positive_imgs]
            neg_down = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        if img is not None else None for img in negative_imgs]
            h_out, w_out = new_h, new_w
        else:
            pos_down = positive_imgs
            neg_down = negative_imgs
            h_out, w_out = h, w

        # 构造差分值矩阵（bits, h, w）
        diff_list = []
        valid_mask_small = np.ones((h_out, w_out), dtype=bool)
        for i in range(bits):
            p_img = pos_down[i]
            n_img = neg_down[i]
            if p_img is None or n_img is None:
                valid_mask_small[:] = False
                break
            # 转为 int16 避免 uint8 溢出
            diff = p_img.astype(np.int16) - n_img.astype(np.int16)
            diff_list.append(diff)

        if not valid_mask_small.any():
            proj_map_full = np.full((h, w), np.nan, dtype=np.float32)
            valid_mask_full = np.zeros((h, w), dtype=bool)
            return proj_map_full, valid_mask_full

        diff_tensor = np.stack(diff_list, axis=0)  # (bits, h, w)

        # 根据差分值的符号构造格雷码
        gray_matrix = (diff_tensor > 0).astype(np.int32)

        # 格雷码 -> 二进制
        binary_matrix = np.zeros_like(gray_matrix)
        binary_matrix[0] = gray_matrix[0]
        for i in range(1, bits):
            binary_matrix[i] = binary_matrix[i - 1] ^ gray_matrix[i]

        # 二进制 -> 十进制
        weights = np.array([1 << (bits - i - 1) for i in range(bits)], dtype=np.int32)
        weights = weights.reshape(bits, 1, 1)
        proj_map_small = np.sum(binary_matrix * weights, axis=0).astype(np.float32)

        # 差分绝对值过滤：每一位至少有一点灰度差，否则为无效区域
        abs_sum = np.sum(np.abs(diff_tensor), axis=0).astype(np.float32)
        threshold = bits * 20.0
        low_confidence = abs_sum < threshold
        valid_mask_small = np.logical_and(valid_mask_small, ~low_confidence)

        # 在下采样的空间中设置 NaN
        proj_map_small[~valid_mask_small] = np.nan

        # 放大回原始分辨率（使用最近邻插值，保持整数编码值）
        if scale != 1.0:
            proj_map_full = cv2.resize(proj_map_small, (w, h),
                                       interpolation=cv2.INTER_NEAREST)
            # valid_mask 同样放大回原尺寸
            valid_mask_uint8 = valid_mask_small.astype(np.uint8) * 255
            valid_mask_full_uint8 = cv2.resize(valid_mask_uint8, (w, h),
                                                interpolation=cv2.INTER_NEAREST)
            valid_mask_full = valid_mask_full_uint8 > 0
        else:
            proj_map_full = proj_map_small
            valid_mask_full = valid_mask_small

        return proj_map_full, valid_mask_full

    # ==================== 位姿解码：返回水平+垂直两个坐标映射 ====================

    def decode_pose(self, compensated):
        """
        对单个位姿执行完整的格雷码解码，得到相机-投影仪的像素对应关系

        参数：
            compensated: dict, 来自 ImageLoader.apply_light_compensation() 的输出
                {
                    'h_positive': [...],  # 水平正格雷码
                    'h_negative': [...],  # 水平反格雷码
                    'v_positive': [...],  # 垂直正格雷码
                    'v_negative': [...],  # 垂直反格雷码
                }

        返回：
            result: dict
                {
                    'proj_col': np.ndarray (H, W) float32, 相机像素对应的投影仪列坐标
                    'proj_row': np.ndarray (H, W) float32, 相机像素对应的投影仪行坐标
                    'valid_mask': np.ndarray (H, W) bool, 双方向均有效的像素掩码
                    'valid_pixels': int, 有效像素总数
                    'total_pixels': int, 总像素数
                }
        """
        bits = self.config.graycode_bits
        max_raw_value = (1 << bits)  # 2^bits, 解码值上限 (0 ~ 2^bits - 1)
        proj_w, proj_h = self.config.projector_size

        # 解码水平方向（得到投影仪列坐标）
        h_pos = compensated['h_positive']
        h_neg = compensated['h_negative']
        proj_col, h_valid = self.decode_full_map(h_pos, h_neg)

        # 解码垂直方向（得到投影仪行坐标）
        v_pos = compensated['v_positive']
        v_neg = compensated['v_negative']
        proj_row, v_valid = self.decode_full_map(v_pos, v_neg)

        # ---- 关键修复：把原始格雷码解码值 (0 ~ 2^bits-1) 缩放到投影仪实际像素坐标 ----
        # 原理：N 位格雷码可以编码 2^N 个唯一值，但投影仪分辨率可能不等于 2^N
        # 例如 11 位格雷码 → 0-2047；但 1920x1080 投影仪需要映射到 0-1919 / 0-1079
        scale_col = proj_w / max_raw_value
        scale_row = proj_h / max_raw_value
        # 对有效像素做缩放（NaN 保持 NaN）
        proj_col = proj_col * scale_col
        proj_row = proj_row * scale_row

        # 双方向均有效的掩码
        valid_mask = np.logical_and(h_valid, v_valid)

        # 检查投影仪坐标是否在分辨率范围内（缩放后应基本全部有效）
        # 保留一个小余量，防止浮点数精度问题
        eps = 1.0
        valid_col = np.logical_and(proj_col >= -eps, proj_col < proj_w + eps)
        valid_row = np.logical_and(proj_row >= -eps, proj_row < proj_h + eps)
        valid_mask = np.logical_and(valid_mask, valid_col)
        valid_mask = np.logical_and(valid_mask, valid_row)

        # 将不在范围内的值设为 NaN
        invalid = ~valid_mask
        proj_col[invalid] = np.nan
        proj_row[invalid] = np.nan

        total_pixels = valid_mask.size
        valid_pixels = int(np.sum(valid_mask))

        return {
            'proj_col': proj_col,
            'proj_row': proj_row,
            'valid_mask': valid_mask,
            'valid_pixels': valid_pixels,
            'total_pixels': total_pixels
        }

    # ==================== 角点匹配：从解码图中查询角点坐标 ====================

    def lookup_corners(self, proj_col_map, proj_row_map, camera_corners):
        """
        根据相机检测到的角点坐标，在解码图中查询对应的投影仪二维像素坐标

        参数：
            proj_col_map: np.ndarray (H, W) float32, 投影仪列坐标解码图
            proj_row_map: np.ndarray (H, W) float32, 投影仪行坐标解码图
            camera_corners: np.ndarray (N, 1, 2) float32, 相机角点坐标 (x=col, y=row)

        返回：
            projector_corners: np.ndarray (N, 1, 2) float32, 投影仪角点坐标
                               (proj_col, proj_row)
            valid_flags: np.ndarray (N,), bool, 每个角点是否成功解码
        """
        if camera_corners is None or camera_corners.size == 0:
            return None, np.zeros(0, dtype=bool)

        N = camera_corners.shape[0]
        projector_corners = np.zeros((N, 1, 2), dtype=np.float32)
        valid_flags = np.zeros(N, dtype=bool)

        H, W = proj_col_map.shape

        for i in range(N):
            # OpenCV 的角点坐标是 (x=col, y=row)，即 (horizontal, vertical)
            x = float(camera_corners[i, 0, 0])  # 列
            y = float(camera_corners[i, 0, 1])  # 行

            # 取整到最近的像素
            col = int(round(x))
            row = int(round(y))

            # 边界检查
            if col < 0 or col >= W or row < 0 or row >= H:
                valid_flags[i] = False
                continue

            # 双线性插值获取投影仪坐标
            col_val = self._bilinear_lookup(proj_col_map, x, y)
            row_val = self._bilinear_lookup(proj_row_map, x, y)

            if np.isnan(col_val) or np.isnan(row_val):
                valid_flags[i] = False
                continue

            projector_corners[i, 0, 0] = col_val
            projector_corners[i, 0, 1] = row_val
            valid_flags[i] = True

        return projector_corners, valid_flags

    @staticmethod
    def _bilinear_lookup(image, x, y):
        """
        在浮点坐标 (x, y) 处进行双线性插值查找

        参数：
            image: 2D np.ndarray
            x, y: 浮点坐标 (x=列, y=行)

        返回：
            value: 插值后的值，无效则返回 NaN
        """
        H, W = image.shape
        x0 = int(x)
        y0 = int(y)
        x1 = x0 + 1
        y1 = y0 + 1

        if x0 < 0 or x1 >= W or y0 < 0 or y1 >= H:
            return np.nan

        fx = x - x0
        fy = y - y0

        # 获取四个邻点
        v00 = image[y0, x0]
        v01 = image[y0, x1]
        v10 = image[y1, x0]
        v11 = image[y1, x1]

        # 检查是否有 NaN
        if np.isnan(v00) or np.isnan(v01) or np.isnan(v10) or np.isnan(v11):
            return np.nan

        # 双线性插值
        val = (v00 * (1 - fx) * (1 - fy) +
               v01 * fx * (1 - fy) +
               v10 * (1 - fx) * fy +
               v11 * fx * fy)
        return val
