# -*- coding: utf-8 -*-
"""
相机-投影仪标定桌面应用 GUI 模块
----------------------------------
使用 PyQt5 实现简洁的工业风格界面，功能包括：
1. 路径选择与图像批量加载
2. 标定参数配置（棋盘格、格雷码、误差阈值等）
3. 一键执行相机标定、格雷码解码、投影仪标定、双目标定
4. 实时进度条、日志输出
5. 重投影误差可视化、角点可视化
6. 标定结果表格预览
7. 一键保存 5 个 npy 参数文件
"""

import os
import sys
import numpy as np
from datetime import datetime

try:
    import cv2
except ImportError:
    print("[错误] 请先安装 opencv-python: pip install opencv-python")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLineEdit, QLabel, QFileDialog, QMessageBox,
        QProgressBar, QTextEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
        QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView,
        QSplitter, QTabWidget, QCheckBox
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    from PyQt5.QtGui import QFont, QPixmap, QImage
except ImportError:
    print("[错误] 请先安装 PyQt5: pip install PyQt5")
    sys.exit(1)

# 导入标定模块
from config import CalibrationConfig
from calibration_pipeline import CalibrationPipeline
from chessboard_detector import ChessboardDetector


# ==================== 标定线程 ====================

class CalibrationWorker(QThread):
    """后台执行标定流程的线程，避免 UI 卡住"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict, bool)  # (result_dict, success)

    def __init__(self, root_dir, output_dir, config):
        super().__init__()
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.config = config
        self._running = True

    def stop(self):
        self._running = False

    def _log(self, msg):
        self.log_signal.emit(msg)

    def _progress(self, percent, msg):
        if isinstance(percent, float):
            p = int(percent)
        else:
            p = percent
        self.progress_signal.emit(p, msg)

    def run(self):
        try:
            pipeline = CalibrationPipeline(
                self.config, self._log, self._progress)
            result = pipeline.run(self.root_dir, self.output_dir)
            self.finished_signal.emit(result, True)
        except Exception as e:
            self._log(f"[运行时错误] {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            self.finished_signal.emit({}, False)


# ==================== 主窗口 ====================

class CalibrationMainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("相机 - 投影仪联合标定工具 (v1.0)")
        self.resize(1300, 800)

        self.config = CalibrationConfig()
        self._pipeline = None
        self._stereo_result = None

        self._init_ui()
        self._apply_style()

    def _init_ui(self):
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        top = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        left_panel = self._build_left_panel()
        right_panel = self._build_right_panel()
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        top.addWidget(splitter, 1)
        main_layout.addLayout(top, 3)

        # 底部：进度条 + 日志
        progress_row = QHBoxLayout()
        self.progress_label = QLabel("就绪")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_row.addWidget(self.progress_label, 1)
        progress_row.addWidget(self.progress_bar, 3)
        main_layout.addLayout(progress_row)

        log_label = QLabel("运行日志")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(160)
        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_text)

        self.setCentralWidget(central)

        self._append_log("欢迎使用相机-投影仪联合标定工具")
        self._append_log(f"请先设置图像根目录与标定参数")

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        # 1. 图像根目录
        dir_group = QGroupBox("1. 图像路径设置")
        dir_layout = QHBoxLayout()
        self.root_dir_edit = QLineEdit()
        self.root_dir_edit.setPlaceholderText("点击右侧按钮选择标定图像根目录")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._choose_root_dir)
        dir_layout.addWidget(self.root_dir_edit, 3)
        dir_layout.addWidget(btn_browse, 1)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # 2. 参数
        param_group = QGroupBox("2. 标定参数设置")
        form = QFormLayout(param_group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setVerticalSpacing(8)

        self.spin_cols = QSpinBox()
        self.spin_cols.setRange(3, 50)
        self.spin_cols.setValue(self.config.pattern_size[0])

        self.spin_rows = QSpinBox()
        self.spin_rows.setRange(3, 50)
        self.spin_rows.setValue(self.config.pattern_size[1])

        self.spin_square = QDoubleSpinBox()
        self.spin_square.setRange(1.0, 500.0)
        self.spin_square.setDecimals(2)
        self.spin_square.setValue(self.config.square_size)
        self.spin_square.setSuffix(" mm")

        self.spin_bits = QSpinBox()
        self.spin_bits.setRange(4, 16)
        self.spin_bits.setValue(self.config.graycode_bits)

        self.spin_proj_w = QSpinBox()
        self.spin_proj_w.setRange(320, 8000)
        self.spin_proj_w.setValue(self.config.projector_size[0])

        self.spin_proj_h = QSpinBox()
        self.spin_proj_h.setRange(240, 6000)
        self.spin_proj_h.setValue(self.config.projector_size[1])

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.01, 100.0)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(self.config.reprojection_threshold)
        self.spin_threshold.setSuffix(" px")

        out_row = QHBoxLayout()
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("选择保存 npy 参数文件的目录")
        btn_out = QPushButton("...")
        btn_out.setFixedWidth(40)
        btn_out.clicked.connect(self._choose_output_dir)
        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(btn_out, 0)
        out_wrapper = QWidget()
        out_wrapper.setLayout(out_row)

        form.addRow("棋盘格内点列数：", self.spin_cols)
        form.addRow("棋盘格内点行数：", self.spin_rows)
        form.addRow("方格边长：", self.spin_square)
        form.addRow("格雷码位数：", self.spin_bits)
        form.addRow("投影仪宽度(px)：", self.spin_proj_w)
        form.addRow("投影仪高度(px)：", self.spin_proj_h)
        form.addRow("误差过滤阈值：", self.spin_threshold)
        form.addRow("输出目录：", out_wrapper)
        layout.addWidget(param_group)

        # 3. 按钮
        op_group = QGroupBox("3. 标定操作")
        op_layout = QVBoxLayout(op_group)
        op_layout.setSpacing(6)

        self.btn_preview = QPushButton("① 预览角点检测")
        self.btn_preview.clicked.connect(self._action_preview)

        self.btn_calib = QPushButton("② 开始完整标定")
        self.btn_calib.setMinimumHeight(38)
        self.btn_calib.clicked.connect(self._action_start_calib)

        self.btn_save = QPushButton("③ 保存标定结果 (npy)")
        self.btn_save.clicked.connect(self._action_save)

        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.clicked.connect(lambda: self.log_text.clear())

        op_layout.addWidget(self.btn_preview)
        op_layout.addWidget(self.btn_calib)
        op_layout.addWidget(self.btn_save)
        op_layout.addSpacing(4)
        op_layout.addWidget(self.btn_clear_log)
        layout.addWidget(op_group)

        layout.addStretch(1)

        version = QLabel("版本 v1.0  | 对标 MATLAB 双目标定工具箱")
        version.setStyleSheet("color:#888; font-size:11px;")
        version.setAlignment(Qt.AlignCenter)
        layout.addWidget(version)

        return panel

    def _build_right_panel(self):
        tabs = QTabWidget()

        # Tab 1: 角点检测可视化（文件夹 + 图像两级列表 + 大图 + 状态栏）
        preview_tab = QWidget()
        preview_layout = QVBoxLayout(preview_tab)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(6)

        # 顶部状态栏：加载结果摘要
        self.preview_summary = QLabel("尚未加载图像。请设置图像根目录后点击「① 预览角点检测」按钮扫描所有图像。")
        self.preview_summary.setStyleSheet(
            "padding:6px 10px; background:#eaf2fb; border:1px solid #cdd8e3; border-radius:3px; color:#224; font-size:13px;")
        preview_layout.addWidget(self.preview_summary)

        # 主体：QSplitter 左列表 / 右图像
        preview_splitter = QSplitter(Qt.Horizontal)

        # 左：位姿/图像树状列表
        from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem
        self.preview_tree = QTreeWidget()
        self.preview_tree.setHeaderLabels(["文件", "角点", "尺寸", "角点数"])
        self.preview_tree.setColumnWidth(0, 260)
        self.preview_tree.setColumnWidth(1, 70)
        self.preview_tree.setColumnWidth(2, 120)
        self.preview_tree.setColumnWidth(3, 90)
        self.preview_tree.setMinimumWidth(360)
        self.preview_tree.itemClicked.connect(self._on_preview_tree_clicked)
        preview_splitter.addWidget(self.preview_tree)

        # 右：图像显示区
        right_box = QWidget()
        right_layout = QVBoxLayout(right_box)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        self.preview_image_label = QLabel("图像预览")
        self.preview_image_label.setAlignment(Qt.AlignCenter)
        self.preview_image_label.setMinimumHeight(360)
        self.preview_image_label.setStyleSheet(
            "background:#f5f5f5; border:1px solid #ddd; color:#888; padding:30px; font-size:13px;")
        right_layout.addWidget(self.preview_image_label, 1)

        self.preview_image_status = QLabel("")
        self.preview_image_status.setStyleSheet(
            "padding:4px 10px; background:#f0f3f6; border:1px solid #cdd4db;"
            "border-radius:3px; color:#334; font-size:12px;")
        right_layout.addWidget(self.preview_image_status)
        preview_splitter.addWidget(right_box)
        preview_splitter.setStretchFactor(0, 2)
        preview_splitter.setStretchFactor(1, 5)
        preview_layout.addWidget(preview_splitter, 1)

        tabs.addTab(preview_tab, "角点检测预览")

        # Tab 2: 标定结果表格
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setContentsMargins(8, 8, 8, 8)
        self.summary_label = QLabel("尚未完成标定")
        self.summary_label.setStyleSheet("padding:8px; font-size:14px; font-weight:bold; color:#336;")
        result_layout.addWidget(self.summary_label)

        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["参数名称", "数值"])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        result_layout.addWidget(self.result_table, 1)

        tabs.addTab(result_widget, "标定结果")

        # Tab 3: 误差可视化
        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        error_layout.setContentsMargins(8, 8, 8, 8)
        self.error_table = QTableWidget()
        self.error_table.setColumnCount(4)
        self.error_table.setHorizontalHeaderLabels(
            ["位姿名称", "相机误差(px)", "投影仪误差(px)", "平均(px)"])
        self.error_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.error_table.verticalHeader().setVisible(False)
        error_layout.addWidget(self.error_table)
        tabs.addTab(error_widget, "误差详情")

        # Tab 4: 位姿详情
        self.pose_text = QTextEdit()
        self.pose_text.setReadOnly(True)
        self.pose_text.setPlaceholderText("完整的位姿信息和详细参数将显示在此")
        tabs.addTab(self.pose_text, "位姿详情")

        return tabs

    def _apply_style(self):
        """简洁工业风格"""
        self.setStyleSheet("""
            QMainWindow { background:#eef2f5; }
            QGroupBox {
                background:#fff;
                border:1px solid #cdd4db;
                border-radius:4px;
                margin-top:10px;
                font-weight:bold;
                color:#334;
                padding:6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
                color:#334;
            }
            QPushButton {
                background:#3b6ea5;
                color:#fff;
                border:none;
                padding:6px 14px;
                border-radius:3px;
                font-size:13px;
            }
            QPushButton:hover { background:#4a82b9; }
            QPushButton:pressed { background:#2e5a8a; }
            QPushButton:disabled { background:#a8b3bf; color:#eee; }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background:#fff;
                border:1px solid #c3cad1;
                border-radius:3px;
                padding:3px 5px;
                min-height:20px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color:#3b6ea5;
            }
            QProgressBar {
                border:1px solid #cdd4db;
                border-radius:3px;
                text-align:center;
                background:#fff;
                height:20px;
            }
            QProgressBar::chunk { background:#3b6ea5; }
            QTextEdit {
                background:#fff;
                border:1px solid #cdd4db;
                border-radius:3px;
                font-family:Consolas, "Courier New", monospace;
                font-size:12px;
                color:#333;
            }
            QTabWidget::pane { border:1px solid #cdd4db; background:#fff; }
            QTabBar::tab {
                background:#d9dee3; padding:6px 14px; margin-right:2px;
                border-top-left-radius:4px; border-top-right-radius:4px;
                color:#444;
            }
            QTabBar::tab:selected { background:#fff; color:#336; font-weight:bold; }
            QTableWidget {
                background:#fff; gridline-color:#dfe5eb;
                border:1px solid #cdd4db;
            }
            QHeaderView::section {
                background:#e9edf0; padding:5px; border:1px solid #cdd4db;
                font-weight:bold; color:#334;
            }
            QLabel { color:#333; }
        """)

    def _append_log(self, msg):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)

    def _collect_config(self):
        cfg = CalibrationConfig()
        cfg.pattern_size = (int(self.spin_cols.value()), int(self.spin_rows.value()))
        cfg.square_size = float(self.spin_square.value())
        cfg.graycode_bits = int(self.spin_bits.value())
        cfg.projector_size = (int(self.spin_proj_w.value()), int(self.spin_proj_h.value()))
        cfg.reprojection_threshold = float(self.spin_threshold.value())
        return cfg

    def _choose_root_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择标定图像根目录", "")
        if path:
            self.root_dir_edit.setText(path)
            self._append_log(f"已选择图像根目录: {path}")

    def _choose_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", "")
        if path:
            self.out_dir_edit.setText(path)
            self._append_log(f"已选择输出目录: {path}")

    def _action_preview(self):
        """扫描所有位姿文件夹下的所有图像，逐张检测角点并填充树状列表。"""
        root_dir = self.root_dir_edit.text().strip()
        if not root_dir or not os.path.exists(root_dir):
            QMessageBox.warning(self, "警告", "请先设置有效的图像根目录")
            return

        cfg = self._collect_config()
        self._append_log("开始扫描所有位姿并进行角点检测...")
        try:
            from image_loader import ImageLoader
            loader = ImageLoader(cfg)
            loader.load_all_poses(root_dir)

            detector = ChessboardDetector(cfg)

            # 清空当前树
            self.preview_tree.clear()
            self._preview_cache = {}  # key: item, value: {corners, image_path, size}
            total_images = 0
            total_corners_ok = 0
            poses_with_any_ok = 0

            # 先在树中插入所有位姿节点
            pose_items = {}
            for pose_name in loader.pose_names:
                from PyQt5.QtWidgets import QTreeWidgetItem
                pose_item = QTreeWidgetItem([pose_name, "", "", ""])
                pose_item.setFlags(pose_item.flags())
                pose_item.setData(0, Qt.UserRole, None)  # marker: pose-node
                self.preview_tree.addTopLevelItem(pose_item)
                pose_items[pose_name] = pose_item

            # 遍历所有位姿、所有图像
            import glob
            for pose_name in loader.pose_names:
                pose_path = os.path.join(root_dir, pose_name)
                if not os.path.isdir(pose_path):
                    continue
                exts = ["*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
                files = []
                for ext in exts:
                    files.extend(glob.glob(os.path.join(pose_path, ext)))
                files = sorted(set(files), key=lambda f: os.path.basename(f))
                if not files:
                    continue

                pose_corners_ok = 0
                for fpath in files:
                    fname = os.path.basename(fpath)
                    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    total_images += 1
                    h, w = img.shape[:2]

                    # 角点检测（对高分辨率图像做一点下采样加速预览）
                    working = img
                    ret, corners = detector.detect_corners(working)

                    n_corners = 0
                    ok_mark = "✗"
                    if ret and corners is not None:
                        n_corners = len(corners)
                        ok_mark = "✓"
                        total_corners_ok += 1
                        pose_corners_ok += 1

                    item = QTreeWidgetItem([fname, ok_mark, f"{w}x{h}", str(n_corners)])
                    item.setData(0, Qt.UserRole, fpath)  # 存图像绝对路径，点击时再读取
                    item.setData(1, Qt.UserRole, 1 if ret and corners is not None else 0)
                    pose_items[pose_name].addChild(item)

                # 位姿节点的状态显示
                pose_item = pose_items[pose_name]
                pose_item.setText(1, f"{pose_corners_ok}/{len(files)}")
                pose_item.setText(2, str(len(files)) + " 张")
                pose_item.setText(3, "角点")
                if pose_corners_ok > 0:
                    poses_with_any_ok += 1
                    pose_item.setExpanded(True)

            total_poses = len(loader.pose_names)
            summary = (f"扫描完成：共 {total_poses} 个位姿，{total_images} 张图像，"
                       f"{total_corners_ok} 张角点检测成功。点击左侧文件可查看大图。")
            self.preview_summary.setText(summary)
            self._append_log(summary)

            # 如果之前有图像，清空预览
            self.preview_image_label.clear()
            self.preview_image_label.setText("图像预览（点击左侧文件查看）")
            self.preview_image_status.setText("")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"角点预览失败: {str(e)}")
            self._append_log(f"[错误] {str(e)}")

    def _on_preview_tree_clicked(self, item, column):
        """点击树状列表中的项：如果是图像节点，则显示该图像 + 角点叠加。"""
        fpath = item.data(0, Qt.UserRole)
        if fpath is None or not isinstance(fpath, str) or not os.path.exists(fpath):
            # 是位姿节点，不做图像显示
            return

        try:
            cfg = self._collect_config()
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                self.preview_image_label.setText(f"无法读取图像: {os.path.basename(fpath)}")
                self.preview_image_status.setText("")
                return
            h, w = img.shape[:2]

            # 角点检测
            detector = ChessboardDetector(cfg)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = detector.detect_corners(gray)
            n_corners = len(corners) if ret and corners is not None else 0

            if ret and corners is not None:
                drawn = detector.draw_corners(img, corners, True)
            else:
                drawn = img

            # 缩放到预览标签大小
            label_w = self.preview_image_label.width()
            label_h = self.preview_image_label.height()
            max_w = max(label_w - 20, 400)
            max_h = max(label_h - 20, 300)
            scale = min(float(max_w) / w, float(max_h) / h, 1.0)
            if scale < 1.0:
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                drawn = cv2.resize(drawn, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if len(drawn.shape) == 2:
                rgb = cv2.cvtColor(drawn, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
            qh, qw, qch = rgb.shape
            qimg = QImage(rgb.data, qw, qh, qch * qw, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())
            self.preview_image_label.setPixmap(pixmap)
            self.preview_image_label.setText("")

            ok_text = "✓ 成功" if n_corners > 0 else "✗ 未检测到角点"
            self.preview_image_status.setText(
                f"文件: {os.path.basename(fpath)}  路径: {fpath}  尺寸: {w}x{h}  "
                f"角点数: {n_corners}  状态: {ok_text}"
            )
        except Exception as e:
            self.preview_image_label.setText(f"预览错误: {str(e)}")
            self.preview_image_status.setText("")

    def _action_start_calib(self):
        root_dir = self.root_dir_edit.text().strip()
        output_dir = self.out_dir_edit.text().strip() or root_dir
        if not os.path.exists(root_dir):
            QMessageBox.warning(self, "警告", "请先设置有效的图像根目录")
            return

        cfg = self._collect_config()
        self._append_log("开始完整标定流程...")

        self.btn_calib.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker = CalibrationWorker(root_dir, output_dir, cfg)
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(self._update_progress)
        self.worker.finished_signal.connect(self._on_calib_finished)
        self.worker.start()

    def _update_progress(self, percent, msg):
        self.progress_bar.setValue(max(0, min(100, int(percent))))
        self.progress_label.setText(msg)

    def _on_calib_finished(self, result, success):
        self.btn_calib.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.btn_save.setEnabled(True)

        if success and result:
            self._stereo_result = result
            self._update_result_ui(result)
            self.summary_label.setText(
                f"标定成功 | 全局 RMS: {result.get('rms', -1):.4f}px | "
                f"有效位姿: {result.get('pose_count', 0)}"
            )
            QMessageBox.information(self, "成功",
                                    "标定完成！\n参数文件已保存到指定目录。\n请在 '标定结果' 标签页查看详细数据。")
        else:
            self.summary_label.setText("标定失败，请查看日志")
            QMessageBox.critical(self, "失败", "标定失败，请查看底部日志以排查原因")

    def _update_result_ui(self, result):
        rows = []
        rows.append(("有效位姿数", str(result.get('pose_count', 0))))
        rows.append(("全局 RMS 误差 (px)", f"{result.get('rms', -1):.6f}"))
        rows.append(("平均重投影误差 (px)", f"{result.get('mean_error', -1):.6f}"))

        cm = result.get('camera_mtx')
        if cm is not None:
            rows.append(("--- 相机 ---", ""))
            rows.append(("内参 fx", f"{cm[0, 0]:.4f}"))
            rows.append(("内参 fy", f"{cm[1, 1]:.4f}"))
            rows.append(("主点 cx", f"{cm[0, 2]:.4f}"))
            rows.append(("主点 cy", f"{cm[1, 2]:.4f}"))
            rows.append(("内参矩阵", f"{np.array2string(cm, precision=4, separator=', ')}"))

        cd = result.get('camera_dist')
        if cd is not None:
            rows.append(("相机畸变系数", f"{cd.ravel()}"))

        pm = result.get('projector_mtx')
        if pm is not None:
            rows.append(("--- 投影仪 ---", ""))
            rows.append(("内参 fx", f"{pm[0, 0]:.4f}"))
            rows.append(("内参 fy", f"{pm[1, 1]:.4f}"))
            rows.append(("主点 cx", f"{pm[0, 2]:.4f}"))
            rows.append(("主点 cy", f"{pm[1, 2]:.4f}"))
            rows.append(("内参矩阵", f"{np.array2string(pm, precision=4, separator=', ')}"))

        pd = result.get('projector_dist')
        if pd is not None:
            rows.append(("投影仪畸变系数", f"{pd.ravel()}"))

        R = result.get('R')
        if R is not None:
            rows.append(("--- 外参 (相机→投影仪) ---", ""))
            rows.append(("旋转矩阵 R", f"{np.array2string(np.asarray(R), precision=6, separator=', ')}"))

        T = result.get('T')
        if T is not None:
            rows.append(("平移向量 T", f"{np.asarray(T).ravel()}"))

        self.result_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.result_table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.result_table.setItem(i, 1, QTableWidgetItem(str(v)))

        per_pose = result.get('per_pose_error', {})
        self.error_table.setRowCount(len(per_pose))
        for i, (name, err) in enumerate(per_pose.items()):
            self.error_table.setItem(i, 0, QTableWidgetItem(name))
            cam = err.get('camera_error', -1)
            proj_ = err.get('projector_error', -1)
            avg = err.get('total_error', -1)
            cam_s = f"{cam:.4f}" if cam >= 0 else "-"
            proj_s = f"{proj_:.4f}" if proj_ >= 0 else "-"
            avg_s = f"{avg:.4f}" if avg >= 0 else "-"
            self.error_table.setItem(i, 1, QTableWidgetItem(cam_s))
            self.error_table.setItem(i, 2, QTableWidgetItem(proj_s))
            self.error_table.setItem(i, 3, QTableWidgetItem(avg_s))

        detail = "======== 标定完成时记录的位姿信息 ========\n\n"
        detail += f"有效位姿: {result.get('valid_poses', [])}\n\n"
        detail += "详细误差：\n"
        for name, err in per_pose.items():
            detail += f"  {name}: {err}\n"
        self.pose_text.setText(detail)

    def _action_save(self):
        if self._stereo_result is None:
            QMessageBox.warning(self, "警告", "暂无可保存的标定结果，请先执行标定")
            return

        output_dir = self.out_dir_edit.text().strip()
        if not output_dir:
            output_dir = QFileDialog.getExistingDirectory(self, "选择保存目录", "")
            if not output_dir:
                return
            self.out_dir_edit.setText(output_dir)

        try:
            from params_saver import ParamSaver
            saver = ParamSaver(output_dir)
            saver.save_all_params(self._stereo_result)
            files_info = "\n".join([os.path.basename(f) for f in saver.saved_files])
            self._append_log(f"已保存标定参数到: {output_dir}")
            QMessageBox.information(self, "已保存", f"参数文件已保存：\n{files_info}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")
            self._append_log(f"[错误] {str(e)}")


def run_gui():
    """启动 GUI 应用"""
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = CalibrationMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_gui()
