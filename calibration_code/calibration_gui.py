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
        QSplitter, QTabWidget, QCheckBox, QComboBox,
        QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
    from PyQt5.QtGui import QFont, QPixmap, QImage, QCursor, QPainter
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


# ==================== 可缩放图像视图 ====================

class ZoomableGraphicsView(QGraphicsView):
    """支持鼠标拖拽平移和滚轮缩放的图像视图"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = None
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 5.0
        self._is_panning = False
        self._last_pos = QPointF()
        self._empty_text = None

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

        self._placeholder_label = QLabel("图像预览（点击左侧文件查看）")
        self._placeholder_label.setAlignment(Qt.AlignCenter)
        self._placeholder_label.setStyleSheet(
            "color:#888; font-size:14px;")

    def set_empty_text(self, text):
        """设置空白状态时的提示文字"""
        self._empty_text = text
        if self._pixmap_item is None:
            self._placeholder_label.setText(text)

    def set_image(self, pixmap):
        """设置要显示的图像"""
        self._scene.clear()
        if pixmap is None or pixmap.isNull():
            self._pixmap_item = None
            self._zoom_factor = 1.0
            self.resetTransform()
            return

        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())

        self.reset_view()

    def reset_view(self):
        """重置视图：自适应显示整个图像"""
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()

    def zoom_in(self):
        """放大"""
        self.scale(1.2, 1.2)
        self._zoom_factor *= 1.2

    def zoom_out(self):
        """缩小"""
        self.scale(1/1.2, 1/1.2)
        self._zoom_factor /= 1.2

    def wheelEvent(self, event):
        """滚轮缩放"""
        if event.angleDelta().y() > 0:
            self.scale(1.15, 1.15)
            self._zoom_factor *= 1.15
        else:
            self.scale(1/1.15, 1/1.15)
            self._zoom_factor /= 1.15
        event.accept()

    def mousePressEvent(self, event):
        """鼠标按下：开始拖拽"""
        if event.button() == Qt.LeftButton:
            self._is_panning = True
            self._last_pos = self.mapToScene(event.pos())
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """鼠标释放：结束拖拽"""
        if event.button() == Qt.LeftButton:
            self._is_panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        """鼠标移动：拖拽平移"""
        if self._is_panning and self._pixmap_item is not None:
            current_pos = self.mapToScene(event.pos())
            delta = current_pos - self._last_pos
            self.translate(delta.x(), delta.y())
            self._last_pos = current_pos
        super().mouseMoveEvent(event)


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

        self.cb_pattern_type = QComboBox()
        self.cb_pattern_type.addItems(["棋盘格", "圆形网格"])
        self.cb_pattern_type.setCurrentIndex(0)
        self.cb_pattern_type.currentIndexChanged.connect(self._on_pattern_type_changed)

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

        self.spin_circle_cols = QSpinBox()
        self.spin_circle_cols.setRange(3, 50)
        self.spin_circle_cols.setValue(self.config.circle_pattern_size[0])

        self.spin_circle_rows = QSpinBox()
        self.spin_circle_rows.setRange(3, 50)
        self.spin_circle_rows.setValue(self.config.circle_pattern_size[1])

        self.spin_circle_spacing = QDoubleSpinBox()
        self.spin_circle_spacing.setRange(1.0, 500.0)
        self.spin_circle_spacing.setDecimals(2)
        self.spin_circle_spacing.setValue(self.config.circle_spacing)
        self.spin_circle_spacing.setSuffix(" mm")

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

        self.lbl_cols = QLabel("棋盘格内点列数：")
        self.lbl_rows = QLabel("棋盘格内点行数：")
        self.lbl_square = QLabel("方格边长：")

        self.lbl_circle_cols = QLabel("圆形网格列数：")
        self.lbl_circle_rows = QLabel("圆形网格行数：")
        self.lbl_circle_spacing = QLabel("圆心间距：")

        form.addRow("标定板类型：", self.cb_pattern_type)
        form.addRow(self.lbl_cols, self.spin_cols)
        form.addRow(self.lbl_rows, self.spin_rows)
        form.addRow(self.lbl_square, self.spin_square)
        form.addRow(self.lbl_circle_cols, self.spin_circle_cols)
        form.addRow(self.lbl_circle_rows, self.spin_circle_rows)
        form.addRow(self.lbl_circle_spacing, self.spin_circle_spacing)
        form.addRow("格雷码位数：", self.spin_bits)
        form.addRow("投影仪宽度(px)：", self.spin_proj_w)
        form.addRow("投影仪高度(px)：", self.spin_proj_h)
        form.addRow("误差过滤阈值：", self.spin_threshold)
        form.addRow("输出目录：", out_wrapper)
        layout.addWidget(param_group)

        self._on_pattern_type_changed(0)

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
        self.preview_image_view = ZoomableGraphicsView()
        self.preview_image_view.setStyleSheet(
            "background:#f5f5f5; border:1px solid #ddd;")
        self.preview_image_view.setMinimumHeight(360)
        right_layout.addWidget(self.preview_image_view, 1)

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
        if self.cb_pattern_type.currentIndex() == 0:
            cfg.pattern_type = CalibrationConfig.PATTERN_CHESSBOARD
            cfg.pattern_size = (int(self.spin_cols.value()), int(self.spin_rows.value()))
            cfg.square_size = float(self.spin_square.value())
        else:
            cfg.pattern_type = CalibrationConfig.PATTERN_CIRCLES_GRID
            cfg.circle_pattern_size = (int(self.spin_circle_cols.value()), int(self.spin_circle_rows.value()))
            cfg.circle_spacing = float(self.spin_circle_spacing.value())
        cfg.graycode_bits = int(self.spin_bits.value())
        cfg.projector_size = (int(self.spin_proj_w.value()), int(self.spin_proj_h.value()))
        cfg.reprojection_threshold = float(self.spin_threshold.value())
        return cfg

    def _on_pattern_type_changed(self, index):
        """标定板类型切换回调：显示/隐藏对应的参数控件"""
        is_chessboard = index == 0
        self.lbl_cols.setVisible(is_chessboard)
        self.spin_cols.setVisible(is_chessboard)
        self.lbl_rows.setVisible(is_chessboard)
        self.spin_rows.setVisible(is_chessboard)
        self.lbl_square.setVisible(is_chessboard)
        self.spin_square.setVisible(is_chessboard)

        self.lbl_circle_cols.setVisible(not is_chessboard)
        self.spin_circle_cols.setVisible(not is_chessboard)
        self.lbl_circle_rows.setVisible(not is_chessboard)
        self.spin_circle_rows.setVisible(not is_chessboard)
        self.lbl_circle_spacing.setVisible(not is_chessboard)
        self.spin_circle_spacing.setVisible(not is_chessboard)

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
        """扫描所有位姿文件夹，仅对最后一张全白投影图（4N.bmp）进行角点检测并填充树状列表。"""
        root_dir = self.root_dir_edit.text().strip()
        if not root_dir or not os.path.exists(root_dir):
            QMessageBox.warning(self, "警告", "请先设置有效的图像根目录")
            return

        cfg = self._collect_config()
        self._append_log("开始扫描所有位姿并进行角点检测（仅全白投影图）...")
        try:
            from image_loader import ImageLoader
            loader = ImageLoader(cfg)
            loader.load_all_poses(root_dir)

            detector = ChessboardDetector(cfg)

            self.preview_tree.clear()
            self._preview_cache = {}
            total_corners_ok = 0

            pose_items = {}
            for pose_name in loader.pose_names:
                from PyQt5.QtWidgets import QTreeWidgetItem
                pose_item = QTreeWidgetItem([pose_name, "", "", ""])
                pose_item.setFlags(pose_item.flags())
                pose_item.setData(0, Qt.UserRole, None)
                self.preview_tree.addTopLevelItem(pose_item)
                pose_items[pose_name] = pose_item

            for pose_name in loader.pose_names:
                white_img = loader.get_white_image(pose_name)
                if white_img is None:
                    continue

                ret, corners = detector.detect_corners(white_img)

                n_corners = 0
                ok_mark = "✗"
                if ret and corners is not None:
                    n_corners = len(corners)
                    ok_mark = "✓"
                    total_corners_ok += 1

                white_path = loader.get_white_image_path(pose_name)
                fname = os.path.basename(white_path) if white_path else f"{4 * cfg.graycode_bits}.bmp"
                h, w = white_img.shape[:2]

                item = QTreeWidgetItem([fname, ok_mark, f"{w}x{h}", str(n_corners)])
                item.setData(0, Qt.UserRole, white_path)
                item.setData(1, Qt.UserRole, 1 if ret and corners is not None else 0)
                pose_items[pose_name].addChild(item)

                pose_item = pose_items[pose_name]
                pose_item.setText(1, ok_mark)
                pose_item.setText(2, f"{w}x{h}")
                pose_item.setText(3, str(n_corners))
                if ret:
                    pose_item.setExpanded(True)

            total_poses = len(loader.pose_names)
            summary = (f"扫描完成：共 {total_poses} 个位姿，"
                       f"{total_corners_ok} 个全白投影图角点检测成功。点击左侧文件可查看大图。")
            self.preview_summary.setText(summary)
            self._append_log(summary)

            self.preview_image_view.set_image(QPixmap())
            self.preview_image_status.setText("")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"角点预览失败: {str(e)}")
            self._append_log(f"[错误] {str(e)}")

    def _on_preview_tree_clicked(self, item, column):
        """点击树状列表中的项：如果是图像节点，则显示该图像 + 角点叠加。"""
        fpath = item.data(0, Qt.UserRole)
        if fpath is None or not isinstance(fpath, str) or not os.path.exists(fpath):
            return

        try:
            cfg = self._collect_config()
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                self.preview_image_status.setText(f"无法读取图像: {os.path.basename(fpath)}")
                return
            h, w = img.shape[:2]

            detector = ChessboardDetector(cfg)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = detector.detect_corners(gray)
            n_corners = len(corners) if ret and corners is not None else 0

            if ret and corners is not None:
                drawn = detector.draw_corners(img, corners, True)
            else:
                drawn = img

            if len(drawn.shape) == 2:
                rgb = cv2.cvtColor(drawn, cv2.COLOR_GRAY2RGB)
            else:
                rgb = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
            qh, qw, qch = rgb.shape
            qimg = QImage(rgb.data, qw, qh, qch * qw, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg.copy())

            self.preview_image_view.set_image(pixmap)

            ok_text = "✓ 成功" if n_corners > 0 else "✗ 未检测到角点"
            self.preview_image_status.setText(
                f"文件: {os.path.basename(fpath)}  路径: {fpath}  尺寸: {w}x{h}  "
                f"角点数: {n_corners}  状态: {ok_text}"
            )
        except Exception as e:
            self.preview_image_status.setText(f"预览错误: {str(e)}")

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
