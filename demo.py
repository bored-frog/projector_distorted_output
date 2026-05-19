# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import win32api
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSlider, QLabel, QFrame, QPushButton,
                             QComboBox)   # 新增 QComboBox
from PyQt5.QtCore import Qt, QPoint, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QImage, QPixmap, QPainterPath
from scipy.interpolate import LinearNDInterpolator
import dxcam
from qframelesswindow.utils import ScreenCaptureFilter

# ---------- 启用 OpenCL 加速 ----------
cv2.ocl.setUseOpenCL(True)
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL 已启用 (GPU 加速)")
else:
    print("OpenCL 不可用，将使用 CPU")

# ====================== 核心配置 ======================
OUTPUT_WIDTH = 1920
OUTPUT_HEIGHT = 1080
LEFT_FEATHER_WIDTH = 0
RIGHT_FEATHER_WIDTH = 0

CAPTURE_LEFT_HALF = True

# 11行×11列网格控制点
grid_rows = 11
grid_cols = 11
grid_points = [[QPoint() for _ in range(grid_cols)] for _ in range(grid_rows)]

def init_grid_points():
    for r in range(grid_rows):
        for c in range(grid_cols):
            x = c * OUTPUT_WIDTH // (grid_cols - 1)
            y = r * OUTPUT_HEIGHT // (grid_rows - 1)
            grid_points[r][c] = QPoint(x, y)
init_grid_points()

selected_row = -1
selected_col = -1
POINT_RADIUS = 5

src_grid = [[QPoint(c * OUTPUT_WIDTH // (grid_cols - 1), r * OUTPUT_HEIGHT // (grid_rows - 1)) for c in range(grid_cols)] for r in range(grid_rows)]

# ====================== 桌面捕获 ======================
def capture_desktop():
    hdesktop = win32gui.GetDesktopWindow()
    width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
    height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
    left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
    top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)
    hdc = win32gui.GetDC(hdesktop)
    hdc_mem = win32ui.CreateDCFromHandle(hdc)
    hdc_mem2 = hdc_mem.CreateCompatibleDC()
    data_bitmap = win32ui.CreateBitmap()
    data_bitmap.CreateCompatibleBitmap(hdc_mem, width, height)
    hdc_mem2.SelectObject(data_bitmap)
    hdc_mem2.BitBlt((0, 0), (width, height), hdc_mem, (left, top), win32con.SRCCOPY)
    signed_ints_array = data_bitmap.GetBitmapBits(True)
    img = np.frombuffer(signed_ints_array, dtype='uint8')
    img.shape = (height, width, 4)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    win32gui.DeleteObject(data_bitmap.GetHandle())
    hdc_mem2.DeleteDC()
    hdc_mem.DeleteDC()
    win32gui.ReleaseDC(hdesktop, hdc)
    return img

def create_feather_mask(w, h, left_feather, right_feather):
    mask = np.ones((h, w), dtype=np.float32)
    if left_feather > 0:
        for x in range(min(left_feather, w)):
            fade = x / left_feather
            mask[:, x] = fade
    if right_feather > 0:
        for x in range(min(right_feather, w)):
            fade = x / right_feather
            mask[:, w - 1 - x] = fade
    return mask

# ---------- 映射表 ----------
map_x = None
map_y = None
need_update_map = True

def update_warp_map():
    global map_x, map_y, need_update_map
    dst_pts = []
    src_pts_x = []
    src_pts_y = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            dst_pts.append([grid_points[r][c].x(), grid_points[r][c].y()])
            src_pts_x.append(src_grid[r][c].x())
            src_pts_y.append(src_grid[r][c].y())
    dst_pts = np.array(dst_pts, dtype=np.float32)
    src_x_arr = np.array(src_pts_x, dtype=np.float32)
    src_y_arr = np.array(src_pts_y, dtype=np.float32)

    interp_x = LinearNDInterpolator(dst_pts, src_x_arr, fill_value=-1)
    interp_y = LinearNDInterpolator(dst_pts, src_y_arr, fill_value=-1)

    x_coords = np.arange(OUTPUT_WIDTH)
    y_coords = np.arange(OUTPUT_HEIGHT)
    xv, yv = np.meshgrid(x_coords, y_coords)
    xi = np.stack([xv.ravel(), yv.ravel()], axis=-1)

    map_x = interp_x(xi).reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH).astype(np.float32)
    map_y = interp_y(xi).reshape(OUTPUT_HEIGHT, OUTPUT_WIDTH).astype(np.float32)
    need_update_map = False

feather_mask = None
update_warp_map()
_last_left_feather = -1
_last_right_feather = -1

def warp_and_fusion(frame, left_feather, right_feather, grid_pts):
    global need_update_map, map_x, map_y, feather_mask, _last_left_feather, _last_right_feather
    h_in, w_in = frame.shape[:2]

    if need_update_map:
        update_warp_map()

    # 修正：用上次缓存的羽化值做比较
    if (feather_mask is None or
            left_feather != _last_left_feather or
            right_feather != _last_right_feather):
        feather_mask = create_feather_mask(OUTPUT_WIDTH, OUTPUT_HEIGHT, left_feather, right_feather)
        _last_left_feather = left_feather
        _last_right_feather = right_feather

    margin = 0.5
    invalid = (map_x < margin) | (map_x > w_in - 1 - margin) | \
              (map_y < margin) | (map_y > h_in - 1 - margin)

    cur_map_x = map_x.copy()
    cur_map_y = map_y.copy()
    cur_map_x[invalid] = -1
    cur_map_y[invalid] = -1

    warped = cv2.remap(
        frame, cur_map_x, cur_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    warped[invalid] = 0
    warped = (warped * feather_mask[:, :, np.newaxis]).astype(np.uint8)
    return warped


class FusionCanvas(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(OUTPUT_WIDTH, OUTPUT_HEIGHT)
        self.bg_image = None
        try:
            bg = cv2.imread("dist/bg1.png")
            if bg is not None:
                h_raw, w_raw = bg.shape[:2]
                if w_raw > OUTPUT_WIDTH or h_raw > OUTPUT_HEIGHT:
                    self.bg_image = bg[0:OUTPUT_HEIGHT, 0:OUTPUT_WIDTH]
                else:
                    self.bg_image = bg
            else:
                print("警告：未找到 bg1.png，将使用黑色背景。")
        except Exception as e:
            print("加载背景图出错:", e)
        if self.bg_image is None:
            self.setStyleSheet("background-color: #111111;")
        else:
            self.setStyleSheet("")

        self.row_lock_mode = False
        self.col_lock_mode = False
        self.corner_adjust_mode = False  # 四角调节模式

        self.drag_start_mouse_pos = None
        self.drag_start_row_points = None
        self.drag_start_col_points = None

        # 四角模式相关变量
        self.selected_corner_index = -1  # 0:左上 1:右上 2:右下 3:左下
        self.corners_start = []          # 按下时四个角点位置

    def _get_corners(self):
        """返回当前网格的四个角点 [左上, 右上, 右下, 左下]"""
        return [
            grid_points[0][0],                          # 左上
            grid_points[0][grid_cols - 1],              # 右上
            grid_points[grid_rows - 1][grid_cols - 1],  # 右下
            grid_points[grid_rows - 1][0]               # 左下
        ]

    def _set_corner(self, index, point):
        """设置指定角点位置"""
        if index == 0:
            grid_points[0][0] = point
        elif index == 1:
            grid_points[0][grid_cols-1] = point
        elif index == 2:
            grid_points[grid_rows-1][grid_cols-1] = point
        elif index == 3:
            grid_points[grid_rows-1][0] = point

    def _bilinear_interpolate_grid(self, corners):
        """根据四个角点双线性插值全部网格点"""
        A, B, C, D = corners  # 左上, 右上, 右下, 左下
        for r in range(grid_rows):
            v = r / (grid_rows - 1) if grid_rows > 1 else 0
            for c in range(grid_cols):
                u = c / (grid_cols - 1) if grid_cols > 1 else 0
                x = (1-u)*(1-v)*A.x() + u*(1-v)*B.x() + u*v*C.x() + (1-u)*v*D.x()
                y = (1-u)*(1-v)*A.y() + u*(1-v)*B.y() + u*v*C.y() + (1-u)*v*D.y()
                grid_points[r][c] = QPoint(int(round(x)), int(round(y)))

    def paintEvent(self, event):
        global need_update_map
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.bg_image is not None:
            warped = warp_and_fusion(self.bg_image,
                                     LEFT_FEATHER_WIDTH, RIGHT_FEATHER_WIDTH,
                                     grid_points)
            h, w, ch = warped.shape
            qt_img = QImage(warped.data, w, h, w * ch, QImage.Format_BGR888).copy()
            painter.drawImage(0, 0, qt_img)
        else:
            painter.fillRect(self.rect(), QColor(0x11, 0x11, 0x11))

        pen = QPen(QColor(200, 200, 200, 120), 1, Qt.DashLine)
        painter.setPen(pen)
        for r in range(grid_rows):
            path = QPainterPath()
            path.moveTo(grid_points[r][0])
            for c in range(1, grid_cols):
                path.lineTo(grid_points[r][c])
            painter.drawPath(path)
        for c in range(grid_cols):
            path = QPainterPath()
            path.moveTo(grid_points[0][c])
            for r in range(1, grid_rows):
                path.lineTo(grid_points[r][c])
            painter.drawPath(path)

        # 绘制控制点
        for r in range(grid_rows):
            for c in range(grid_cols):
                pt = grid_points[r][c]
                # 四角模式下角点特殊颜色
                if self.corner_adjust_mode:
                    if (r == 0 and c == 0) or (r == 0 and c == grid_cols-1) or \
                       (r == grid_rows-1 and c == 0) or (r == grid_rows-1 and c == grid_cols-1):
                        painter.setBrush(QColor(0, 210, 255))  # 亮蓝表示可调节
                    else:
                        painter.setBrush(QColor(100, 100, 100))  # 灰色表示失效
                elif self.col_lock_mode and selected_col == c:
                    painter.setBrush(QColor(255, 60, 60))
                elif self.row_lock_mode and selected_row == r:
                    painter.setBrush(QColor(255, 60, 60))
                elif selected_row == r and selected_col == c:
                    painter.setBrush(QColor(255, 60, 60))
                else:
                    painter.setBrush(QColor(255, 210, 0))
                painter.drawEllipse(pt, POINT_RADIUS, POINT_RADIUS)

    def mousePressEvent(self, e):
        global selected_row, selected_col
        selected_row = -1
        selected_col = -1

        if e.button() == Qt.LeftButton:
            pos = e.pos()

            # ---- 四角调节模式 ----
            if self.corner_adjust_mode:
                corners = self._get_corners()
                for i, pt in enumerate(corners):
                    if (pos - pt).manhattanLength() < 15:
                        self.selected_corner_index = i
                        self.corners_start = [QPoint(p) for p in corners]  # 保存副本
                        self.drag_start_mouse_pos = pos
                        self.update()
                        return
                return

            # ---- 原始功能 ----
            for r in range(grid_rows):
                for c in range(grid_cols):
                    if (pos - grid_points[r][c]).manhattanLength() < 15:
                        selected_row = r
                        selected_col = c
                        if self.col_lock_mode:
                            self.drag_start_mouse_pos = pos
                            self.drag_start_col_points = [
                                QPoint(grid_points[i][selected_col]) for i in range(grid_rows)
                            ]
                            self.update()
                            return
                        if self.row_lock_mode:
                            self.drag_start_mouse_pos = pos
                            self.drag_start_row_points = [
                                QPoint(grid_points[selected_row][i]) for i in range(grid_cols)
                            ]
                            self.update()
                            return
                        self.update()
                        return
        self.update()

    def mouseMoveEvent(self, e):
        global selected_row, selected_col, need_update_map

        # ---- 四角调节模式 ----
        if self.corner_adjust_mode and self.selected_corner_index >= 0:
            delta = e.pos() - self.drag_start_mouse_pos
            # 更新选中角点的新位置
            new_pt = self.corners_start[self.selected_corner_index] + delta
            new_x = max(0, min(OUTPUT_WIDTH, new_pt.x()))
            new_y = max(0, min(OUTPUT_HEIGHT, new_pt.y()))
            # 构造新的四个角点列表
            updated_corners = [QPoint(p) for p in self.corners_start]
            updated_corners[self.selected_corner_index] = QPoint(new_x, new_y)
            # 用双线性插值刷新全部网格点
            self._bilinear_interpolate_grid(updated_corners)
            need_update_map = True
            self.update()
            return

        # ---- 原有功能 ----
        if selected_row < 0 and selected_col < 0:
            return

        if self.col_lock_mode and self.drag_start_col_points is not None and selected_col >= 0:
            delta = e.pos() - self.drag_start_mouse_pos
            for r in range(grid_rows):
                new_pt = self.drag_start_col_points[r] + delta
                new_x = max(0, min(OUTPUT_WIDTH, new_pt.x()))
                new_y = max(0, min(OUTPUT_HEIGHT, new_pt.y()))
                grid_points[r][selected_col] = QPoint(new_x, new_y)
            need_update_map = True
            self.update()
            return

        if self.row_lock_mode and self.drag_start_row_points is not None and selected_row >= 0:
            delta = e.pos() - self.drag_start_mouse_pos
            for c in range(grid_cols):
                new_pt = self.drag_start_row_points[c] + delta
                new_x = max(0, min(OUTPUT_WIDTH, new_pt.x()))
                new_y = max(0, min(OUTPUT_HEIGHT, new_pt.y()))
                grid_points[selected_row][c] = QPoint(new_x, new_y)
            need_update_map = True
            self.update()
            return

        if selected_col >= 0:
            new_x = max(0, min(OUTPUT_WIDTH, e.pos().x()))
            new_y = max(0, min(OUTPUT_HEIGHT, e.pos().y()))
            grid_points[selected_row][selected_col] = QPoint(new_x, new_y)
            need_update_map = True
            self.update()

    def mouseReleaseEvent(self, e):
        global selected_row, selected_col
        selected_row = -1
        selected_col = -1
        self.drag_start_mouse_pos = None
        self.drag_start_row_points = None
        self.drag_start_col_points = None
        # 四角模式释放
        self.selected_corner_index = -1
        self.corners_start = []
        self.update()


class GeometryWindow(QMainWindow):
    closed = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("投影控制器 - 几何校正")
        self.setGeometry(100, 100, OUTPUT_WIDTH + 40, OUTPUT_HEIGHT + 40)
        self.canvas = FusionCanvas()
        self.setCentralWidget(self.canvas)
        self.current_screen_idx = 0

    def set_screen(self, idx):
        """设置窗口将显示的屏幕索引"""
        self.current_screen_idx = idx
        if self.isVisible():
            self.show_on_screen(idx)

    def show_on_screen(self, idx):
        screens = QApplication.screens()
        if 0 <= idx < len(screens):
            screen = screens[idx]
            self.setGeometry(screen.geometry())  # 安全移至目标屏幕
            self.winId()  # 确保句柄
            if self.windowHandle() and hasattr(self.windowHandle(), 'setScreen'):
                self.windowHandle().setScreen(screen)  # Qt 5.15+ 精确设置
        self.showFullScreen()
        self.raise_()

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()

class CaptureThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    def __init__(self, output_idx=0):
        super().__init__()
        self.running = True
        self.output_idx = output_idx
        self.camera = None

    def run(self):
        self.camera = dxcam.create(output_idx=self.output_idx, output_color="BGR")
        if self.camera is None:
            print(f"无法创建索引 {self.output_idx} 的 DXGI 摄像机")
            self.running = False
            return
        self.camera.start(target_fps=60, video_mode=True)
        while self.running:
            frame = self.camera.get_latest_frame()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            if w == 3840:
                if CAPTURE_LEFT_HALF:
                    frame = frame[:, 0:1920]
                else:
                    frame = frame[:, 1920:3840]
            elif w != OUTPUT_WIDTH or h != OUTPUT_HEIGHT:
                frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

            warped = warp_and_fusion(frame, LEFT_FEATHER_WIDTH, RIGHT_FEATHER_WIDTH, grid_points)
            self.frame_ready.emit(warped)
        if self.camera is not None:
            self.camera.stop()

    def set_output_index(self, idx):
        """停止当前捕获并切换到新显示器"""
        if self.output_idx == idx and self.camera is not None:
            return
        # 停止当前线程循环
        self.running = False
        self.wait()  # 等待 run 退出
        # 更新索引并重启
        self.output_idx = idx
        self.running = True
        self.start()

    def stop(self):
        self.running = False
        self.wait()


class OutputWindow(QMainWindow):
    closed = pyqtSignal()
    def __init__(self, output_idx=0):
        super().__init__()
        self.setWindowTitle("投影输出（DXGI 高速捕获）")
        self.setFixedSize(OUTPUT_WIDTH, OUTPUT_HEIGHT)
        self.label = QLabel()
        self.setCentralWidget(self.label)

        self.capture_thread = CaptureThread(output_idx=output_idx)
        self.capture_thread.frame_ready.connect(self.update_display)
        self.capture_thread.start()

        self.screen_filter = ScreenCaptureFilter(self)
        self.installEventFilter(self.screen_filter)
        self.current_screen_idx = output_idx

    def set_screen(self, idx):
        self.current_screen_idx = idx
        if self.isVisible():
            screens = QApplication.screens()
            if 0 <= idx < len(screens):
                self.setScreen(screens[idx])

    def update_display(self, img_bgr):
        h, w, ch = img_bgr.shape
        qt_img = QImage(img_bgr.data, w, h, ch * w, QImage.Format_BGR888)
        self.label.setPixmap(QPixmap.fromImage(qt_img))

    def switch_display(self, idx):
        """供控制面板调用，切换副屏"""
        self.capture_thread.set_output_index(idx)
        self.set_screen(idx)

    # 显示窗口时使用同一套逻辑
    def show_on_screen(self, idx):
        screens = QApplication.screens()
        if 0 <= idx < len(screens):
            screen = screens[idx]
            # 方法一：通过 windowHandle 设置（适用于 Qt 5.1+）
            self.winId()  # 确保原生窗口已创建
            if self.windowHandle():
                self.windowHandle().setScreen(screen)
            # 方法二（更稳健）：直接移动窗口到目标屏幕的矩形区域
            self.setGeometry(screen.geometry())
        self.showFullScreen()
        self.raise_()

    def closeEvent(self, event):
        self.capture_thread.stop()
        self.closed.emit()
        event.accept()


class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("投影控制面板")
        self.setFixedSize(800, 600)
        self.current_display_idx = 0  # 初始为0

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 左羽化
        left_layout = QHBoxLayout()
        self.left_label = QLabel(f"左羽化宽度：{LEFT_FEATHER_WIDTH}")
        self.left_slider = QSlider(Qt.Horizontal)
        self.left_slider.setRange(0, 200)
        self.left_slider.setValue(LEFT_FEATHER_WIDTH)
        self.left_slider.valueChanged.connect(self.update_left_feather)
        left_layout.addWidget(self.left_label)
        left_layout.addWidget(self.left_slider)
        layout.addLayout(left_layout)

        # 右羽化
        right_layout = QHBoxLayout()
        self.right_label = QLabel(f"右羽化宽度：{RIGHT_FEATHER_WIDTH}")
        self.right_slider = QSlider(Qt.Horizontal)
        self.right_slider.setRange(0, 200)
        self.right_slider.setValue(RIGHT_FEATHER_WIDTH)
        self.right_slider.valueChanged.connect(self.update_right_feather)
        right_layout.addWidget(self.right_label)
        right_layout.addWidget(self.right_slider)
        layout.addLayout(right_layout)

        # 副屏选择行（新增）
        display_layout = QHBoxLayout()
        display_layout.addWidget(QLabel("选择副屏："))
        self.display_combo = QComboBox()
        # 获取可用的 DXGI 输出列表
        try:
            outputs = dxcam.output_info()
            if outputs:
                for idx, data in enumerate(outputs.split('\n')):
                    if data == '':
                        continue
                    primary = ''
                    for p in data.split(' '):
                        if 'Primary' in p:
                            p_status = p.split(':')[1]
                            if p_status == 'True':
                                primary = '主屏'
                    self.display_combo.addItem(f"屏(idx={idx}){primary} ", idx)
            else:
                self.display_combo.addItem("Default (0)", 0)
        except Exception as e:
            print("枚举显示器失败:", e)
            self.display_combo.addItem("Default (0)", 0)
        self.display_combo.currentIndexChanged.connect(self.on_display_changed)
        display_layout.addWidget(self.display_combo)
        layout.addLayout(display_layout)

        # 按钮布局
        btn_layout = QHBoxLayout()
        self.btn_controller = QPushButton("显示投影控制器")
        self.btn_controller.clicked.connect(self.toggle_geometry_window)
        self.btn_output = QPushButton("显示投影输出")
        self.btn_output.clicked.connect(self.toggle_output_window)

        self.btn_row_lock = QPushButton("开启行锁定模式")
        self.btn_row_lock.clicked.connect(self.toggle_row_lock_mode)
        self.btn_col_lock = QPushButton("开启列锁定模式")
        self.btn_col_lock.clicked.connect(self.toggle_col_lock_mode)
        self.btn_corner_adjust = QPushButton("开启四角调节模式")
        self.btn_corner_adjust.clicked.connect(self.toggle_corner_adjust_mode)

        btn_layout.addWidget(self.btn_controller)
        btn_layout.addWidget(self.btn_output)
        btn_layout.addWidget(self.btn_row_lock)
        btn_layout.addWidget(self.btn_col_lock)
        btn_layout.addWidget(self.btn_corner_adjust)
        layout.addLayout(btn_layout)

        layout.addStretch()

        self.geometry_win = None
        self.output_win = None

        self.screen_filter = ScreenCaptureFilter(self)
        self.installEventFilter(self.screen_filter)

    def update_left_feather(self, v):
        global LEFT_FEATHER_WIDTH
        LEFT_FEATHER_WIDTH = v
        self.left_label.setText(f"左羽化宽度：{v}")
        if self.geometry_win and self.geometry_win.isVisible():
            self.geometry_win.canvas.update()

    def update_right_feather(self, v):
        global RIGHT_FEATHER_WIDTH
        RIGHT_FEATHER_WIDTH = v
        self.right_label.setText(f"右羽化宽度：{v}")
        if self.geometry_win and self.geometry_win.isVisible():
            self.geometry_win.canvas.update()

    def on_display_changed(self, index):
        """副屏下拉框变化时，切换输出窗口的捕获源"""
        idx = self.display_combo.itemData(index)
        self.current_display_idx = idx
        print(f"切换到副屏索引: {idx}")
        # 如果输出窗口正在显示，立即移动它
        if self.output_win and self.output_win.isVisible():
            self.output_win.switch_display(idx)

    def toggle_row_lock_mode(self):
        canvas = self._get_canvas()
        if canvas:
            canvas.row_lock_mode = not canvas.row_lock_mode
            if canvas.row_lock_mode:
                if canvas.col_lock_mode:
                    canvas.col_lock_mode = False
                    self.btn_col_lock.setText("开启列锁定模式")
                if canvas.corner_adjust_mode:
                    canvas.corner_adjust_mode = False
                    self.btn_corner_adjust.setText("开启四角调节模式")
                self.btn_row_lock.setText("关闭行锁定模式")
            else:
                self.btn_row_lock.setText("开启行锁定模式")
                self._reset_selection(canvas)

    def toggle_col_lock_mode(self):
        canvas = self._get_canvas()
        if canvas:
            canvas.col_lock_mode = not canvas.col_lock_mode
            if canvas.col_lock_mode:
                if canvas.row_lock_mode:
                    canvas.row_lock_mode = False
                    self.btn_row_lock.setText("开启行锁定模式")
                if canvas.corner_adjust_mode:
                    canvas.corner_adjust_mode = False
                    self.btn_corner_adjust.setText("开启四角调节模式")
                self.btn_col_lock.setText("关闭列锁定模式")
            else:
                self.btn_col_lock.setText("开启列锁定模式")
                self._reset_selection(canvas)

    def toggle_corner_adjust_mode(self):
        canvas = self._get_canvas()
        if canvas:
            canvas.corner_adjust_mode = not canvas.corner_adjust_mode
            if canvas.corner_adjust_mode:
                if canvas.row_lock_mode:
                    canvas.row_lock_mode = False
                    self.btn_row_lock.setText("开启行锁定模式")
                if canvas.col_lock_mode:
                    canvas.col_lock_mode = False
                    self.btn_col_lock.setText("开启列锁定模式")
                self.btn_corner_adjust.setText("关闭四角调节模式")
            else:
                self.btn_corner_adjust.setText("开启四角调节模式")
                self._reset_selection(canvas)

    def _get_canvas(self):
        if self.geometry_win:
            return self.geometry_win.canvas
        return None

    def _reset_selection(self, canvas):
        global selected_row, selected_col
        selected_row = -1
        selected_col = -1
        canvas.drag_start_mouse_pos = None
        canvas.drag_start_row_points = None
        canvas.drag_start_col_points = None
        canvas.selected_corner_index = -1
        canvas.corners_start = []
        canvas.update()

    def toggle_geometry_window(self):
        if self.geometry_win is None:
            self.geometry_win = GeometryWindow()
            self.geometry_win.closed.connect(self.on_geometry_closed)
        if self.geometry_win.isVisible():
            self.geometry_win.close()
        else:
            if self.output_win and self.output_win.isVisible():
                self.output_win.close()
            self.geometry_win.show_on_screen(self.current_display_idx)
            self.btn_controller.setText("隐藏投影控制器")

    def toggle_output_window(self):
        if self.output_win is None:
            idx = self.current_display_idx
            self.output_win = OutputWindow(output_idx=idx)
            self.output_win.closed.connect(self.on_output_closed)
        if self.output_win.isVisible():
            self.output_win.close()
        else:
            if self.geometry_win and self.geometry_win.isVisible():
                self.geometry_win.close()
            self.output_win.show_on_screen(self.current_display_idx)
            self.btn_output.setText("隐藏投影输出")

    def on_geometry_closed(self):
        self.geometry_win = None
        self.btn_controller.setText("显示投影控制器")

    def on_output_closed(self):
        self.output_win = None
        self.btn_output.setText("显示投影输出")

    def closeEvent(self, event):
        if self.geometry_win: self.geometry_win.close()
        if self.output_win: self.output_win.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    control_win = ControlWindow()
    control_win.show()
    sys.exit(app.exec_())