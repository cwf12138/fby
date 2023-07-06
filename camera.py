import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QRect, QPoint,QSize

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("截图程序")
        self.resize(800, 600)

        # 创建主窗口的布局
        main_layout = QVBoxLayout()

        # 创建预览标签
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.preview_label)

        # 创建截图按钮
        self.capture_button = QPushButton("截图")
        self.capture_button.clicked.connect(self.capture_screen)
        main_layout.addWidget(self.capture_button)

        # 创建主窗口的中心部件，并设置布局
        central_widget = QLabel()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 截图相关变量
        self.is_capturing = False
        self.rect_size = QSize(560, 448)  # 矩形框的大小
        self.rect_pos = None  # 矩形框的位置
        self.drag_start_pos = None  # 拖动起始位置

    def capture_screen(self):
        # 切换截图状态
        self.is_capturing = not self.is_capturing

        if self.is_capturing:
            # 重置矩形框位置为窗口中心
            self.rect_pos = QPoint((self.width() - self.rect_size.width()) // 2, (self.height() - self.rect_size.height()) // 2)
        else:
            # 截图并保存
            screenshot = QApplication.primaryScreen().grabWindow(0, self.rect_pos.x(), self.rect_pos.y(), self.rect_size.width(), self.rect_size.height())
            screenshot.save("screenshot.png")
            print("截图已保存：screenshot.png")

        self.update()

    def paintEvent(self, event):
        # 绘制截图框
        if self.is_capturing:
            painter = QPainter(self)
            painter.setPen(Qt.blue)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(QRect(self.rect_pos, self.rect_size))
            painter.end()

    def mousePressEvent(self, event):
        # 记录拖动起始位置
        if self.is_capturing and QRect(self.rect_pos, self.rect_size).contains(event.pos()):
            self.drag_start_pos = event.pos() - self.rect_pos

    def mouseMoveEvent(self, event):
        # 拖动矩形框
        if self.is_capturing and self.drag_start_pos:
            self.rect_pos = event.pos() - self.drag_start_pos
            self.update()

    def mouseReleaseEvent(self, event):
        # 清空拖动起始位置
        if self.is_capturing:
            self.drag_start_pos = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
