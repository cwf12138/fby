import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QCursor
from PyQt5.QtCore import Qt, QRect, QPoint


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
        self.start_pos = None
        self.end_pos = None

    def capture_screen(self):
        # 开始截图
        self.is_capturing = True
        self.setMouseTracking(True)

    def paintEvent(self, event):
        # 绘制截图框
        if self.is_capturing and self.start_pos and self.end_pos:
            painter = QPainter(self)
            painter.setPen(Qt.red)
            painter.drawRect(QRect(self.start_pos, self.end_pos))
            painter.end()

    def mousePressEvent(self, event):
        # 记录起始位置
        if self.is_capturing:
            self.start_pos = event.pos()

    def mouseMoveEvent(self, event):
        # 更新截图框
        if self.is_capturing and self.start_pos:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        # 停止截图并保存
        if self.is_capturing and self.start_pos and self.end_pos:
            self.is_capturing = False
            screenshot = QApplication.primaryScreen().grabWindow(0)
            screenshot = screenshot.copy(QRect(self.start_pos, self.end_pos))
            screenshot.save("screenshot.png")
            print("截图已保存：screenshot.png")
            self.setMouseTracking(False)
            self.start_pos = None
            self.end_pos = None
            self.update()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
