import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from test import main,result_data
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片上传程序")
        self.resize(800, 400)
        self.image_path=""

        # 创建主窗口的布局
        main_layout = QHBoxLayout()

        # 创建左侧部分的布局
        left_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("border: 1px solid black;")
        left_layout.addWidget(self.image_label)

        # 创建右侧部分的布局
        right_layout = QVBoxLayout()
        self.result_label = QLabel("结果")
        self.result_label.setStyleSheet("font-size: 16px;")
        right_layout.addWidget(self.result_label)

        # 将左右两个部分的布局添加到主布局中
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

        # 创建主窗口的中心部件，并设置布局
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 创建上传按钮，并连接槽函数
        self.upload_button = QPushButton("上传图片")
        self.upload_button.clicked.connect(self.upload_image)
        left_layout.addWidget(self.upload_button)

    def upload_image(self):
        file_dialog = QFileDialog()
        image_path, _ = file_dialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if image_path:
            pixmap = QPixmap(image_path)
            self.image_label.setPixmap(pixmap.scaled(300, 300, aspectRatioMode=2))

            # 更新结果标签中的信息
            #self.result_label.setText(f"图片路径：{image_path}")
            self.image_path=image_path
            self.write_data()
            main()
            self.result_label.setText("{}".format(result_data()))
            print(image_path)
    def write_data(self):
        # 打开文件并追加内容
        file_path = "./dataset/test.txt"  # 文件路径
        path_photo=self.image_path
        content = "\n{}\t2".format(path_photo)
        print(content)
        # 打开文件并追加内容 
        with open(file_path, "a") as file:
            file.write(content)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
