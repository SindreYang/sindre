# You may need to uncomment these lines on some systems:
import vtk.qt
from PyQt5.QtWidgets import (QMessageBox)
vtk.qt.QVTKRWIBase = "QGLWidget"
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets, QtCore
import qdarkstyle
from App import LMDB_Viewer

class DualLMDBViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("双LMDB查看器")
        self.setGeometry(100, 100, 1600, 900)

        # 创建中央部件和水平布局
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QHBoxLayout(central_widget)

        # 创建两个LMDB查看器实例
        self.viewer1 = LMDB_Viewer(config_file = "viewer_config1.ini")
        self.viewer2 = LMDB_Viewer(config_file = "viewer_config2.ini")

        # 添加到布局
        layout.addWidget(self.viewer1)
        layout.addWidget(self.viewer2)

        # 连接viewer1的count变化信号到viewer2的更新函数
        self.viewer1.countChanged.connect(self.on_viewer1_count_changed)

    def on_viewer1_count_changed(self, count):
        """当viewer1的count变化时，更新viewer2"""
        # 确保不超出viewer2的范围
        if count <= self.viewer2.max_count:
            self.viewer2.count = count
            self.viewer2.UpdateDisplay()

    def onClose(self):
        """保存配置到文件"""
        self.viewer1.save_config()
        self.viewer1.vtkWidget.close()
        self.viewer2.save_config()
        self.viewer2.vtkWidget.close()



def main():
    # 适应高分辨率
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)

    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=qdarkstyle.DarkPalette()))

    # 选择启动模式
    choice = QMessageBox.question(None, "启动模式","是否启用单LMDB查看器？\n\n是 - 单查看器模式\n否 - 双查看器模式",QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)

    if choice == QMessageBox.No:
        # 双查看器模式
        window = DualLMDBViewer()
    elif choice == QMessageBox.Yes:
        # 单查看器模式
        window = LMDB_Viewer()
    else:
        # 取消
        return

    window.show()
    app.exec_()


if __name__ == '__main__':
    main()
