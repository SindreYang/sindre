import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QMessageBox, QSplitter,
                             QStatusBar, QToolBar, QListWidget, QDialog, QLabel,
                             QDockWidget, QAction, QMenuBar, QMenu, QComboBox,
                             QColorDialog, QLineEdit, QGridLayout, QListWidgetItem,
                             QSlider, QCheckBox, QFormLayout, QGroupBox, QFrame, QPlainTextEdit)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSize, QTimer, QDateTime
from PyQt5.QtGui import QKeySequence, QColor, QIcon, QBrush, QPen, QPixmap, QFont
import numpy as np
import vedo
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from pathlib import Path
from sindre.utils3d.Annotator3D.components.KeypointAnnotator import KeypointAnnotator
from sindre.utils3d.Annotator3D.components.LabelDock import LabelDockWidget
from sindre.utils3d.Annotator3D.components.SegAnnotator import SplineSegmentAnnotator
from sindre.utils3d.Annotator3D.components.CutAnnotator import CutAnnotator
from sindre.utils3d.algorithm import save_np_json
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三角网格标注工具")
        self.setGeometry(100, 100, 1600, 900)
        
        # 初始化变量
        self.current_annotator = None
        self.current_annotator_name = ''
        self.mesh = None
        self.label_dock = LabelDockWidget(self)
        self.model_files = []
        self.current_annotation_data = None
        self.current_path = None
        self.current_dir = None
        self.vp =None

        # 初始化UI
        self.init_ui()
        self.check_config(save=False )
    
    def init_ui(self):
        """初始化UI"""
        # 左侧Dock：文件列表 + 标签管理
        self.left_dock = QDockWidget("文件与标签管理", self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.left_dock)

        # 左侧Dock内容布局
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # 模型文件列表
        file_group = QGroupBox("模型文件列表")
        file_layout = QVBoxLayout(file_group)
        self.file_list_widget = QListWidget()
        self.file_list_widget.setMinimumHeight(120)
        file_layout.addWidget(self.file_list_widget)
        left_layout.addWidget(file_group)

        # 标签管理
        label_group = QGroupBox("标签管理")
        label_layout = QVBoxLayout(label_group)
        label_layout.addWidget(self.label_dock)
        left_layout.addWidget(label_group)

        self.left_dock.setWidget(left_widget)
        self.left_dock.setMinimumWidth(250)

        # 主菜单
        self.init_menus()
        
        # 中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D视图
        # 创建VTK部件
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtk_widget)
        self.vp = vedo.Plotter(N=1, qt_widget=self.vtk_widget)
        self.vp.show(bg="white", title="3D模型视图")
        self.vp.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 工具Dock窗口（可调整）
        self.tool_dock = QDockWidget("标注工具", self)
        self.tool_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.tool_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.tool_dock)
        
        # 设置工具dock大小
        self.tool_dock.setMinimumWidth(300)
        self.tool_dock.setMaximumWidth(500)


        # 日志
        self.log_dock = QDockWidget("日志", self)
        self.log_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.plain_text = QPlainTextEdit()
        self.plain_text.setReadOnly(True)  # 设为只读（根据需求调整）
        self.plain_text.setFont(QFont("Consolas", 9))  # 等宽字体，适合日志/代码
        self.plain_text.setStyleSheet("background-color: #f8f8f8;")
        self.log_dock.setWidget(self.plain_text)
        self.addDockWidget(Qt.RightDockWidgetArea,self.log_dock)

        # 初始化信号
        self.init_signals()


    def init_menus(self):
        """创建菜单"""
        self.main_menu_bar = QMenuBar()
        
        # 文件菜单
        self.file_menu = QMenu("文件", self)
        self.open_action = QAction("打开文件", self)
        self.open_dir_action = QAction("打开目录", self)
        self.exit_action = QAction("退出", self)
        
        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.open_dir_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)
        
        # 标注菜单
        self.annotate_menu = QMenu("标注", self)
        self.keypoints_action = QAction("关键点标注", self)
        self.segment_action = QAction("分割标注", self)
        self.bbox_action = QAction("边界框标注", self)
        self.transform_action =QAction("网格处理-变换位置", self)
        self.sculpt_action =QAction("网格处理-局部塑形", self)
        self.cut_action =QAction("网格处理-裁剪", self)

        self.annotate_menu.addAction(self.segment_action)
        self.annotate_menu.addAction(self.keypoints_action)
        self.annotate_menu.addAction(self.transform_action)
        self.annotate_menu.addAction(self.cut_action)
        self.annotate_menu.addAction(self.bbox_action)
        self.annotate_menu.addAction(self.sculpt_action)
        
        # 视图菜单
        self.view_menu = QMenu("视图", self)
        self.clear_file_action = QAction("清空文件列表", self)
        self.toggle_tool_dock_action = QAction("显示/隐藏面板", self)

        self.view_menu.addAction(self.toggle_tool_dock_action)
        self.view_menu.addAction(self.clear_file_action)

        # 帮助菜单
        self.help_menu = QMenu("帮助", self)
        
        # 添加到菜单栏
        self.main_menu_bar.addMenu(self.file_menu)
        self.main_menu_bar.addMenu(self.annotate_menu)
        self.main_menu_bar.addMenu(self.view_menu)
        self.main_menu_bar.addMenu(self.help_menu)
        
        self.setMenuBar(self.main_menu_bar)
    
    def init_signals(self):
        """初始化信号连接"""
        # 连接标签Dock的信号
        self.label_dock.signals.signal_info.connect(self.update_status)
        # 文件菜单
        self.open_action.triggered.connect(self.append_file)
        self.open_dir_action.triggered.connect(self.append_dir)
        self.exit_action.triggered.connect(self.close)
        
        # 标注菜单
        self.keypoints_action.triggered.connect(self.start_keypoint_annotation)
        self.segment_action.triggered.connect(self.start_segment_annotation)
        self.bbox_action.triggered.connect(self.start_bbox_annotation)
        self.transform_action.triggered.connect(self.start_transform_annotation)
        self.cut_action.triggered.connect(self.start_cut_annotation)
        
        # 视图菜单
        self.toggle_tool_dock_action.triggered.connect(self.toggle_tool_dock)
        self.clear_file_action.triggered.connect(self.clear_file_list)


        # 文件列表点击事件
        self.file_list_widget.itemClicked.connect(self.on_file_selected)
        

    def clear_file_list(self):
        """清空文件目录"""
        self.model_files = []
        self.file_list_widget.clear()
    
    def append_file(self):
        """打开3D模型文件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "打开3D模型文件", "",
            "3D模型文件 (*.ply *.obj *.stl *.vtk *.vtp);;所有文件 (*.*)"
        )
        if file_path:
            if file_path not in self.model_files:
                self.model_files.append(file_path)
            # 更新文件列表
            item = QListWidgetItem(os.path.basename(file_path))
            item.setData(Qt.UserRole, file_path)
            self.file_list_widget.addItem(item)
            self.on_file_selected(item)

    def append_dir(self):
        """打开目录，加载模型文件列表"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择模型目录")
        if dir_path:
            self.current_dir = dir_path
            # 扫描支持的模型格式
            supported_ext = ['.stl', '.ply', '.vtp', '.obj', '.vtk']
            for file in os.listdir(dir_path):
                if Path(file).suffix.lower() in supported_ext:
                    new_path =os.path.join(dir_path, file)
                    if new_path not in self.model_files:
                        self.model_files.append(new_path)

            # 更新文件列表
            for file_path in self.model_files:
                item = QListWidgetItem(os.path.basename(file_path))
                item.setData(Qt.UserRole, file_path)
                self.file_list_widget.addItem(item)

            self.update_status(f"加载目录: {dir_path} | 模型数量: {len(self.model_files)}")

    def on_file_selected(self, item):
        """选择文件列表中的模型"""
        file_path = item.data(Qt.UserRole)
        if file_path:
            if self.current_path is not None:
                reply = QMessageBox.question(
                    self, "确认切换", "确定要切换数据吗？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    if self.current_annotator:
                        self.close_annotation(None)
                    self.update_render(file_path)
            else:

                self.update_render(file_path)



    def update_status(self, text):
        """更新状态栏"""
        self.status_bar.showMessage(text)
        # 自动滚动到底部
        self.plain_text.appendPlainText(f"[{QDateTime.currentDateTime().toString('HH:mm:ss')}] {text}")
        scroll_bar = self.plain_text.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def update_render(self,file_path):
        self.current_path = file_path

        try:
            # 清除当前视图
            self.vp.clear(deep=True)
            # 读取模型
            self.mesh = vedo.Mesh(file_path)
            # 显示模型
            self.vp.add(self.mesh)
            self.vp.reset_camera()
            self.vp.render()
            # 更新状态
            self.update_status(f"已加载模型: {os.path.basename(file_path)},\n\t顶点数量: {self.mesh.npoints},面数量: {self.mesh.ncells}\n")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
            self.update_status("加载模型失败")

    def setup_annotator_signals(self):
        """设置标注器通用信号连接"""
        if self.current_annotator:
            self.current_annotator.signals.signal_info.connect(self.update_status)
            self.current_annotator.signals.signal_dock.connect(self.tool_dock.setWidget)
            self.current_annotator.signals.signal_close.connect(self.close_annotation)

    def start_keypoint_annotation(self):
        """开始关键点标注"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先打开一个3D模型")
            return

        
        # 创建关键点标注器，传入标签Dock组件
        self.current_annotator = KeypointAnnotator(label_dock=self.label_dock)
        self.current_annotator_name='keypoint'
        
        # 连接信号
        self.setup_annotator_signals()
        
        # 设置UI
        self.current_annotator.setup_ui()
        
        # 渲染标注器
        self.current_annotator.render(self.vp)
    
    def start_segment_annotation(self):
        """开始分割标注"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先打开一个3D模型")
            return

        # 创建关键点标注器，传入标签Dock组件
        self.current_annotator = SplineSegmentAnnotator(label_dock=self.label_dock)
        self.current_annotator_name='Spline'
        
        # 连接信号
        self.setup_annotator_signals()
        # 设置UI
        self.current_annotator.setup_ui()
        
        # 渲染标注器
        self.current_annotator.render_ui(self.vp)
        
    
    def start_bbox_annotation(self):
        """开始边界框标注"""
        QMessageBox.information(self, "提示", "边界框标注功能开发中...")
        
    def start_cut_annotation(self):
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先打开一个3D模型")
            return

        # 创建标注器
        self.current_annotator = CutAnnotator()
        self.current_annotator_name='cut'
        
        # 连接信号
        self.setup_annotator_signals()
        
        # 设置UI
        self.current_annotator.setup_ui()
        
        # 渲染标注器
        self.current_annotator.render(self.vp)
        
    def start_transform_annotation(self):
        """变换位置"""
        import pyvista as pv
        if self.mesh is None:
             QMessageBox.information(self, "提示", "请先导入mesh...")
        def set_matrix(mat):
            self.matrix =mat
        def save_mesh():
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出mesh", "transform_mesh.ply", "所有文件 (*.*)"
            )
            if file_path:
                np.savetxt(file_path+"_matrix.txt",np.array(self.matrix))
                self.mesh.apply_transform(self.matrix).write(file_path)
            
                   
        self.setHidden(True)
        pl = pv.Plotter(title="s-->save mesh")
        actor = pl.add_mesh(self.mesh.dataset)
        widget = pl.add_affine_transform_widget(
            actor,
            scale=1.0,
            axes=np.array(
                (
                    (-1, 0, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                ),
            ),
            release_callback=set_matrix,
            
        )
        #axes = pl.add_axes_at_origin()
        axes = pv.AxesAssemblySymmetric(scale=self.mesh.bounds().max())
        pl.add_actor(axes)
        pl.add_axes()
        
        
        pl.add_key_event("s", save_mesh)
        pl.show()
        self.setHidden(False)
        
    



    def check_config(self,save=True):
        if save:
            info ={
                "model_files":self.model_files,
                "label_info":self.label_dock.label_manager.labels,
            }
            np.save("LabelConfig.npy", info)
        else:
            if os.path.exists("LabelConfig.npy"):
                reply = QMessageBox.question(
                    self, "检测到旧配置", "是否导入之前配置？",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    data = np.load("LabelConfig.npy",allow_pickle=True).item()
                    self.model_files =data["model_files"]

                    # 更新文件列表
                    self.file_list_widget.clear()
                    for file_path in self.model_files:
                        item = QListWidgetItem(os.path.basename(file_path))
                        item.setData(Qt.UserRole, file_path)
                        self.file_list_widget.addItem(item)

                    # 更新标签
                    self.label_dock.label_manager.labels =data["label_info"]
                    self.label_dock.label_manager.reset_all_labels()
                    self.label_dock.update_label_buttons()
                    self.update_status(f"成功导入配置: {os.path.abspath("LabelConfig.npy")}")


    def close_annotation(self, data):
        if data:
            save_path =self.current_path+f'{self.current_annotator_name}.json'
            self.update_status(f"保存标注信息：{save_path}")
            save_np_json(save_path,data)
        if self.current_annotator:
            self.current_annotator.disconnect()
            self.current_annotator.clean()
            self.current_annotator.close()
            self.tool_dock.setWidget(None)
            self.current_annotator = None
            self.current_path=None
        # 自动选择下一个
        next_index = ( self.file_list_widget.row(self.file_list_widget.currentItem()) + 1) % self.file_list_widget.count()
        next_item = self.file_list_widget.item(next_index)
        if next_item:
            self.file_list_widget.setCurrentItem(next_item)
            self.on_file_selected(next_item)

    
    def toggle_tool_dock(self):
        """切换工具Dock窗口显示/隐藏"""
        if self.tool_dock.isVisible():
            self.tool_dock.hide()
            self.left_dock.hide()
            self.log_dock.hide()
            self.update_status("工具面板已隐藏")
        else:
            self.tool_dock.show()
            self.left_dock.show()
            self.log_dock.show()
            self.update_status("工具面板已显示")
    

    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self, "退出确认", "确定要退出三角网格标注工具吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
            self.check_config(save=True )
        else:
            event.ignore()

def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")

    # 设置全局样式表
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f0f0f0;
        }
        QDockWidget {
            background-color: white;
            border: 1px solid #ddd;
        }
        QStatusBar {
            background-color: #f5f5f5;
            color: #333;
        }
        QMenuBar {
            background-color: #f5f5f5;
            color: #333;
        }
        QMenu::item:selected {
            background-color: #2196F3;
            color: white;
        }
        QComboBox {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
            min-width: 120px;
        }
        QLineEdit {
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        QListWidget::item:selected {
            background-color: #e6f3ff;
            color: #0066cc;
        }
        QSlider::groove:horizontal {
            background: #ddd;
            height: 8px;
            border-radius: 4px;
        }
        QSlider::handle:horizontal {
            background: #4CAF50;
            width: 18px;
            height: 18px;
            border-radius: 9px;
        }
        QGroupBox {
            border: 1px solid #ddd;
            border-radius: 4px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }
    """)

    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    os.environ["DISPLAY"] = ":0"
    main()