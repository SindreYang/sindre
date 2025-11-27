import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QMessageBox, QSplitter,
                            QStatusBar, QToolBar, QListWidget, QDialog, QLabel,
                            QDockWidget, QAction, QMenuBar, QMenu, QComboBox,
                            QColorDialog, QLineEdit, QGridLayout, QListWidgetItem,
                            QSlider, QCheckBox, QFormLayout, QGroupBox, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSize, QTimer
from PyQt5.QtGui import QKeySequence, QColor, QIcon, QBrush, QPen, QPixmap
import numpy as np
import vedo
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera

from sindre.utils3d.Label3d.components.KeypointAnnotator import KeypointAnnotator
from sindre.utils3d.Label3d.components.LabelDock import LabelDockWidget
from sindre.utils3d.Label3d.components.SegAnnotator import SplineSegmentAnnotator
from sindre.utils3d.Label3d.components.CutAnnotator import CutAnnotator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三角网格标注工具")
        self.setGeometry(100, 100, 1600, 900)
        
        # 初始化变量
        self.current_annotator = None
        self.mesh = None
        self.label_dock = None
        
        # 初始化UI
        self.init_ui()
        self.init_signals()
    
    def init_ui(self):
        """初始化UI"""
        # 创建独立的标签管理Dock（固定在左侧）
        self.label_dock = LabelDockWidget(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.label_dock)
        
        # 主菜单
        self.create_menus()
        
        # 中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 3D视图
        self.init_3d_view()
        
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
        
        
        # 连接标签Dock的信号
        self.label_dock.signals.signal_info.connect(self.update_status)
    
    def create_menus(self):
        """创建菜单"""
        self.main_menu_bar = QMenuBar()
        
        # 文件菜单
        self.file_menu = QMenu("文件", self)
        self.open_action = QAction("打开模型", self)
        self.save_action = QAction("保存标注", self)
        self.exit_action = QAction("退出", self)
        
        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.save_action)
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
        
        self.annotate_menu.addAction(self.keypoints_action)
        self.annotate_menu.addAction(self.segment_action)
        self.annotate_menu.addAction(self.bbox_action)
        self.annotate_menu.addAction(self.transform_action)
        self.annotate_menu.addAction(self.sculpt_action)
        self.annotate_menu.addAction(self.cut_action)
        
        # 视图菜单
        self.view_menu = QMenu("视图", self)
        self.reset_view_action = QAction("重置视图", self)
        self.toggle_tool_dock_action = QAction("显示/隐藏工具面板", self)
        
        self.view_menu.addAction(self.reset_view_action)
        self.view_menu.addAction(self.toggle_tool_dock_action)
        
        # 帮助菜单
        self.help_menu = QMenu("帮助", self)
        self.about_action = QAction("关于", self)
        
        self.help_menu.addAction(self.about_action)
        
        # 添加到菜单栏
        self.main_menu_bar.addMenu(self.file_menu)
        self.main_menu_bar.addMenu(self.annotate_menu)
        self.main_menu_bar.addMenu(self.view_menu)
        self.main_menu_bar.addMenu(self.help_menu)
        
        self.setMenuBar(self.main_menu_bar)
    
    def init_signals(self):
        """初始化信号连接"""
        # 文件菜单
        self.open_action.triggered.connect(self.open_model)
        self.save_action.triggered.connect(self.save_annotations)
        self.exit_action.triggered.connect(self.close)
        
        # 标注菜单
        self.keypoints_action.triggered.connect(self.start_keypoint_annotation)
        self.segment_action.triggered.connect(self.start_segment_annotation)
        self.bbox_action.triggered.connect(self.start_bbox_annotation)
        self.transform_action.triggered.connect(self.start_transform_annotation)
        self.cut_action.triggered.connect(self.start_cut_annotation)
        
        # 视图菜单
        self.reset_view_action.triggered.connect(self.reset_3d_view)
        self.toggle_tool_dock_action.triggered.connect(self.toggle_tool_dock)
        
        # 帮助菜单
        self.about_action.triggered.connect(self.show_about)
    
    def init_3d_view(self):
        """初始化3D视图"""
        # 创建VTK部件
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtk_widget)
        
        # 创建vedo绘图器
        self.vp = vedo.Plotter(N=1, qt_widget=self.vtk_widget)
        self.vp.show(bg="white", title="3D模型视图")
        
        # 设置交互样式
        self.vp.interactor.SetInteractorStyle(vtkInteractorStyleTrackballCamera())
    

    
    def open_model(self):
        """打开3D模型文件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "打开3D模型文件", "",
            "3D模型文件 (*.ply *.obj *.stl *.vtk *.vtp);;所有文件 (*.*)"
        )
        #file_path=r"C:\Users\yx\Downloads\J10113717111_17\扫描模型\J10113717111_UpperJaw.stl"
        if file_path:
            try:
                # 清除当前视图
                self.vp.clear()
                
                # 读取模型
                self.mesh = vedo.Mesh(file_path)
                
                # 显示模型
                self.vp.add(self.mesh)
                self.vp.reset_camera()
                self.vp.render()
                
                # 更新状态
                self.status_bar.showMessage(f"已加载模型: {os.path.basename(file_path)},顶点数量: {self.mesh.npoints},面数量: {self.mesh.ncells}")
                
       
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")
                self.status_bar.showMessage("加载模型失败")
    

    
    def get_button_style(self, color):
        """获取按钮样式"""
        return f"""
            QPushButton {{
                padding: 12px 20px;
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                text-align: left;
                padding-left: 25px;
            }}
            QPushButton:hover {{
                opacity: 0.9;
            }}
            QPushButton:pressed {{
                opacity: 0.8;
            }}
        """
    
    def start_keypoint_annotation(self):
        """开始关键点标注"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先打开一个3D模型")
            return
        
        # 清除当前标注器
        self.cleanup_current_annotator()
        
        # 创建关键点标注器，传入标签Dock组件
        self.current_annotator = KeypointAnnotator(label_dock=self.label_dock)
        
        # 连接信号
        self.current_annotator.signals.signal_info.connect(self.update_status)
        self.current_annotator.signals.signal_dock.connect(self.update_tool_dock)
        self.current_annotator.signals.signal_close.connect(self.complete_annotation)
        
        # 设置UI
        self.current_annotator.setup_ui()
        
        # 渲染标注器
        self.current_annotator.render(self.vp)
    
    def start_segment_annotation(self):
        """开始分割标注"""
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先打开一个3D模型")
            return
        
        # 清除当前标注器
        self.cleanup_current_annotator()
        
        # 创建关键点标注器，传入标签Dock组件
        self.current_annotator = SplineSegmentAnnotator(label_dock=self.label_dock)
        
        # 连接信号
        self.current_annotator.signals.signal_info.connect(self.update_status)
        self.current_annotator.signals.signal_dock.connect(self.update_tool_dock)
        self.current_annotator.signals.signal_close.connect(self.complete_annotation)
        
        # 设置UI
        self.current_annotator.setup_ui()
        
        # 渲染标注器
        self.current_annotator.render(self.vp)
        
    
    def start_bbox_annotation(self):
        """开始边界框标注"""
        QMessageBox.information(self, "提示", "边界框标注功能开发中...")
        
    def start_cut_annotation(self):
        if not self.mesh:
            QMessageBox.warning(self, "警告", "请先打开一个3D模型")
            return
        # 清除当前标注器
        self.cleanup_current_annotator()
        
        # 创建标注器
        self.current_annotator = CutAnnotator()
        
        # 连接信号
        self.current_annotator.signals.signal_info.connect(self.update_status)
        self.current_annotator.signals.signal_dock.connect(self.update_tool_dock)
        self.current_annotator.signals.signal_close.connect(self.complete_annotation)
        
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
        
    
    def update_status(self, text):
        """更新状态栏"""
        self.status_bar.showMessage(text)
    
    def update_tool_dock(self, widget):
        """更新工具Dock窗口内容"""
        self.tool_dock.setWidget(widget)
    
    def complete_annotation(self, completed):
        """完成标注"""
        if completed:
            self.cleanup_current_annotator()
    
    def cleanup_current_annotator(self):
        """清理当前标注器"""
        if self.current_annotator:
            self.current_annotator.close()
            self.current_annotator = None
            
    
    def reset_3d_view(self):
        """重置3D视图"""
        if self.vp:
            self.vp.reset_camera()
            self.vp.render()
            self.status_bar.showMessage("视图已重置")
    
    def toggle_tool_dock(self):
        """切换工具Dock窗口显示/隐藏"""
        if self.tool_dock.isVisible():
            self.tool_dock.hide()
            self.status_bar.showMessage("工具面板已隐藏")
        else:
            self.tool_dock.show()
            self.status_bar.showMessage("工具面板已显示")
    
    def save_annotations(self):
        """保存标注"""
        if self.current_annotator and hasattr(self.current_annotator, 'save_keypoints'):
            self.current_annotator.save_keypoints()
        else:
            QMessageBox.warning(self, "警告", "没有正在进行的标注")
    
    def show_about(self):
        """显示关于信息"""
        about_text = """
        三角网格标注工具 v1.3
        
        新增功能：
        • 独立的标签管理面板（左侧固定）
        • 每个标签只能标记一次
        • 标签使用完成后显示提示
        • 优化的用户界面布局
        
        原有功能：
        • 关键点标注（支持多标签和颜色）
        • 标签配置导入导出
        • 关键点编辑和删除
        • 球体大小实时调节
        • 多种3D模型格式支持
        
        支持格式：
        • PLY (.ply)
        • OBJ (.obj)
        • STL (.stl)
        • VTK (.vtk)
        • VTP (.vtp)
        
        使用说明：
        1. 在左侧标签管理面板选择要使用的标签
        2. 点击模型表面添加关键点
        3. 每个标签只能使用一次
        4. 可以在右侧面板编辑和管理关键点
        
        版权所有 © 2024
        """
        
        QMessageBox.about(self, "关于", about_text)
    
    def closeEvent(self, event):
        """关闭事件"""
        reply = QMessageBox.question(
            self, "退出确认", "确定要退出三角网格标注工具吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            event.accept()
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
    main()