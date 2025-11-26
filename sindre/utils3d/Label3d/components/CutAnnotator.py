import json
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QDockWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QAction, QMenuBar, QMenu, QGridLayout,QButtonGroup
)
import vedo
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QFileDialog, QMessageBox, QSplitter,
                            QStatusBar, QToolBar, QListWidget, QDialog, QLabel,
                            QDockWidget, QAction, QMenuBar, QMenu, QComboBox,
                            QColorDialog, QLineEdit, QGridLayout, QListWidgetItem,
                            QSlider, QCheckBox, QFormLayout, QGroupBox, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSize, QTimer
from PyQt5.QtGui import QKeySequence, QColor, QIcon, QBrush, QPen, QPixmap
import vedo
from sindre.utils3d.Label3d.core.manager import CoreSignals




class MyPlaneCutter(vedo.PlaneCutter):
    # 重写快捷键
    def _keypress(self, vobj, event):
        if vobj.GetKeySym() == "r":  # reset planes
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "u":  # invert cut
            self.invert()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "x":  # set normal along x
            self.widget.SetNormal((1, 0, 0))
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "y":  # set normal along y
            self.widget.SetNormal((0, 1, 0))
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "z":  # set normal along z
            self.widget.SetNormal((0, 0, 1))
            self.widget.GetPlane(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
  


class MyBoxCutter(vedo.BoxCutter):
    # 重写快捷键
    def _keypress(self, vobj, event):
        if vobj.GetKeySym() == "r":  # reset planes
            self._implicit_func.SetBounds(self._init_bounds)
            self.widget.GetPlanes(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "u":
            self.invert()
            self.widget.GetInteractor().Render()



class MySphereCutter(vedo.SphereCutter):
    def _keypress(self, vobj, event):
        if vobj.GetKeySym() == "r":  # reset planes
            self._implicit_func.SetBounds(self._init_bounds)
            self.widget.GetPlanes(self._implicit_func)
            self.widget.PlaceWidget()
            self.widget.GetInteractor().Render()
        elif vobj.GetKeySym() == "u":
            self.invert()
            self.widget.GetInteractor().Render()



class CutAnnotator(QWidget):
    """位置调整标注器"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = CoreSignals()
        self.keypoints = []
        self.plt = None
        self.current_cutter =None
        self.current_mesh =None

       
    
    def setup_ui(self):
        """设置UI"""
        self.dock_content = QWidget()
        self.dock_layout = QVBoxLayout(self.dock_content)
        self.dock_layout.setContentsMargins(10, 10, 10, 10)
        self.dock_layout.setSpacing(10)
        
        
        # 初始化裁剪选择器
        self.mesh_switch_widget = QWidget()
        self.mesh_switch_layout = QHBoxLayout(self.mesh_switch_widget)
        self.mesh_switch_layout.setContentsMargins(0, 5, 0, 5)
        self.mesh_switch_layout.setSpacing(8)  # 按钮间距
        self.mesh_button_group = QButtonGroup(self)
        self.mesh_button_group.setExclusive(True) #设置为互斥模式
        self.planecutter_btn = QPushButton(f"平面裁剪")
        self.boxcutter_btn = QPushButton(f"矩形裁剪")
        self.spherecutter_btn = QPushButton(f"球形裁剪")
        self.mesh_button_group.addButton(self.planecutter_btn)
        self.mesh_button_group.addButton(self.boxcutter_btn)
        self.mesh_button_group.addButton(self.spherecutter_btn)
        self.mesh_switch_layout.addWidget(self.planecutter_btn)
        self.mesh_switch_layout.addWidget(self.boxcutter_btn)
        self.mesh_switch_layout.addWidget(self.spherecutter_btn)
        
        
        self.dock_layout.addWidget(self.mesh_switch_widget)
        
        
        # 控制按钮
        self.btn_complete = QPushButton("保存")
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                padding: 8px 16px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """

        self.btn_complete.setStyleSheet(button_style)
        

        self.dock_layout.addWidget(self.btn_complete)
        
        # 添加操作说明
        info_label = QLabel("操作说明：")
        info_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        self.dock_layout.addWidget(info_label)
        
        instructions = [
            "r  --重置 ",
            "u  --反选 ",
            "x,y,z --选择裁剪平面"
            "ctrl+s -- 保存"
        ]
        
        for instruction in instructions:
            lbl = QLabel(instruction)
            lbl.setStyleSheet("font-size: 12px; margin: 2px 0;")
            self.dock_layout.addWidget(lbl)
        
        # 添加拉伸因子
        self.dock_layout.addStretch()
        
        # 连接信号
        self.btn_complete.clicked.connect(self.complete_annotations)
        self.planecutter_btn.clicked.connect(self.start_planecutter)
        self.boxcutter_btn.clicked.connect(self.start_boxcutter)
        self.spherecutter_btn.clicked.connect(self.start_spherecutter)
    
        
        # 释放dock组件信号
        self.signals.signal_dock.emit(self.dock_content)
        
        # 释放信息信号
        self.signals.signal_info.emit("位置调整标注器已启动")
        
        return self.dock_content
    
    
    def start_planecutter(self):
        if self.current_cutter is not  None:
            self.current_cutter.off()
        self.current_cutter=MyPlaneCutter(self.current_mesh)
        self.current_cutter.name ="cutter"
        self.plt.add(self.current_cutter)
        self.current_cutter.on()
        self.plt.render()
        
    def start_boxcutter(self):
        if self.current_cutter is not  None:
            self.current_cutter.off()
        self.current_cutter=MyBoxCutter(self.current_mesh)
        self.current_cutter.name ="cutter"
        self.plt.add(self.current_cutter)
        self.current_cutter.on()
        self.plt.render()

    def start_spherecutter(self):
        if self.current_cutter is not  None:
            self.current_cutter.off()
        self.current_cutter=MySphereCutter(self.current_mesh)
        self.current_cutter.name ="cutter"
        self.plt.add(self.current_cutter)
        self.current_cutter.on()
        self.plt.render()
        
        
    def callback_save_mesh(self,evt):
        if hasattr(evt, 'keypress'):
            key = evt.keypress.lower()
            if key == "space": # 空格键
                # 保存当前mesh
                if self.current_mesh:
                    pass
                    
                    
                
    

        
    
    def render(self, plt):
        """渲染关键点标注"""
        self.plt = plt
        if self.current_mesh is None:
            self.current_mesh=self.plt.get_meshes()[0].clone()
            self.plt.clear()

    
    def complete_annotations(self):
        """完成标注"""
        # 检查未使用的标签
        file_path, _ = QFileDialog.getSaveFileName(
                self, "保存mesh", "cut_mesh.ply", "所有文件 (*.*)"
            )
        if file_path:
            self.current_mesh.write(file_path)
            self.signals.signal_info.emit(f"标注完成")
            self.signals.signal_close.emit(True)
    
    
        
        
        
        