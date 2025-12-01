import json
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QMainWindow, QDockWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QAction, QMenuBar, QMenu, QGridLayout
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



class KeypointAnnotator(QWidget):
    """关键点标注器"""
    def __init__(self, parent=None, label_dock=None):
        super().__init__(parent)
        self.signals = CoreSignals()
        self.keypoints = {}
        self.plt = None
        self.label_dock = label_dock  # 引用标签Dock组件
        self.selected_keypoint = None
        self.sphere_radius = 0.4  # 默认球体大小
        self.next_id = 0  # 用于按顺序分配ID
        self.callback_cid =[] #回调标识
    
    def setup_ui(self):
        """设置UI"""
        self.dock_content = QWidget()
        self.dock_layout = QVBoxLayout(self.dock_content)
        self.dock_layout.setContentsMargins(10, 10, 10, 10)
        self.dock_layout.setSpacing(10)
        

        # 当前选择的标签显示
        self.current_label_widget = QWidget()
        self.current_label_layout = QHBoxLayout(self.current_label_widget)
        self.current_label_layout.setContentsMargins(0, 0, 0, 0)
        self.current_label_layout.setSpacing(5)
        
        self.current_label_icon = QWidget()
        self.current_label_icon.setFixedSize(20, 20)
        self.current_label_name = QLabel("请在左侧选择标签")
        self.current_label_name.setStyleSheet("font-size: 12px; color: #666;")
        
        self.current_label_layout.addWidget(self.current_label_icon)
        self.current_label_layout.addWidget(self.current_label_name)
        self.dock_layout.addWidget(self.current_label_widget)

        

        
        # 控制按钮
        self.btn_complete = QPushButton("完成标注")
        
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
            "• 第一步：在左侧选择要使用的标签",
            "• 第二步：鼠标左键点击模型添加关键点",
            "• 每个标签只能使用一次",
            "• 鼠标右键：选择/取消选择关键点",
            "P 放大标记，shift+P 缩小标记"
        ]
        
        for instruction in instructions:
            lbl = QLabel(instruction)
            lbl.setStyleSheet("font-size: 12px; margin: 2px 0;")
            self.dock_layout.addWidget(lbl)
        
        # 添加拉伸因子
        self.dock_layout.addStretch()
        
        # 连接信号
        self.btn_complete.clicked.connect(self.complete_annotations)
        
        # 连接标签Dock的信号
        if self.label_dock:
            self.label_dock.signals.signal_labels_updated.connect(self.on_labels_updated)
        
        # 更新按钮状态
        self.update_display()
        
        # 释放dock组件信号
        self.signals.signal_dock.emit(self.dock_content)
        
        # 释放信息信号
        self.signals.signal_info.emit("关键点标注已启动 - 请在左侧选择标签后添加关键点")
        
        return self.dock_content
    
    def update_display(self):
        """更新当前标签显示"""
        current_label_name = self.label_dock.label_manager.current_label
        if current_label_name in self.label_dock.label_manager.labels:
            label_info = self.label_dock.label_manager.labels[current_label_name]
            color = label_info['color']
            qcolor = QColor(int(color[0]), int(color[1]), int(color[2]))
            self.current_label_icon.setStyleSheet(f"background-color: {qcolor.name()}; border-radius: 2px;")
            if self.label_dock.label_manager.is_label_used(current_label_name):
                self.current_label_name.setText(f"当前标签: {current_label_name} (已使用)")
                self.current_label_name.setStyleSheet("font-size: 12px; color: #ff0000;")
            else:
                self.current_label_name.setText(f"当前标签: {current_label_name}")
                self.current_label_name.setStyleSheet("font-size: 12px; color: #333;")
        else:
            self.current_label_name.setText(f"请选择标签....")
            self.current_label_name.setStyleSheet("font-size: 12px; color:  #ff0000;")


    
    def on_labels_updated(self, labels):
        """标签配置更新"""
        self.signals.signal_info.emit(f"标签配置更新,{labels}")
        self.update_display()
        if self.plt and self.label_dock:
            new_label_keys  = labels.keys()
            save_keys = self.keypoints.keys()
            for key in save_keys:
                if key not in new_label_keys:
                    # 用户删除标签
                    self.plt.remove(key)
                    self.keypoints.pop(key)
                else:
                    if not labels[key]['used'] :
                        # 用户重置标签
                        self.plt.remove(key)
                        self.keypoints.pop(key)


    

   

    

   

    
       

    def render(self, plt):
        """渲染关键点标注"""
        self.plt = plt

        # 添加键盘和鼠标回调
        callback1= plt.add_callback('on left click', self.on_left_click)
        callback2=  plt.add_callback('on right click', self.on_right_click)
        self.callback_cid+=[callback1,callback2]
        
    
    def on_left_click(self, evt):
        """左键点击添加关键点"""
        if not self.label_dock:
            self.signals.signal_info.emit("错误：标签管理组件未初始化")
            return
        
        current_label_name = self.label_dock.label_manager.current_label
        
        # 检查是否选择了标签
        if not current_label_name or current_label_name not in self.label_dock.label_manager.labels:
            self.signals.signal_info.emit("请先在左侧选择一个标签")
            return
        
        # 检查标签是否已使用
        if self.label_dock.label_manager.is_label_used(current_label_name):
            self.signals.signal_info.emit(f"标签 '{current_label_name}' 已使用，请选择其他标签")
            return

        
        if hasattr(evt, 'actor') and evt.actor:
            # 获取点击位置
            if hasattr(evt, 'picked3d') and evt.picked3d is not None:
                pts = evt.picked3d
                color = self.label_dock.label_manager.get_label_color(current_label_name)
                
                # 创建关键点球体
                keypoint = vedo.Sphere(pos=pts, r=self.sphere_radius, c=color, alpha=0.8).pickable(True)
                keypoint.name = current_label_name
                self.plt.add(keypoint)
                new_kp = {
                    'keypoints': pts,
                    'color': color,
                }
                
                self.keypoints[current_label_name]=new_kp
                self.label_dock.use_label(current_label_name)
                self.signals.signal_info.emit(f"添加关键点标签: {current_label_name}")
    
    def on_right_click(self, evt):
        """右键删除关键点"""
        if hasattr(evt, 'actor') and evt.actor:
            name = evt.actor.name
            if name in self.keypoints:
                self.plt.remove(name)
                self.keypoints.pop(name)
                self.label_dock.unuse_label(name)
                self.signals.signal_info.emit(f"删除标签: {name}")
                self.label_dock.label_manager.current_label=name
                self.update_display()

            



    
    def complete_annotations(self):
        """完成标注"""
        # 检查未使用的标签
        unused = self.label_dock.label_manager.get_unused_labels()
        if unused:
            msg = f"以下标签尚未使用：\n"
            msg += "\n".join([f"• {label}" for label in unused])
            msg += "\n\n是否继续保存？"
            
            reply = QMessageBox.question(self, "未使用标签提醒", msg,
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            return reply == QMessageBox.Yes
        
        if not self.keypoints:
            self.signals.signal_info.emit("没有关键点可完成标注")
            return

        self.signals.signal_info.emit(f"标注完成 - 共添加 {len(self.keypoints)} 个关键点)")
        self.signals.signal_close.emit(self.keypoints)

    def clean(self):
        self.keypoints = []
        self.selected_keypoint = None
        self.sphere_radius = 0.4  # 默认球体大小
        self.next_id = 0  # 用于按顺序分配ID

        # 标签
        self.label_dock.clean()
        # 回调
        for i in self.callback_cid:
            self.plt.remove_callback(i)
        self.callback_cid.clear()
        self.plt = None
    
        
        
        
        