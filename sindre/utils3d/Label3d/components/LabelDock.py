

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
from PyQt5.QtWidgets import QApplication
from sindre.utils3d.Label3d.core.manager import CoreSignals, LabelManager
class ColorHolder:
    # 共享变量
    def __init__(self, r, g, b):
        self.r = r
        self.g = g
        self.b = b
    def get_color(self):
        return [self.r,self.g,self.b]
        
class LabelDockWidget(QDockWidget):
    """独立的标签管理Dock组件"""
    def __init__(self, parent=None):
        super().__init__("标签管理", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.setFeatures(QDockWidget.NoDockWidgetFeatures)  # 固定，不能关闭或移动
        self.setFixedWidth(220)
        
        self.label_manager = LabelManager()
        self.signals = CoreSignals()
        
        self.setup_ui()
        self.update_label_buttons()
    
    def setup_ui(self):
        """设置UI"""
        self.main_widget = QWidget()
        self.setWidget(self.main_widget)
        
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(8)
        
        # # 标题
        # title_label = QLabel("标签管理")
        # title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px; text-align: center;")
        # self.main_layout.addWidget(title_label)
        
        # 标签按钮容器
        self.labels_container = QWidget()
        self.labels_layout = QVBoxLayout(self.labels_container)
        self.labels_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_layout.setSpacing(6)
        self.main_layout.addWidget(self.labels_container)
        
        # 操作按钮
        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout(self.buttons_widget)
        self.buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.buttons_layout.setSpacing(5)
        
        
        self.append_btn = QPushButton("添加")
        self.append_btn.setFixedSize(30, 30)
        self.reset_btn = QPushButton("重置")
        self.reset_btn.setFixedSize(30, 30)
        self.import_btn = QPushButton("导入")
        self.import_btn.setFixedSize(30, 30)
        self.export_btn = QPushButton("导出")
        self.export_btn.setFixedSize(30, 30)
        
        self.buttons_layout.addWidget(self.append_btn)
        self.buttons_layout.addWidget(self.reset_btn)
        self.buttons_layout.addWidget(self.import_btn)
        self.buttons_layout.addWidget(self.export_btn)
        
        self.main_layout.addWidget(self.buttons_widget)
        

        # 连接信号
        self.append_btn.clicked.connect(self.on_add_label)
        self.reset_btn.clicked.connect(self.on_reset_all_labels)
        self.import_btn.clicked.connect(self.on_import_labels)
        self.export_btn.clicked.connect(self.on_export_labels)
        
        # 添加拉伸
        self.main_layout.addStretch()
        
        
    def use_label(self, label_name):
        """使用标签（标记为已使用）"""
        if self.label_manager.mark_label_used(label_name):
            self.update_label_buttons()
            self.signals.signal_labels_updated.emit(self.label_manager.labels)
            self.show_label_completion_message(label_name)
            # 标记使用,自动将current_label设置为下一个
            unused_labels = self.label_manager.get_unused_labels()
            if len(unused_labels)>0:
                self.label_manager.current_label=unused_labels[0]
                self.signals.signal_info.emit(f"已选择标签: {unused_labels[0]}")
            return True
        return False
    
    def unuse_label(self, label_name):
        """取消使用标签"""
        if self.label_manager.mark_label_unused(label_name):
            self.update_label_buttons()
            self.signals.signal_labels_updated.emit(self.label_manager.labels)
            return True
        return False
    
    def update_label_buttons(self):
        """更新标签按钮"""
        # 清除现有按钮
        for i in reversed(range(self.labels_layout.count())):
            widget = self.labels_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        
        # 创建新的标签按钮
        for label_name in self.label_manager.get_label_names():
            label_info = self.label_manager.labels[label_name]
            label_widget = self.create_label_widget(label_name, label_info)
            self.labels_layout.addWidget(label_widget)
    
    def create_label_widget(self, label_name, label_info):
        """创建标签控件"""
        widget = QWidget()
        widget.setFixedHeight(40)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 颜色块
        color_widget = QWidget()
        color = label_info['color']
        qcolor = QColor(int(color[0]), int(color[1]), int(color[2]))
        color_widget.setStyleSheet(f"background-color: {qcolor.name()}; border-radius: 4px;")
        color_widget.setFixedSize(30, 30)
        
        # 标签名称
        name_label = QLabel(label_name)
        name_label.setStyleSheet("font-size: 12px;")
        
        # 状态指示器
        status_widget = QWidget()
        status_widget.setFixedSize(10, 10)
        if label_info['used']:
            status_widget.setStyleSheet("background-color: #4CAF50; border-radius: 5px;")  # 绿色表示已使用
        else:
            status_widget.setStyleSheet("background-color: #cccccc; border-radius: 5px;")  # 灰色表示未使用
        
        layout.addWidget(color_widget)
        layout.addWidget(name_label, 1)
        layout.addWidget(status_widget)
        
        # 设置点击事件
        widget.mousePressEvent = lambda event, name=label_name: self.on_label_clicked(name, event)
        
        # 设置样式
        if label_info['used']:
            widget.setStyleSheet("""
                QWidget { background-color: #f0f0f0; border-radius: 4px; }
                QWidget:hover { background-color: #e0e0e0; }
            """)
        else:
            widget.setStyleSheet("""
                QWidget { background-color: white; border: 1px solid #dddddd; border-radius: 4px; }
                QWidget:hover { background-color: #f8f8f8; border-color: #cccccc; }
            """)
        
        return widget
    
    def on_label_clicked(self, label_name, event):
        """标签点击事件"""
        self.signals.signal_labels_clicked.emit(label_name)
        if event.button() == Qt.LeftButton:
            if not self.label_manager.is_label_used(label_name):
                self.label_manager.current_label = label_name
                self.signals.signal_info.emit(f"已选择标签: {label_name}")
                self.update_label_buttons()
            else:
                self.signals.signal_info.emit(f"标签 '{label_name}' 已使用")
        
        elif event.button() == Qt.RightButton:
            self.show_label_context_menu(label_name, event.globalPos())
    
    def show_label_context_menu(self, label_name, pos):
        """显示标签上下文菜单"""
        menu = QMenu()
        
        
        # 重置标签
        reset_action = menu.addAction("重置标签")
        reset_action.triggered.connect(lambda: self.on_reset_label(label_name))
        
        
        # 编辑标签
        edit_action = menu.addAction("编辑标签")
        edit_action.triggered.connect(lambda: self.on_edit_label(label_name))
        
       
        
        # 删除标签（默认标签除外）
        delete_action = menu.addAction("删除标签")
        delete_action.triggered.connect(lambda: self.on_delete_label(label_name))
        
        # 添加新标签
        add_action = menu.addAction("添加标签")
        add_action.triggered.connect(self.on_add_label)
        
        menu.exec_(pos)
        
        

    
    def on_edit_label(self, old_label_name):
        """编辑标签"""
        label_info = self.label_manager.labels[old_label_name]
        
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑标签")
        dialog.setFixedSize(300, 150)
        
        layout = QVBoxLayout(dialog)
        
        name_layout = QHBoxLayout()
        name_label = QLabel("标签名称:")
        name_input = QLineEdit(old_label_name)
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        
        color_layout = QHBoxLayout()
        color_label = QLabel("标签颜色:")
        color_preview = QPushButton()
        color = label_info['color']
        qcolor = QColor(int(color[0]), int(color[1]), int(color[2]))
        color_preview.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #ccc; border-radius: 2px;")
        color_preview.setFixedSize(30, 30)
        
        temp_color_holder = ColorHolder(255, 0, 0)
        color_preview.clicked.connect(lambda: self.choose_color_dialog(dialog, color_preview, temp_color_holder))
        color_layout.addWidget(color_label)
        color_layout.addWidget(color_preview)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        
        def confirm_edit():
            new_name = name_input.text().strip()
            if not new_name:
                QMessageBox.warning(dialog, "警告", "标签名称不能为空！")
                return
            temp_color=temp_color_holder.get_color()
            # 使用新的 update_label 方法
            if self.label_manager.update_label(old_label_name, new_name=new_name, new_color=temp_color):
                self.update_label_buttons()
                self.signals.signal_labels_updated.emit(self.label_manager.labels)
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "警告", f"更新标签失败！标签名 '{new_name}' 可能已存在。")
        
        ok_btn.clicked.connect(confirm_edit)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(name_layout)
        layout.addLayout(color_layout)
        layout.addLayout(btn_layout)
        
        dialog.exec_()
    
    def choose_color_dialog(self, parent_dialog, color_preview, color_holder):
        """选择颜色对话框"""
        color = QColorDialog.getColor(QColor(color_holder.r, color_holder.g,color_holder.b), 
                                    parent_dialog, "选择标签颜色")
        if color.isValid():
            color_holder.r = color.red()
            color_holder.g = color.green()
            color_holder.b = color.blue()
            color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #ccc; border-radius: 2px;")
    
    def on_add_label(self):
        """添加新标签"""
        dialog = QDialog(self)
        dialog.setWindowTitle("添加新标签")
        dialog.setFixedSize(300, 150)
        
        layout = QVBoxLayout(dialog)
        
        # 标签名称
        name_layout = QHBoxLayout()
        name_label = QLabel("标签名称:")
        name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_input)
        
        # 标签颜色
        color_layout = QHBoxLayout()
        color_label = QLabel("标签颜色:")
        color_preview = QPushButton()
        color_preview.setStyleSheet("background-color: #ff0000; border: 1px solid #ccc; border-radius: 2px;")
        color_preview.setFixedSize(30, 30)
        
        temp_color_holder = ColorHolder(255, 0, 0)
        color_preview.clicked.connect(lambda: self.choose_color_dialog(dialog, color_preview, temp_color_holder))
        color_layout.addWidget(color_label)
        color_layout.addWidget(color_preview)
        
        # 按钮
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        
        def confirm_add():
            label_name = name_input.text().strip()
            if not label_name:
                QMessageBox.warning(dialog, "警告", "标签名称不能为空！")
                return
            
            final_color =temp_color_holder.get_color()
            # 使用新的 add_label 方法，标签名即为键
            if self.label_manager.add_label(label_name, final_color):
                self.update_label_buttons()
                self.signals.signal_labels_updated.emit(self.label_manager.labels)
                self.signals.signal_info.emit(f"添加新标签: {label_name}")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "警告", f"添加标签失败！标签名 '{label_name}' 已存在。")
    
        ok_btn.clicked.connect(confirm_add)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(name_layout)
        layout.addLayout(color_layout)
        layout.addLayout(btn_layout)
        
        dialog.exec_()
    
    def on_delete_label(self, label_name):
        """删除标签"""
        reply = QMessageBox.question(self, "确认删除", f"确定要删除标签 '{label_name}' 吗？",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.label_manager.remove_label(label_name):
                self.update_label_buttons()
                self.signals.signal_labels_updated.emit(self.label_manager.labels)
                self.signals.signal_info.emit(f"删除标签: {label_name}")
                
                
    def on_reset_label(self,label_name):
        reply = QMessageBox.question(self, "确认删除", f"确定要重置'{label_name}' 吗？",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            if self.label_manager.is_label_used(label_name):
                self.label_manager.mark_label_unused(label_name)
                self.update_label_buttons()
                self.signals.signal_labels_updated.emit(self.label_manager.labels)
                self.signals.signal_info.emit(f"删除标签: {label_name}")
        
    
    def on_reset_all_labels(self):
        """重置所有标签"""
        reply = QMessageBox.question(self, "确认重置", "确定要重置所有标签为未使用状态吗？",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.label_manager.reset_all_labels()
            self.update_label_buttons()
            self.signals.signal_labels_updated.emit(self.label_manager.labels)
            self.signals.signal_info.emit("已重置所有标签为未使用状态")
    
    def on_import_labels(self):
        """导入标签配置 (从 .npy 文件)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "导入标签配置", "", "NumPy 文件 (*.npy);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                # 从 .npy 文件加载字典
                loaded_labels = np.load(file_path, allow_pickle=True).item()
                
                # 验证加载的是否为字典
                if not isinstance(loaded_labels, dict):
                    raise ValueError("加载的文件内容不是一个有效的标签字典。")

                # 更新标签管理器的状态
                self.label_manager.labels = loaded_labels
                # 如果加载的字典不为空，设置 current_label 为第一个标签
                if loaded_labels:
                    self.label_manager.current_label = next(iter(loaded_labels.keys()))
                else:
                    self.label_manager.current_label = None

                # 更新 UI 和发送信号
                self.update_label_buttons()
                self.signals.signal_labels_updated.emit(self.label_manager.labels)
                self.signals.signal_info.emit(f"成功导入标签配置: {os.path.basename(file_path)}")

            except Exception as e:
                # 捕获所有可能的错误（文件损坏、格式错误、路径错误等）
                QMessageBox.warning(self, "警告", f"导入标签配置失败！\n错误信息: {str(e)}")
    
    def on_export_labels(self):
        """导出标签配置 (到 .npy 文件)"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出标签配置", "labels_config.npy", "NumPy 文件 (*.npy);;所有文件 (*.*)"
        )
        
        if file_path:
            # 确保文件后缀是 .npy
            if not file_path.endswith('.npy'):
                file_path += '.npy'

            try:
                # 直接保存 labels 字典到 .npy 文件
                np.save(file_path, self.label_manager.labels)
                self.signals.signal_info.emit(f"成功导出标签配置: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"导出标签配置失败！\n错误信息: {str(e)}")
    
    
    
    def show_label_completion_message(self, label_name):
        """显示标签完成提示"""
        # 创建一个临时的提示窗口
        msg_widget = QWidget()
        msg_widget.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        msg_widget.setStyleSheet("""
            QWidget {
                background-color: #4CAF50;
                color: white;
                border-radius: 8px;
                padding: 10px 20px;
            }
        """)
        
        layout = QHBoxLayout(msg_widget)
        icon_label = QLabel("✓")
        icon_label.setStyleSheet("font-size: 20px; margin-right: 10px;")
        text_label = QLabel(f"标签 '{label_name}' 标记完成！")
        text_label.setStyleSheet("font-size: 14px;")
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        
        # 显示在屏幕中央
        screen_geometry = QApplication.desktop().availableGeometry()
        msg_widget.adjustSize()
        x = (screen_geometry.width() - msg_widget.width()) // 2
        y = (screen_geometry.height() - msg_widget.height()) // 2
        msg_widget.move(x, y)
        
        msg_widget.show()
        
        # 2秒后自动隐藏
        QTimer.singleShot(2000, msg_widget.deleteLater)