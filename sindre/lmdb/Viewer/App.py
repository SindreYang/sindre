# -*- coding: UTF-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@path   ：ToothSegData -> 3D_Lmdb_Viewer.py
@IDE    ：PyCharm
@Author ：sindre
@Email  ：yx@mviai.com
@Date   ：2023/9/1 13:30
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2023/9/1 :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(一)本代码的质量保证期（简称“质保期”）为上线内 1个月，质保期内乙方对所代码实行包修改服务。
(二)本代码提供三包服务（包阅读、包编译、包运行）不包熟
(三)本代码所有解释权归权归神兽所有，禁止未开光盲目上线
(四)请严格按照保养手册对代码进行保养，本代码特点：
      i. 运行在风电、水电的机器上
     ii. 机器机头朝东，比较喜欢太阳的照射
    iii. 集成此代码的人员，应拒绝黄赌毒，容易诱发本代码性能越来越弱
声明：未履行将视为自主放弃质保期，本人不承担对此产生的一切法律后果
如有问题，热线: 114

"""
__author__ = 'sindre'

import json
import traceback

import numpy as np
# You may need to uncomment these lines on some systems:
import vtk.qt
from PyQt5.QtCore import QStringListModel, QItemSelectionModel
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import (QFileDialog, QInputDialog, QMessageBox, QLineEdit, 
                            QDialog, QVBoxLayout, QComboBox, QLabel, QPushButton,
                            QDialogButtonBox, QGroupBox)
from sindre.lmdb import get_data_value

vtk.qt.QVTKRWIBase = "QGLWidget"
import vtk
import os
import vedo
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from PyQt5.QtWidgets import QApplication, QWidget, QTreeWidgetItem
from PyQt5 import QtWidgets, QtCore
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Points
from UI.View_UI import Ui_Form,DataConfigDialog
import qdarkstyle
from sindre.lmdb import Reader,Writer
from sindre.utils3d import labels2colors, SindreMesh
from sindre.utils3d.algorithm import face_labels_to_vertex_labels
from sindre.general.logs import CustomLogger

class config_thread(QtCore.QThread):
    progress_int = QtCore.pyqtSignal(int)

    def __init__(self, db_path, data_config, name_key, start_idx, end_idx):
        super().__init__()
        self.data_config =data_config
        self.db_path = db_path
        self.name_key = name_key
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.len_idx = end_idx - start_idx

    def run(self):
        self.progress_int.emit(0)
        for idx in range(self.start_idx, self.end_idx):
            with Reader(self.db_path) as db:
                data = db[idx]
            idx_str = str(idx)

            # 提取目标值（统一异常处理，简化默认值）
            try:
                value=get_data_value(data,self.name_key)
            except Exception:
                value = f"{idx_str}_unknown"


            # 格式化值
            if isinstance(value, np.ndarray):
                if np.issubdtype(value.dtype, np.str_):
                    # 字符串数组：转字符串后取后30位
                    formatted = str(value)[-30:]
                else:
                    # 非字符串数组：单元素取原值后30位，多元素显示形状
                    formatted = str(value.item())[-30:] if value.shape == () else f"shape:{value.shape}"
            else:
                # 非数组类型：字符串取后30位，其他直接转字符串
                formatted = str(value)[-30:] if isinstance(value, str) else str(value)


            # 使用configparser存储映射
            formatted=f"{idx_str}_{formatted}" # 保证数据唯一性
            self.data_config[idx_str]= formatted
            self.data_config[formatted]=idx_str

            # 5. 发送进度信号
            progress = int((idx - self.start_idx) / (self.end_idx - self.start_idx) * 99)
            self.progress_int.emit(progress)

        self.progress_int.emit(100)

class LMDB_Viewer(QtWidgets.QWidget):
    # 添加信号
    countChanged = QtCore.pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)

        self.app_ui = Ui_Form()
        self.app_ui.setupUi(self)
        self.log =CustomLogger("LMDB_Viewer").get_logger()
    
        # 基础变量
        self.vp = None
        self.count = 0
        self.max_count = 0
        self.current_mesh = None
        self.db_path = None
        self.fileName = None
        self.page_size = 15
        self.current_page = 1
        self.vertex_labels =None
        self.db_mb_size=0


        # 用户配置
        self.data_config = {
            "data_type":None,
            "vertex_key": None,
            "vertex_label_key": None,
            "face_key": None,
            "face_label_key": None,
            "name_key": None,
            "image_key":None,
            "bbox_key":None,
            "keypoints_key": None,
            "segmentation_key":None,
        }
        self.init_ui()

    def init_ui(self):

        # 信息视图
        self.app_ui.treeWidget.setHeaderLabels(["键名", "类型", "大小"])
        self.app_ui.treeWidget.setColumnCount(3)
        self.app_ui.treeWidget.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        # 按钮绑定 
        self.app_ui.openmdbBt.clicked.connect(self.OpenFile)
        self.app_ui.NextButton.clicked.connect(self.NextFile)
        self.app_ui.PreButton.clicked.connect(self.PreFile)
        self.app_ui.JumpButton.clicked.connect(self.JumpCount)
        self.app_ui.SearchButton.clicked.connect(self.search)
        self.app_ui.NameView.clicked.connect(self.show_selected_value)
        self.app_ui.state_bt.clicked.connect(self.SetState)
        self.app_ui.Pre_view_Button.clicked.connect(self.Pre_Page)
        self.app_ui.Next_view_Button.clicked.connect(self.Next_Page)
        self.app_ui.Pre_num_view_Button.clicked.connect(self.Pre_Num_Page)
        self.app_ui.Next_num_view_Button.clicked.connect(self.Next_Num_Page)
        # 功能区
        self.app_ui.functionButton.clicked.connect(self.toggle_sub_buttons)
        # 创建子按钮容器（弹出式）
        self.sub_functionButton = QWidget()
        self.sub_functionButton.setWindowFlags(QtCore.Qt.Popup)  # 无标题栏的弹出窗口
        sub_layout = QtWidgets.QHBoxLayout(self.sub_functionButton)
        sub_layout.setContentsMargins(2, 2, 2, 2)  # 减小边距，紧凑显示
        # 导出按钮（绑定fun1：ExportMesh）
        self.append_btn = QPushButton("添加渲染")
        self.append_btn.clicked.connect(self.AppendMesh)  # 绑定删除功能
        self.append_btn.clicked.connect(self.sub_functionButton.hide)  # 点击后隐藏子窗
        self.clear_btn = QPushButton("清空渲染")
        self.clear_btn.clicked.connect(self.ClearMesh)  # 绑定删除功能
        self.clear_btn.clicked.connect(self.sub_functionButton.hide)  # 点击后隐藏子窗
        self.cache_btn = QPushButton("缓存索引")
        self.cache_btn.clicked.connect(self.CacheIndex)  # 绑定删除功能
        self.cache_btn.clicked.connect(self.sub_functionButton.hide)  # 点击后隐藏子窗
        self.export_btn = QPushButton("导出当前")
        self.export_btn.clicked.connect(self.ExportMesh)  # 绑定导出功能
        self.export_btn.clicked.connect(self.sub_functionButton.hide)  # 点击后隐藏子窗口
        # 删除按钮（绑定fun2：比如DeleteMesh）
        self.delete_btn = QPushButton("删除当前")
        self.delete_btn.clicked.connect(self.DeleteMesh)  # 绑定删除功能
        self.delete_btn.clicked.connect(self.sub_functionButton.hide)  # 点击后隐藏子窗


        # 添加子按钮到布局
        sub_layout.addWidget(self.append_btn)
        sub_layout.addWidget(self.clear_btn)
        sub_layout.addWidget(self.cache_btn)
        sub_layout.addWidget(self.export_btn)
        sub_layout.addWidget(self.delete_btn)
        # 初始隐藏子按钮容器
        self.sub_functionButton.hide()



        # 3D界面 
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.app_ui.horizontalLayout.addWidget(self.vtkWidget)
        self.vp = Plotter(N=1, qt_widget=self.vtkWidget)
        self.vp.show(bg="black")
        



   
    
    def pre_processing(self):
        self.UpdateDisplay()
        self.load_view_data()

    ###############################按钮逻辑#######################################

    def toggle_sub_buttons(self):
        """切换子按钮的显示/隐藏状态"""
        if self.sub_functionButton.isHidden():
            # 计算子按钮显示位置（主按钮下方）
            btn_pos = self.app_ui.functionButton.mapToGlobal(self.app_ui.functionButton.rect().bottomLeft())
            self.sub_functionButton.move(btn_pos)
            self.sub_functionButton.show()
        else:
            self.sub_functionButton.hide()


    def AppendMesh(self):
        if not self.db_path:
            return
        # 添加 mesh
        name = "AppendMesh" #渲染标识名
        with Reader(self.db_path) as db:
            data = db[self.count]
            keys =db.get_data_keys(self.count)

        if hasattr(self, "append_data_config"):
            # 有缓存,则直接渲染
            data_config=self.append_data_config
            try:
                show_obj,current_obj =self._get_display_obj(data,data_config)
            except Exception as e:
                QMessageBox.warning(self,"渲染错误",e)
                return

            current_obj.name = name
            self.vp.remove(current_obj)
            self.vp.add(show_obj)
            self.vp.render()
        else:
            # 无缓存,要求用户配置
            data_config=self.data_config
            dialog = DataConfigDialog(keys,data_config, self)
            if dialog.exec_() == QDialog.Accepted:
                data_config = dialog.get_config()
                if data_config == self.data_config:
                    QMessageBox.warning(self,"重复渲染","选择的键已经渲染")
                    return
                try:
                    show_obj,current_obj =self._get_display_obj(data,data_config)
                except Exception as e:
                    QMessageBox.warning(self,"渲染错误",e)
                    return
                current_obj.name = name
                self.vp.add(current_obj)
                self.vp.render()
                # 缓存一个配置，方便用户后续无需重新配置
                self.append_data_config=data_config





    def ClearMesh(self):
        if hasattr(self, "append_data_config"):
            # 有缓存,则删除
            del self.append_data_config
            self.vp.remove("AppendMesh")
            self.vp.render()



    def CacheIndex(self):
        if not self.db_path:
            return
        # 缓存索引
        self.log.info("构建全局缓存索引")
        start_index = 0
        end_index = self.max_count+1

        # 防止用户快速点击视图按钮
        self.app_ui.Next_view_Button.setEnabled(False)
        self.app_ui.Pre_view_Button.setEnabled(False)

        # 启动写入配置线程
        self.write_thread = config_thread(
            self.db_path,
            self.data_config,
            self.data_config["name_key"],
            start_index,
            end_index
        )
        self.write_thread.progress_int.connect(self.app_ui.fun_progressBar.setValue)
        self.write_thread.finished.connect(self.update_view_data)
        self.write_thread.start()


    def DeleteMesh(self):
        try:
            # 确保有可删除的对象
            if self.max_count==0:
                QMessageBox.warning(self, "导出失败", "没有可删除的对象！")
                return
            # 弹出对话框核对
            ok_ = QMessageBox.question(self, "确认删除",f"确认删除当前数据库索引：{self.count}",
                                       QMessageBox.Yes | QMessageBox.No)
            if ok_ == QMessageBox.Yes:
                with Writer(self.db_path,1024*100) as writer:
                    writer.delete_sample(self.count)
                QMessageBox.information(self, "删除成功", f"已删除当前数据库索引：{self.count},重新加载生效!")
        except Exception as e:
            QMessageBox.critical(self, "删除错误", f"出错:\n{str(e)}")
            traceback.print_exc()
    def ExportMesh(self):
        """导出当前视图中的数据，支持json、ply、stl、obj、sm格式"""
        try:
            # 确保有可导出的对象
            if self.current_mesh is None:
                QMessageBox.warning(self, "导出失败", "没有可导出的对象！")
                return

            # 是否导入json
            ok_ = QMessageBox.question(self, "是否将所有信息导出到json",f"确认导出到json数据库索引：{self.count},否则将保存mesh",
                                       QMessageBox.Yes | QMessageBox.No)
            if ok_ == QMessageBox.Yes:
                file_path, _ = QFileDialog.getSaveFileName(
                    self,
                    "保存json文件",
                    os.path.join(os.path.expanduser("~"), "Desktop", f"Json_{self.count}.json"),
                    "JSON Files (*.json)"
                )
                # 如果用户取消选择，则返回
                if not file_path:
                    return
                with Reader(self.db_path) as reader:
                    data = reader[self.count]
                from sindre.utils3d.algorithm import save_np_json
                save_np_json(file_path,data)

                QMessageBox.information(self, "导出成功", f"已成功导出到:\n{file_path}")
                return
            else:
                if isinstance(self.current_mesh,vedo.Image):
                    # 图片
                    file_path, _ = QFileDialog.getSaveFileName(
                        self,
                        "保存图片文件",
                        os.path.join(os.path.expanduser("~"), "Desktop", f"Img_{self.count}.png"),
                    "Image Files (*.png *.jpg *.jpeg)"
                    )
                    # 如果用户取消选择，则返回
                    if not file_path:
                        return
                    # 使用vedo导出网格
                    self.current_mesh.write(file_path)
                    QMessageBox.information(self, "导出成功", f"已成功导出到:\n{file_path}")
                    return

                else:
                    # 网格/点云
                    if isinstance(self.current_mesh,vedo.pointcloud.Points):
                        # 弹出文件保存对话框
                        file_path, _ = QFileDialog.getSaveFileName(
                            self,
                            "保存点云文件",
                            os.path.join(os.path.expanduser("~"), "Desktop", f"mesh_{self.count}.ply"),
                            "PointCloud Files (*.ply *.xyz)"
                        )
                        # 如果用户取消选择，则返回
                        if not file_path:
                            return
                        self.current_mesh.write(file_path)


                    else:
                        # 弹出文件保存对话框
                        file_path, _ = QFileDialog.getSaveFileName(
                            self,
                            "保存点云文件",
                            os.path.join(os.path.expanduser("~"), "Desktop", f"mesh_{self.count}.sm"),
                            "3D Files (*.ply *.stl *.obj *.sm)",
                        )
                        # 如果用户取消选择，则返回
                        if not file_path:
                            return
                        sm = SindreMesh(self.current_mesh)
                        if self.vertex_labels is not None:
                            sm.set_vertex_labels(self.vertex_labels)
                        sm.save(file_path)

                    QMessageBox.information(self, "导出成功", f"已成功导出到:\n{file_path}")

        except Exception as e:
            # 捕获并显示任何错误
            error_msg = f"导出出错:\n{str(e)}"
            QMessageBox.critical(self, "导出错误", error_msg)
            traceback.print_exc()
        
    def change_state_bt_color(self, color=QColor(255, 0, 0)):
        palette = self.app_ui.state_bt.palette()
        palette.setColor(QPalette.Button, color)
        self.app_ui.state_bt.setAutoFillBackground(True)
        self.app_ui.state_bt.setPalette(palette)
        self.app_ui.state_bt.update()

    def SetState(self):
        if not self.db_path:
            return
        # 获取当前文件状态
        key_name = f"STATE_{self.count}" # 公共键
        if key_name in self.data_config:
            current_state = self.data_config[key_name]
        else:
            current_state = "这个数据有以下问题:\n"

        text, ok = QInputDialog.getMultiLineText(self, "输入状态", "请输入需要记录文本:", text=current_state)
        if ok:
            # 保存状态到INI文件
            self.data_config[key_name]= text
            self.save_config()
            self.ShowState()
            self.log.info(f"{key_name}写入:{text}")

    def ShowState(self):
        """显示当前状态"""
        key_name = f"STATE_{self.count}" # 公共键
        if key_name in self.data_config:
            self.app_ui.state_bt.setText("已记录")
            self.change_state_bt_color(color=QColor(0, 255, 0))
        else:
            self.app_ui.state_bt.setText("未记录")
            self.change_state_bt_color(color=QColor(255, 0, 0))

    def JumpCount(self):
        number, ok = QInputDialog.getInt(self, "输入跳转到的序号", f"请输入0-{self.max_count}之间的数值:", min=0,
                                         max=self.max_count)
        if ok:
            if number < 0 or number > self.max_count:
                QMessageBox.critical(self, "错误", "输入的数值超出范围！")
            else:
                self.count = number
                self.UpdateDisplay()
                # 发射信号
                self.countChanged.emit(self.count)

    def NextFile(self):
        if self.count <= self.max_count - 1:
            self.count += 1
            self.UpdateDisplay()
            # 发射信号
            self.countChanged.emit(self.count)

    def PreFile(self):
        if 0 < self.count <= self.max_count:
            self.count -= 1
            self.UpdateDisplay()
            # 发射信号
            self.countChanged.emit(self.count)

    ###############################按钮逻辑#######################################

    ###############################资源视图#######################################

    def load_view_data(self):

        start_index = (self.current_page - 1) * self.page_size
        end_index = self.current_page * self.page_size

        if end_index > self.max_count+1:
            end_index = self.max_count+1
        
        # 防止用户快速点击视图按钮
        self.app_ui.Next_view_Button.setEnabled(False)
        self.app_ui.Pre_view_Button.setEnabled(False)
        
        # 启动写入配置线程
        self.write_thread = config_thread(
            self.db_path,
            self.data_config,
            self.data_config["name_key"], 
            start_index,
            end_index
        )
        self.write_thread.progress_int.connect(self.app_ui.fun_progressBar.setValue)
        self.write_thread.finished.connect(self.update_view_data)
        self.write_thread.start()


    def update_view_data(self):
        start_index = (self.current_page - 1) * self.page_size
        end_index = self.current_page * self.page_size

        
        # 渲染视图
        data = []
        for i in range(start_index, end_index):
            key_name =str(i)
            if key_name in self.data_config:
                data.append(self.data_config[key_name])
            else:
                data.append(f"unknown_{i}")
        
        self.model = QStringListModel()
        self.model.setStringList(data)
        self.app_ui.NameView.setModel(self.model)
        
        # 重新启用按钮
        self.app_ui.Next_view_Button.setEnabled(True)
        self.app_ui.Pre_view_Button.setEnabled(True)

    def Next_Page(self):
        if not self.db_path:
            return
        self.current_page += 1
        self.load_view_data()


    def Pre_Page(self):
        if not self.db_path:
            return
        if self.current_page > 1:
            self.current_page -= 1
            self.load_view_data()

    def Pre_Num_Page(self):
        # 在当前页，选择上一个
        if not self.db_path:
            return
        if self.model.rowCount() == 0:
            return

        # 获取当前选中项的行号
        selection_model = self.app_ui.NameView.selectionModel()
        selected_indexes = selection_model.selectedIndexes()
        current_row = selected_indexes[0].row() if selected_indexes else 0

        # 计算上一个项的行号（不能小于0）
        prev_row = current_row - 1
        if prev_row < 0:
            return  # 已在当前页第一个项，不跳转

        # 选中上一个项
        prev_index = self.model.index(prev_row)
        selection_model.select(prev_index, QItemSelectionModel.Clear | QItemSelectionModel.Select)
        selection_model.setCurrentIndex(prev_index, QItemSelectionModel.NoUpdate)
        key_name  =prev_index.data()
        if key_name in self.data_config:
            # 通过名称获取真实索引
            real_count = self.data_config[key_name]
            if real_count.isdigit():
                self.count=int(real_count)
                self.UpdateDisplay()
                self.countChanged.emit(self.count)


    def Next_Num_Page(self):
        if not self.db_path:
            return
        """在当前页中，选中下一个项（若已在最后一个项，则不操作）"""
        if self.model.rowCount() == 0:
            return

        # 获取当前选中项的行号
        selection_model = self.app_ui.NameView.selectionModel()
        selected_indexes = selection_model.selectedIndexes()
        current_row = selected_indexes[0].row() if selected_indexes else 0

        # 计算下一个项的行号（不能超过当前页最大行号）
        max_row = self.model.rowCount() - 1
        next_row = current_row + 1
        if next_row > max_row:
            return  # 已在当前页最后一个项，不跳转

        # 选中下一个项
        next_index = self.model.index(next_row)
        selection_model.select(next_index, QItemSelectionModel.Clear | QItemSelectionModel.Select)
        selection_model.setCurrentIndex(next_index, QItemSelectionModel.NoUpdate)
        key_name  =next_index.data()
        if key_name in self.data_config:
            # 通过名称获取真实索引
            real_count = self.data_config[key_name]
            if real_count.isdigit():
                self.count=int(real_count)
                self.UpdateDisplay()
                self.countChanged.emit(self.count)



    def search(self):
        if not self.db_path:
            return
        keyword = self.app_ui.search_edit.text().lower()
        filtered_options = []
        
        # 从配置中获取所有选项
        all_options = []
        for  option in self.data_config.keys():
            # 只添加值，不添加键
            if option.isdigit():
                all_options.append(self.data_config[option])

        for option in all_options:
            if keyword in option.lower():
                filtered_options.append(option)
        self.model.setStringList(filtered_options)

    def show_selected_value(self, index):
        selected_option = index.data()
        
        # 从配置中查找对应的索引
        count = None
        if selected_option in self.data_config:
            real_count = self.data_config[selected_option]
            if real_count.isdigit():
                count = int(real_count)
        
        if count is None:
            QMessageBox.warning(self, "警告", "未找到对应的索引!")
            return
        
        ok_ = QMessageBox.question(self, "提示", f"你选择了{selected_option}-->对应序号为{count}的数据",
                                   QMessageBox.Yes | QMessageBox.No)
        if ok_ == QMessageBox.Yes:
            if count is not None:
                self.count = count
                self.UpdateDisplay()

    ###############################资源视图#######################################
    def draw_image_annotations(self, image, bboxes=None, keypoints=None, segmentation=None):
        """在图片上绘制标注"""
        from imgaug.augmentables import Keypoint, KeypointsOnImage
        from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
        from imgaug.augmentables.segmaps import SegmentationMapsOnImage
        colors = SegmentationMapsOnImage.DEFAULT_SEGMENT_COLORS

        if bboxes is not None:
            # 绘制边界框
            bbs_list = []
            color_list=[]
            for bbox in bboxes:
                if len(bbox)==5:
                    bbs_list.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],label=bbox[4]))
                    color_list.append(colors[bbox[4]%len(colors)])
                else:
                    bbs_list.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))
            bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
            for bb, color in zip(bbs.bounding_boxes, color_list):
                image = bb.draw_on_image(image, color=color,alpha=0.8)


        if keypoints is not None:
            # 绘制关键点
            kps_list =[]
            color_list=[]
            for kp in keypoints:
                if len(kp)==3:
                    kps_list.append(Keypoint(x=kp[0], y=kp[1]))
                    color_list.append(colors[kp[3]%len(colors)])
                else:
                    kps_list.append(Keypoint(x=kp[0], y=kp[1]))
                    color_list.append((0,255,0))

            kps = KeypointsOnImage(kps_list, shape=image.shape)
            for kk, color in zip(kps.keypoints, color_list):
                image = kk.draw_on_image(image, color=color)

        if segmentation is not None:
            # 绘制分割掩码（半透明叠加）
            H,W = image.shape[:2]
            segmap = segmentation.reshape(H,W)
            segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
            image = segmap.draw_on_image(image,alpha=0.3)[0]

        return image


    def _labels_flag(self, mesh_vd, labels,is_points=True):
        fss = []
        for i in np.unique(labels):
            if is_points:
                vertices =np.array( mesh_vd.vertices)
                v_i =vertices[labels == i]
            else:
                faces = np.array(mesh_vd.cells)
                faces_indices = np.unique(faces[labels == i])
                v_i = mesh_vd.vertices[faces_indices]
            if len(v_i) > 0:
                cent = np.mean(v_i, axis=0)
                fs = mesh_vd.flagpost(f"{i}", cent)
                fss.append(fs)
        return fss

    def _get_display_obj(self,data,data_config):
        show_obj = None
        current_obj = None
        if data_config["data_type"] == "网格(Mesh)":
            vertices = np.array(get_data_value(data,data_config["vertex_key"]))[...,:3]
            faces = np.array(get_data_value(data,data_config["face_key"]))[...,:3]
            mesh = vedo.Mesh([vertices, faces])
            fss = []

            if data_config["vertex_label_key"]:
                vertex_data = get_data_value(data,data_config["vertex_label_key"])
                if  len(vertex_data.shape) >= 2  and vertex_data.shape[1]==3:
                    # 传入为颜色
                    mesh.pointcolors = vertex_data
                else:
                    # 传入为标签
                    labels=vertex_data.ravel()
                    self.vertex_labels=labels
                    mesh.pointcolors = labels2colors(labels)
                    fss = self._labels_flag(mesh,labels,is_points=True)

            if data_config["face_label_key"] :
                face_data = get_data_value(data,data_config["face_label_key"])
                if len(face_data.shape) >= 2 and face_data.shape[1]==3:
                    # 传入为颜色
                    mesh.cellcolors = face_data
                else:
                    # 传入为标签
                    labels=face_data.ravel()
                    self.vertex_labels=face_labels_to_vertex_labels(np.array(mesh.vertices),np.array(mesh.cells), labels)
                    mesh.cellcolors = labels2colors(labels)
                    fss = self._labels_flag(mesh,labels,is_points=False)

            fss.append(mesh)
            # self.vp.show(fss, axes=3)
            # self.current_mesh = mesh
            current_obj=mesh
            show_obj=fss


        elif data_config["data_type"] == "点云(Point Cloud)":
            points = np.array(get_data_value(data,data_config["vertex_key"])[...,:3])
            pc = Points(points)
            fss = []

            if data_config["vertex_label_key"]:
                vertex_data = get_data_value(data,data_config["vertex_label_key"])
                if len(vertex_data.shape) >= 2 and vertex_data.shape[1]==3:
                    # 传入为颜色
                    pc.pointcolors = vertex_data
                else:
                    # 传入为标签
                    labels=vertex_data.ravel()
                    self.vertex_labels=labels
                    pc.pointcolors = labels2colors(labels)
                    fss = self._labels_flag(pc,labels,is_points=True)

            fss.append(pc)
            # self.vp.show(fss, axes=3)
            # self.current_mesh = pc
            current_obj=pc
            show_obj=fss


        elif data_config["data_type"] == "图片(Image)":
            image = get_data_value(data,data_config["image_key"])
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            # 处理多通道图片
            if len(image.shape) == 3 and image.shape[0] in [1, 3]:  # CHW格式
                image = image.transpose(1, 2, 0)
            if len(image.shape) == 3 and image.shape[2] == 1:  # 单通道转RGB
                image = np.repeat(image, 3, axis=2)
            if len(image.shape) == 2:  # 灰度图转RGB
                image = np.stack([image] * 3, axis=2)


            # 绘制标注
            bboxes = None
            keypoints = None
            segmentation = None
            if data_config.get("bbox_key"):
                bboxes = get_data_value(data,data_config["bbox_key"])
            if data_config.get("keypoints_key"):
                keypoints = get_data_value(data,data_config["keypoints_key"])
            if data_config.get("segmentation_key"):
                segmentation = get_data_value(data,data_config["segmentation_key"])
            annotated_image = self.draw_image_annotations(image, bboxes, keypoints, segmentation)
            # 创建vedo图片对象并显示
            vedo_image = vedo.Image(annotated_image)
            # self.current_mesh = vedo_image
            # self.vp.show(vedo_image)
            current_obj=vedo_image
            show_obj=vedo_image
        return show_obj,current_obj

    def UpdateDisplay(self):
        self.ShowState()
        self.app_ui.treeWidget.clear()
        self.vp.clear(deep=True)
        
        if self.db_path is None:
            QMessageBox.warning(self, "警告", "数据库未打开!")
            return
        with Reader(self.db_path) as db:
            data = db[self.count]
            self.max_count = len(db) - 1

        try:
            show_obj,current_obj =self._get_display_obj(data,self.data_config)
            self.current_mesh = current_obj
            self.vp.show(show_obj, axes=3)


        except KeyError as e:
            QMessageBox.warning(self, "键名错误", f"键名: {str(e)}未在{self.count}数据中找到,请重新配置...")
            with Reader(self.db_path) as db:
                keys =db.get_data_keys(self.count)
            dialog = DataConfigDialog(keys,self.data_config, self)
            if dialog.exec_() == QDialog.Accepted:
                self.data_config = dialog.get_config()
                self.UpdateDisplay()
        except Exception as e:
            QMessageBox.critical(self, "渲染错误", f"渲染数据时出错: {str(e)}")
            traceback.print_exc()
        
        with Reader(self.db_path) as db:
            #spec = db.get_data_specification(0)
            keys = db.get_data_keys(self.count)
            data = db[self.count]

        for key in keys:
            k = str(key)
            current = get_data_value(data,key)
            if isinstance(current, np.ndarray):
                t= f"np_{current.dtype}"
                s= f"{current.shape}"
                if current.size<20:
                    s = str(current)
            elif isinstance(current, dict):
                t= type(current).__name__
                s = f"{len(current)}"
            else:
                t= type(current).__name__
                s=str(current)[:40]

            QtWidgets.QTreeWidgetItem(self.app_ui.treeWidget, [k, t, s])

        if hasattr(self, "append_data_config"):
            self.AppendMesh()
        self.app_ui.NowNumber.display(str(self.count))
        self.app_ui.MaxNumber.display(str(self.max_count))
        self.vp.render()
        # 在更新显示后发射信号
        self.countChanged.emit(self.count)






    def save_config(self):
        """保存配置到文件"""
        if self.db_mb_size!=0:
            with Writer(self.db_path,int(self.db_mb_size+50)) as db:
                db.put_meta("viewer_config",self.data_config)
            self.log.info("保存配置到数据库")

    def onClose(self):
        """保存配置到文件"""
        self.save_config()
        self.vtkWidget.close()
        if self.db_path is not None:
            try:
                self.log.info("尝试修复数据库大小")
                with Writer(self.db_path,1) as db:
                    pass
            except Exception:
                pass


    def OpenFile(self):
        current_path = self.app_ui.path_label.text().strip().replace('"', '').replace('“', '').replace('”', '')
        if current_path and os.path.exists(current_path):
            # 已有有效路径，询问用户是否使用
            reply = QMessageBox.question(
                self, "确认", f"是否打开已选择的路径：\n{current_path}",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes  # 默认Yes，提升效率
            )
            self.fileName = current_path if reply == QMessageBox.Yes else ""
        else:
            # 弹窗，选择文件
            file_path, _ = QFileDialog.getOpenFileName( self, "选取LMDB数据库文件", "./")
            self.fileName = file_path.strip()

        if not self.fileName or not os.path.exists(self.fileName):
            QMessageBox.warning(self, "警告", "未选择有效文件或文件不存在！")
            return


        # 更新UI和路径状态
        self.db_path = self.fileName
        self.db_mb_size = os.path.getsize(self.db_path) / (1024 * 1024)
        self.app_ui.path_label.setText(self.fileName)
        self.log.info(f"路径:{self.db_path},占用空间:{self.db_mb_size}MB")


        # 获取信息
        try:
            with Reader(self.db_path) as db:
                len_db = len(db)
                keys =db.get_data_keys(self.count)
                if "viewer_config" in db.get_meta_keys():
                    self.log.info("从数据库中获取到配置")
                    self.data_config= db.get_meta("viewer_config")

            # 初始化配置
            dialog = DataConfigDialog(keys,self.data_config, self)
            if dialog.exec_() == QDialog.Accepted:
                self.data_config= dialog.get_config()
                self.max_count = len_db
                self.pre_processing()
                self.log.info(f"数据库大小:{self.max_count},开始渲染")
            else:
                QMessageBox.warning(self, "警告", "未完成配置，数据库未加载!")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开数据库失败:{e}")
            traceback.print_exc()


def main_bak():
    # 适应高分辨率
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling) 
    app = QtWidgets.QApplication(sys.argv)
    
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=qdarkstyle.DarkPalette()))
    # 选择启动模式
    choice = QMessageBox.question(None, "启动模式","是否启用单LMDB查看器？\n\n是 - 单查看器模式\n否 - 双查看器模式",QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
    if choice == QMessageBox.No:
        # 双查看器模式
        from DualApp import DualLMDBViewer
        window = DualLMDBViewer()
    elif choice == QMessageBox.Yes:
        # 单查看器模式
        window = LMDB_Viewer()
    else:
        # 取消
        return
    window.show()
    app.aboutToQuit.connect(window.onClose)
    app.exec_()

def main():
    # 适应高分辨率
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=qdarkstyle.DarkPalette()))
    # 单查看器模式
    window = LMDB_Viewer()
    window.show()
    app.aboutToQuit.connect(window.onClose)
    app.exec_()


if __name__ == "__main__":
    main()

