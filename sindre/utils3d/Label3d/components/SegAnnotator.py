"""
曲线分割标注器 - 基于Spline的网格分割标注功能
"""
import json
import os
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog,QButtonGroup
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtGui import QColor
from sklearn.neighbors import KDTree
import vedo
from sindre.utils3d.Label3d.core.manager import CoreSignals
import meshlib.mrmeshnumpy as mrmeshnumpy
from meshlib.mrmeshpy import (smoothRegionBoundary, edgeCurvMetric,
                                fillContourLeftByGraphCut, Mesh, Vector3f,extractMeshContours,
                                findProjection, convertMeshTriPointsToClosedContour,
                                cutMesh)
from importlib.metadata import version
assert version("meshlib")=='3.0.6.229'


class SplineSegmentAnnotator(QWidget):
    """基于Spline的曲线分割标注器"""
    def __init__(self,parent=None, label_dock=None):
        super().__init__(parent)
        self.signals = CoreSignals()
        self.save_info = []  # 存储所有曲线数据
        self.plt = None
        self.label_dock = label_dock  # 引用标签Dock组件
        # 状态
        self.current_spline_points = []  # 当前正在绘制的曲线点
        self.spline_tool = None  # Vedo的spline工具
        self.vdmesh=None # 缓存mesh对象
        self.mmesh=None # 缓存mesh对象
        
        # 网格小组件
        self.mesh_components = []  # 存储Mesh组件：[{\"mesh\": 原始Mesh, \"vdmesh\": Vedo显示Mesh, \"name\": \"组件1\"}, ...]
        self.current_component_idx = 0  # 当前选中的组件索引（默认第1个）

        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        self.dock_content = QWidget()
        self.dock_layout = QVBoxLayout(self.dock_content)
        self.dock_layout.setContentsMargins(10, 10, 10, 10)
        self.dock_layout.setSpacing(10)
        
        # 添加标题
        title_label = QLabel("曲线分割标注")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        self.dock_layout.addWidget(title_label)
        
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
        
        
        # 初始化小组件
        self.mesh_switch_widget = QWidget()
        self.mesh_switch_layout = QHBoxLayout(self.mesh_switch_widget)
        self.mesh_switch_layout.setContentsMargins(0, 5, 0, 5)
        self.mesh_switch_layout.setSpacing(8)  # 按钮间距
        self.mesh_button_group = QButtonGroup(self)
        self.mesh_button_group.setExclusive(True) #设置为互斥模式
        self.dock_layout.addWidget(self.mesh_switch_widget)
        
     
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
            "• 第二步：点击'开始绘制'进入绘制模式",
            "• 第三步：鼠标左键点击模型添加点",
            "• 第四步：右键点击完成当前曲线绘制",
            "• 每个标签只能使用一次",
            "• 编辑模式：拖动点调整曲线形状"
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
            self.label_dock.signals.signal_info.connect(self.on_label_info)
            self.label_dock.signals.signal_labels_updated.connect(self.on_labels_updated)
            self.label_dock.signals.signal_labels_clicked.connect(self.on_labels_clicked)
        
        # 更新按钮状态
        self.update_current_label_display()
        
        # 释放dock组件信号
        self.signals.signal_dock.emit(self.dock_content)
        
        # 释放信息信号
        self.signals.signal_info.emit("曲线分割标注已启动 - 请在左侧选择标签后开始绘制")
        
        return self.dock_content
    
    def update_current_label_display(self):
        """更新当前标签显示"""
        if self.label_dock and self.label_dock.label_manager:
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
                self.current_label_icon.setStyleSheet("background-color: transparent;")
                self.current_label_name.setText("请在左侧选择标签")
                self.current_label_name.setStyleSheet("font-size: 12px; color: #666;")
        else:
            self.current_label_icon.setStyleSheet("background-color: transparent;")
            self.current_label_name.setText("请在左侧选择标签")
            self.current_label_name.setStyleSheet("font-size: 12px; color: #666;")
    
    def on_label_info(self, text):
        """标签信息更新"""
        self.signals.signal_info.emit(text)
        self.update_current_label_display()
    
    def on_labels_updated(self, labels):
        """标签配置更新"""
        self.update_current_label_display()
        if self.plt and self.label_dock:
            for i,info in enumerate(self.save_info):
                label_name = info['label']
                if label_name in labels and  not labels[label_name]['used']:
                    # 用户重置标签
                    self.save_info.pop(i)
                    self.spline_tool=None
                    self.current_spline_points = [] 
                elif label_name not in labels:
                    # 用户删除标签
                    self.save_info.pop(i)
                    self.spline_tool=None
                    self.current_spline_points = [] 
                else:
                    # 用户编辑标签
                    color = self.label_dock.label_manager.get_label_color(label_name)
                    self.save_info[i]['color']=color
            
            self.plt.render()

    def on_labels_clicked(self,labels_name):
        """用户点击标签"""
        selected_data=None
        for info in self.save_info:
            if info["label"]==labels_name:
                selected_data = info
                self.label_dock.label_manager.current_label=labels_name
                self.update_current_label_display()
        # 渲染spline
        if self.spline_tool:
            self.spline_tool.off()
        if selected_data:
            self.vdmesh.alpha(0.5)
            self.spline_tool = self.plt.add_spline_tool(
                    selected_data['pts'], 
                    closed=True,
                    lc=vedo.get_color_name(selected_data['color']),
                    pc="red",
                )
            self.spline_tool.on()
            self.signals.signal_info.emit("进入曲线编辑模式 - 拖动控制点调整形状，按空格键退出编辑")

    
            
    def switch_mesh_component(self, idx):
        """
        切换Mesh组件（按钮点击触发）
        :param idx: int，组件索引（0/1/2）
        """
        if idx < 0 or idx >= len(self.mesh_components):
            return
        
        # 1. 更新当前组件状态
        self.current_component_idx = idx
        self.vdmesh = self.mesh_components[idx]
        self.vdmesh.name = "vdmesh"
        v,f= np.array(self.vdmesh.vertices),np.array(self.vdmesh.cells)
        self.mmesh = mrmeshnumpy.meshFromFacesVerts(f, v)
        
        # 重新渲染
        self.plt.remove("vdmesh")
        self.plt.add(self.vdmesh)
        self.plt.render()
        
        self.signals.signal_info.emit(f"已切换到Mesh组件：{idx}")

        
    
    def render(self, plt):
        """渲染标注"""
        self.plt = plt
        
        if self.vdmesh is None:
            self.vdmesh=self.plt.get_meshes()[0]
            self.vdmesh.name = "vdmesh"
            v,f= np.array(self.vdmesh.vertices),np.array(self.vdmesh.cells)
            self.mmesh = mrmeshnumpy.meshFromFacesVerts(f, v)
            self.mesh_components = self.vdmesh.split()
            
            # 重新渲染
            self.plt.clear()
            self.plt.add(self.vdmesh)
            self.plt.reset_camera()
            self.plt.render()
            
           
            
            # 动态创建组件按钮
            for i,m in enumerate(self.mesh_components):
                btn = QPushButton(f"组件{i+1}")
                btn.setFixedSize(80, 30)  # 固定大小，避免拉伸
                btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f0f0f0;
                        color: #333;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                        font-size: 12px;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                    }
                    QPushButton:checked {
                        background-color: #2196F3;  /* 选中状态高亮（蓝色）*/
                        color: white;
                        border-color: #2196F3;
                    }
                """)
                btn.setCheckable(True)  # 允许选中状态
                btn.clicked.connect(lambda _, idx=i: self.switch_mesh_component(idx))
                self.mesh_button_group.addButton(btn)
                self.mesh_switch_layout.addWidget(btn)
            
 
            
        
        
        # 添加键盘和鼠标回调
        plt.add_callback('on left click', self.on_left_click)
        plt.add_callback('on right click', self.on_right_click)
        plt.add_callback('on key press', self.on_key_press)
    
    def on_left_click(self, evt):
        """左键点击添加点"""
        self.vdmesh.alpha(1)
        current_label_name = self.label_dock.label_manager.current_label
       
        
        if current_label_name not in self.label_dock.label_manager.labels:
            QMessageBox.warning(self, "警告", "请先在左侧选择一个标签！")
            return
        label_info = self.label_dock.label_manager.labels[current_label_name]
        
        if hasattr(evt, 'actor') and evt.actor:
            # 获取点击位置
            if hasattr(evt, 'picked3d') and evt.picked3d is not None:
                # 检查标签是否已使用
                if self.label_dock.label_manager.is_label_used(current_label_name):
                    QMessageBox.warning(self, "警告", f"标签 '{self.label_dock.label_manager.labels[current_label_name]}' 已使用，请选择其他标签！")
                    return
                
                pts =self.vdmesh.closest_point(evt.picked3d)
                self.current_spline_points.append(pts)
                color = label_info['color']
                
                # 如果有足够的点，显示临时线
                if len(self.current_spline_points) >= 2 :
                    # 清除之前的临时显示
                    if hasattr(self, 'temp_points_actor') and self.temp_points_actor:
                        self.plt.remove(self.temp_points_actor)
                    if self.spline_tool is None:
                        self.spline_tool = self.plt.add_spline_tool(
                            self.current_spline_points, 
                            closed=False,
                            lc=vedo.get_color_name(color),
                            lw=3,
                        )
                        self.spline_closed=False
                    else:
                        if  not self.spline_closed and np.linalg.norm(pts-self.current_spline_points[0])<2:
                            self.spline_tool.off()
                            self.spline_tool = self.plt.add_spline_tool(
                                self.current_spline_points, 
                                closed=True,
                                lc=vedo.get_color_name(color),
                                lw=3,
                            )
                            self.spline_closed=True
                        else:
                            self.spline_tool.add(pts)
                else:
                    # 创建临时点显示
                    temp_points = vedo.Points(self.current_spline_points, r=8, c='red')
                    self.temp_points_actor = temp_points
                    self.plt.add(temp_points)
                    
                
                self.plt.render()
                self.signals.signal_info.emit(f"添加点 {len(self.current_spline_points)} - 继续添加或右键完成")
                

                
   
    
    def on_right_click(self, evt):
        """右键点击完成曲线绘制"""
        if self.spline_tool is not None and self.spline_closed  and self.mmesh:
            current_label_name= self.label_dock.label_manager.current_label
            label_info = self.label_dock.label_manager.labels[current_label_name]
            color = label_info['color']
            
            # 获取编辑的曲线
            self.spline_tool.off()
            spline=self.spline_tool.spline()
            
            
            # 投影到网格表面
            tri_points = []
            for p in spline.vertices.tolist():
                v3 = Vector3f(p[0], p[1], p[2])
                projection = findProjection(v3, self.mmesh)
                tri_points.append(projection.mtp)
            
          
                
            # 从投影点创建闭合轮廓
            try:
                contour = convertMeshTriPointsToClosedContour( self.mmesh, tri_points)
          
            
                # # 获取投影点
                # project_pts =[]
                # for pts in extractMeshContours([contour])[0]:
                #     project_pts.append([pts.x,pts.y,pts.z])
                    
                # self.current_spline_points=project_pts
                
                
                
                
                # 使用图割方法选择网格部分
                cut_result = cutMesh(self.mmesh, [contour])
                edge_path = cut_result.resultCut[0]
                one_part = fillContourLeftByGraphCut(
                    self.mmesh.topology,
                    edge_path,
                    edgeCurvMetric( self.mmesh)
                )
                one_part =self.mmesh.topology.getValidFaces() - one_part 
            
                
                # 获取输出
                cut_mesh = Mesh()
                cut_mesh.addPartByMask(self.mmesh, one_part)
                cut_mesh.pack()
                cut_mesh_v = mrmeshnumpy.getNumpyVerts(cut_mesh)
                cut_mesh_f = mrmeshnumpy.getNumpyFaces(cut_mesh.topology)
            except Exception as e:
                print(e)
                QMessageBox.warning(self,"绘制错误","投影错误，请调整分割线")
                self.spline_tool.on()
                return


            # 保存曲线数据
            new_info = {
                "pts":self.current_spline_points,
                'label': current_label_name,
                'vertices':cut_mesh_v,
                'faces':cut_mesh_f,
                'color': color,
            }
            
            self.save_info.append(new_info)
            
            # 清空缓存
            self.spline=None
            self.spline_tool =None
            self.current_spline_points=[]

            
            # 标记标签为已使用
            self.label_dock.use_label(current_label_name)
            self.signals.signal_info.emit(f"创建曲线标签: {current_label_name}")
            
            # 绘制渲染
            cut_vdmesh = vedo.Mesh([cut_mesh_v,cut_mesh_f]).c(color)
            cut_vdmesh.name =current_label_name
            self.plt.add(cut_vdmesh)
            self.vdmesh.alpha(0.2)
            self.plt.render()
            
            
    
    def on_key_press(self, evt):
        """键盘事件处理"""
        print(evt.keypress)
        if hasattr(evt, 'keypress'):
            key = evt.keypress.lower()
            


               
            if key == "space": # 空格键
                if self.spline_tool:
                    self.spline_tool.off()
                    self.spline_tool = None
                    self.signals.signal_info.emit("编辑模式已退出。")
    


    
    def complete_annotations(self):
        """完成标注"""
        # 检查未使用的标签
        if not self.check_unused_labels():
            return
        
        if not self.splines:
            self.signals.signal_info.emit("没有曲线可完成标注")
            return
        
        label_stats = {}
        for spline in self.splines:
            label_name = spline['label_name']
            label_stats[label_name] = label_stats.get(label_name, 0) + 1
        
        stats_text = " | ".join([f"{label}: {count}个" for label, count in label_stats.items()])
        self.signals.signal_info.emit(f"标注完成 - 共添加 {len(self.splines)} 个曲线 ({stats_text})")
        self.signals.signal_close.emit(True)
    

