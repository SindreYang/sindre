"""
sindre.utils3d模块测试用例
测试3D工具类的各种功能
"""

import pytest
import os

# 只测试sindre.utils3d及其子模块的导入和典型API
try:
    import sindre.utils3d
    import sindre.utils3d.sindremesh
    import sindre.utils3d.algorithm
    import sindre.utils3d.pointcloud_augment
    import sindre.utils3d.dental_tools
    import sindre.utils3d.vedo_tools
    import sindre.utils3d.networks
    UTILS3D_AVAILABLE = True
except ImportError:
    UTILS3D_AVAILABLE = False

@pytest.mark.skipif(not UTILS3D_AVAILABLE, reason="utils3d模块不可用")
def test_utils3d_import():
    import sindre.utils3d
    assert hasattr(sindre.utils3d, "mesh")
    assert hasattr(sindre.utils3d, "algorithm")

def test_mesh_api():
    from sindre.utils3d.sindremesh import SindreMesh
    mesh = SindreMesh()
    assert hasattr(mesh, "compute_normals")
    assert hasattr(mesh, "show")

def test_algorithm_api():
    from sindre.utils3d.algorithm import labels2colors
    import numpy as np
    labels = np.array([0, 1, 2, 1, 0])
    colors = labels2colors(labels)
    assert colors.shape[0] == 5

def test_pointcloud_augment_api():
    from sindre.utils3d.pointcloud_augment import Flip_np
    import numpy as np
    points = np.random.rand(10, 3)
    flipper = Flip_np(axis_x=True, axis_y=False)
    flipped = flipper(points)
    assert flipped.shape == (10, 3)

def test_networks_import():
    from sindre.utils3d.networks.pointnet2 import pointnet2_ssg
    from sindre.utils3d.networks.dgcnn import DGCNN
    from sindre.utils3d.networks.conv_occ import ConvPointnet
    from sindre.utils3d.networks.point_transformerV3 import PointTransformerV3

def test_A_star():
    from vedo import Plotter, Mesh, Point, Line,Sphere
    from sindre.utils3d.algorithm import A_Star
    from sindre.utils3d.sindremesh import SindreMesh
    def on_mouse_click(event):
        global selected_points, path_actors
        
        
        # 获取点击位置最近的网格点
        mesh = event.actor
        if not mesh:
            return
        
        # 查找最近的顶点
        point = event.picked3d
        vtx_id = mesh.closest_point(point, return_point_id=True)
        
        # 添加选择点并可视化
        selected_points.append(vtx_id)
        marker = Point(mesh.vertices[vtx_id], c='red', r=12)
        plt.add(marker)
        
        # 如果选择了两个或更多点，计算路径
        if len(selected_points) >= 2:
            # 清除之前的路径
            plt.remove(*path_actors)
            path_actors = []
            
            # 计算所有点之间的连续路径
            all_path_points = []
            for i in range(len(selected_points) - 1):
                start_idx = selected_points[i]
                end_idx = selected_points[i + 1]
                
                # 使用A*算法计算路径
                path_indices = a_star.run(start_idx, end_idx,curvature)
                
                if path_indices:
                    # 获取路径点的坐标
                    path_points = vertices[path_indices]
                    all_path_points.extend(path_points)
                    
                    # 创建路径可视化对象
                    path_line = Line(path_points, c='green', lw=4)
                    path_actors.append(path_line)
                    plt.add(path_line)
                    
                    # 在路径转折点添加标记
                    for j in range(1, len(path_indices) - 1):
                        pt = Point(path_points[j], c='red', r=8)
                        path_actors.append(pt)
                        plt.add(pt)
            
            # 添加最终路径线
            if all_path_points:
                full_path_line = Line(all_path_points, c='yellow', lw=2, alpha=0.5)
                path_actors.append(full_path_line)
                plt.add(full_path_line)
            
            # 添加起点和终点标记
            start_pt = Point(vertices[selected_points[0]], c='green', r=15)
            end_pt = Point(vertices[selected_points[-1]], c='purple', r=15)
            path_actors.extend([start_pt, end_pt])
            plt.add([start_pt, end_pt])
            
        plt.render()

    # 创建测试网格 (一个简单球体)
    sm = SindreMesh( Sphere(res=8))
    sm.get_curvature()
    curvature = sm.vertex_curvature
    curvature =(curvature-curvature.min())/(curvature.max()-curvature.min())
    vertices=sm.vertices



    # 创建A*算法实例
    a_star = A_Star(sm.vertices, sm.faces)

    # 存储用户选择的点
    selected_points = []
    path_actors = []  # 存储路径可视化对象

    # 创建绘图窗口
    plt = Plotter(axes=3, size=(1200, 800))
    # 添加网格到场景
    plt.add(sm.to_vedo)

    # 添加回调函数和键盘快捷键
    plt.add_callback('mouse click', on_mouse_click)
    plt.show().close()
if __name__ == "__main__":
    pytest.main([__file__]) 