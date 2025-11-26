# -*- coding: UTF-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@path   ：sindre_package -> tools.py
@IDE    ：PyCharm
@Author ：sindre
@Email  ：yx@mviai.com
@Date   ：2024/6/17 15:38
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
@History:
- 2024/6/17 :

(一)本代码的质量保证期（简称“质保期”）为上线内 1个月，质保期内乙方对所代码实行包修改服务。
(二)本代码提供三包服务（包阅读、包编译、包运行）不包熟
(三)本代码所有解释权归权归神兽所有，禁止未开光盲目上线
(四)请严格按照保养手册对代码进行保养，本代码特点：
      i. 运行在风电、水电的机器上
     ii. 机器机头朝东，比较喜欢太阳的照射
    iii. 集成此代码的人员，应拒绝黄赌毒，容易诱发本代码性能越来越弱
声明：未履行将视为自主放弃质保期，本人不承担对此产生的一切法律后果
如有问题，热线: 114

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
__author__ = 'sindre'
import json

import trimesh
import vedo
import numpy as np
from typing import *
import vtk
import os
from sindre.general.logs import CustomLogger
from numba import njit, prange
from scipy.spatial import KDTree
log = CustomLogger(logger_name="algorithm").get_logger()



def get_gaussian_heatmap(points, keypoints, sigma=0.5, normalize=False):
    """
    生成高斯热图（仅支持单样本点云和关键点）

    参数:
        points: 点云坐标，形状为 (N, 3)
        keypoints: 关键点坐标，形状为 (K, 3)
        sigma: 高斯标准差，控制热图扩散范围，小则精确定位，大则抗干扰强
        normalize: 是否归一化每个关键点的热图（最大值为1）

    返回:
        heatmap: 热图数组，形状为 (N, K)
    """
    points = points[...,:3]
    # 计算每个点到每个关键点的距离 (N, K)
    dist = np.linalg.norm(
        points[:, np.newaxis, :] - keypoints[np.newaxis, :, :],  # 广播为 (N, K, 3)
        axis=-1
    )

    # 高斯热图
    heatmap = np.exp(-(dist **2) / (2 * sigma** 2))

    # 可选归一化
    if normalize:
        max_vals = heatmap.max(axis=0, keepdims=True)  # 每个关键点的最大值 (1, K)
        heatmap = np.where(max_vals > 1e-6, heatmap / max_vals, heatmap)

    return heatmap


def show_gaussian_heatmap(vertices,heatmap,faces=None):
    """
    用于渲染三维顶点的高斯热力图
    Args:
        vertices: 顶点，形状为 (N, 3)
        heatmap: 热图数组，形状为 (N, K)
        faces: 可选面片,如无，则按照点云渲染

    """
    vertices = np.array(vertices)[...,:3]
    heatmap = np.array(heatmap)
    assert vertices.shape[0] == heatmap.shape[0]
    if faces is None:
        vm = vedo.Points(vertices)
    else:
        vm = vedo.Mesh(vertices, faces)

    max_idx= heatmap.shape[1]-1
    cap = vedo.Text2D(f"{max_idx=} \n"
                      f"{heatmap.shape=}\n"
                      f"max={np.around(heatmap.max(axis=0),3).tolist()}\n"
                      f"min={ np.around(heatmap.min(axis=0),3).tolist()}")

    # 渲染回调
    def render(widget, event):
        idx_=int(widget.value)
        if 0<=idx_<=max_idx:
            # 获取第k个特征点的热图得分（0~1）
            k_scores = heatmap[..., idx_].flatten()  # (N,)
            vm.pointdata["scores"] = k_scores
            vm.cmap("hot", "scores").add_scalarbar(title=f'{idx_=}', horizontal=True)


    plt = vedo.Plotter()
    plt += [vm,cap]
    plt.add_slider(
        render,
        xmin=0,
        xmax=max_idx,
        value=0,
        pos="bottom-right",
        title="idx",
    )
    plt.show().close()


def labels2colors(labels:np.array):
    """
    将labels转换成颜色标签
    Args:
        labels: numpy类型,形状(N)对应顶点的标签；

    Returns:
        RGBA颜色标签;
    """
    labels = labels.reshape(-1)
    from colorsys import hsv_to_rgb
    unique_labels = np.unique(labels)
    num_unique = len(unique_labels)
    
    if num_unique == 0:
        return np.zeros((len(labels), 4), dtype=np.uint8)

    # 生成均匀分布的色相（0-360度），饱和度和亮度固定为较高值
    hues = np.linspace(0, 360, num_unique, endpoint=False)
    s = 0.8  # 饱和度
    v = 0.9  # 亮度
    
    colors = []
    for h in hues:
        # 转换HSV到RGB
        r, g, b = hsv_to_rgb(h / 360.0, s, v)
        # 转换为0-255的整数并添加Alpha通道
        colors.append([int(r * 255), int(g * 255), int(b * 255), 255])
    
    colors = np.array(colors, dtype=np.uint8)
    
    # 创建颜色映射字典
    color_dict = {label: colors[i] for i, label in enumerate(unique_labels)}
    
    # 生成结果数组
    color_labels = np.zeros((len(labels), 4), dtype=np.uint8)
    for label in unique_labels:
        mask = (labels == label)
        color_labels[mask] = color_dict[label]
    
    return color_labels



def color_mapping(value,vmin=-1, vmax=1):
    """将向量映射为颜色，遵从vcg映射标准"""
    import matplotlib.colors as mcolors
    colors = [
        (1.0, 0.0, 0.0, 1.0),  # 红
        (1.0, 1.0, 0.0, 1.0),  # 黄
        (0.0, 1.0, 0.0, 1.0),  # 绿
        (0.0, 1.0, 1.0, 1.0),  # 青
        (0.0, 0.0, 1.0, 1.0)   # 蓝
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("VCG", colors)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    value = norm(np.asarray(value))
    rgba = cmap(value)
    return (rgba * 255).astype(np.uint8)


def vertex_labels_to_face_labels(faces: Union[np.array, list], vertex_labels: Union[np.array, list]) -> np.array:
    """
        将三角网格的顶点标签转换成面片标签，存在一个面片，多个属性，则获取出现最多的属性。

    Args:
        faces: 三角网格面片索引
        vertex_labels: 顶点标签

    Returns:
        面片属性

    """

    # 获取三角网格的面片标签
    face_labels = np.zeros(len(faces))
    for i in range(len(face_labels)):
        face_label = []
        for face_id in faces[i]:
            face_label.append(vertex_labels[face_id])

        # 存在一个面片，多个属性，则获取出现最多的属性
        maxlabel = max(face_label, key=face_label.count)
        face_labels[i] = maxlabel

    return face_labels.astype(np.int32)


def face_labels_to_vertex_labels(vertices: Union[np.array, list], faces: Union[np.array, list],
                                 face_labels: np.array) -> np.array:
    """

    
    将三角网格的面片标签转换成顶点标签

    Args:
        vertices: 
            牙颌三角网格
        faces: 
            面片标签
        face_labels: 
            顶点标签

    Returns:
        顶点属性

    """

    # 获取三角网格的顶点标签
    vertex_labels = np.zeros(len(vertices))
    for i in range(len(faces)):
        for vertex_id in faces[i]:
            vertex_labels[vertex_id] = face_labels[i]

    return vertex_labels.astype(np.int32)




def face_probs_to_vertex_probs(faces, face_probs, n_vertices):
    """将面片概率矩阵转换为顶点概率矩阵（使用max方法）"""
    n_classes = face_probs.shape[1]
    vertex_probs = np.zeros((n_vertices, n_classes))
    # 初始化一个很小的值
    vertex_probs.fill(1e-6)
    for face_idx, face in enumerate(faces):
        prob = face_probs[face_idx]
        for vertex_id in face:
            # 取每个类别的最大值
            vertex_probs[vertex_id] = np.maximum(vertex_probs[vertex_id], prob)
    # 归一化
    row_sums = vertex_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除零
    vertex_probs /= row_sums
    return vertex_probs

def get_axis_rotation(axis: list, angle: float) -> np.array:
    """
        绕着指定轴获取3*3旋转矩阵

    Args:
        axis: 轴向,[0,0,1]
        angle: 旋转角度,90.0

    Returns:
        3*3旋转矩阵

    """

    ang = np.radians(angle)
    R = np.zeros((3, 3))
    ux, uy, uz = axis
    cos = np.cos
    sin = np.sin
    R[0][0] = cos(ang) + ux * ux * (1 - cos(ang))
    R[0][1] = ux * uy * (1 - cos(ang)) - uz * sin(ang)
    R[0][2] = ux * uz * (1 - cos(ang)) + uy * sin(ang)
    R[1][0] = uy * ux * (1 - cos(ang)) + uz * sin(ang)
    R[1][1] = cos(ang) + uy * uy * (1 - cos(ang))
    R[1][2] = uy * uz * (1 - cos(ang)) - ux * sin(ang)
    R[2][0] = uz * ux * (1 - cos(ang)) - uy * sin(ang)
    R[2][1] = uz * uy * (1 - cos(ang)) + ux * sin(ang)
    R[2][2] = cos(ang) + uz * uz * (1 - cos(ang))
    return R


def get_pca_rotation(vertices: np.array) -> np.array:
    """
        通过pca分析顶点，获取3*3旋转矩阵，并应用到顶点；

    Args:
        vertices: 三维顶点

    Returns:
        应用旋转矩阵后的顶点
    """
    from sklearn.decomposition import PCA
    pca_axis = PCA(n_components=3).fit(vertices).components_
    rotation_mat = pca_axis
    vertices = (rotation_mat @ vertices[:, :3].T).T
    return vertices


def get_pca_transform(mesh: vedo.Mesh) -> np.array:
    """
        将输入的顶点数据根据曲率及PCA分析得到的主成分向量，
        并转换成4*4变换矩阵。

    Notes:
        必须为底部非封闭的网格

    Args:
        mesh: vedo网格对象

    Returns:
        4*4 变换矩阵


    """
    """
   
    :param mesh: 
    :return: 
    """
    from sklearn.decomposition import PCA
    vedo_mesh = mesh.clone().decimate(n=5000).clean()
    vertices = vedo_mesh.vertices

    vedo_mesh.compute_curvature(method=1)
    data = vedo_mesh.pointdata['Mean_Curvature']
    verticesn_curvature = vertices[data < 0]

    xaxis, yaxis, zaxis = PCA(n_components=3).fit(verticesn_curvature).components_

    # 通过找边缘最近的点确定z轴方向
    near_point = vedo_mesh.boundaries().center_of_mass()
    vec = near_point - vertices.mean(0)
    user_zaxis = vec / np.linalg.norm(vec)
    if np.dot(user_zaxis, zaxis) > 0:
        # 如果z轴方向与朝向边缘方向相似，那么取反
        zaxis = -zaxis

    """
    plane = vedo.fit_plane(verticesn_curvature)
    m=vedo_mesh.cut_with_plane(plane.center,zaxis).split()[0]
    #m.show()
    vertices = m.points()


    # 将点投影到z轴，重新计算x,y轴
    projected_vertices_xy = vertices - np.dot(vertices, zaxis)[:, None] * zaxis

    # 使用PCA分析投影后的顶点数据
    #xaxis, yaxis = PCA(n_components=2).fit(projected_vertices_xy).components_

    # y = vedo.Arrow(vertices.mean(0), yaxis*5+vertices.mean(0), c="green")
    # x = vedo.Arrow(vertices.mean(0), xaxis*5+vertices.mean(0), c="red")
    # p = vedo.Points(projected_vertices_xy)
    # vedo.show([y,x,p])
    """

    components = np.stack([xaxis, yaxis, zaxis], axis=0)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = components
    transform[:3, 3] = - components @ vertices.mean(0)

    return transform


def apply_transform(vertices: np.array, transform: np.array) -> np.array:
    """
        对4*4矩阵进行应用

    Args:
        vertices: 顶点
        transform: 4*4 矩阵

    Returns:
        变换后的顶点

    """

    # 在每个顶点的末尾添加一个维度为1的数组，以便进行齐次坐标转换
    vertices = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=-1)
    vertices = vertices @ transform
    return vertices[..., :3]


def restore_transform(vertices: np.array, transform: np.array) -> np.array:
    """
        根据提供的顶点及矩阵，进行逆变换(还原应用矩阵之前的状态）

    Args:
        vertices: 顶点
        transform: 4*4变换矩阵

    Returns:
        还原后的顶点坐标

    """
    # 得到转换矩阵的逆矩阵
    inv_transform = np.linalg.inv(transform)

    # 将经过转换后的顶点坐标乘以逆矩阵
    vertices_restored = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=-1)
    vertices_restored = vertices_restored @ inv_transform

    # 最终得到还原后的顶点坐标 vertices_restored
    return  vertices_restored[:, :3]




class NpEncoder(json.JSONEncoder):
    """
    Notes:
        将numpy类型编码成json格式


    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_np_json(output_path: str, obj) -> None:
    """
    保存np形式的json

    Args:
        output_path: 保存路径
        obj: 保存对象


    """

    with open(output_path, 'w') as fp:
        json.dump(obj, fp, cls=NpEncoder)


def get_obb_box(x_pts: np.array, z_pts: np.array, vertices: np.array) -> Tuple[list, list, np.array]:
    """
    给定任意2个轴向交点及顶点，返回定向包围框mesh
    Args:
        x_pts: x轴交点
        z_pts: z轴交点
        vertices: 所有顶点

    Returns:
        包围框的顶点， 面片索引，3*3旋转矩阵

    """

    # 计算中心
    center = np.mean(vertices, axis=0)
    log.debug(center)

    # 定义三个射线
    x_axis = np.array(x_pts - center).reshape(3)
    z_axis = np.array(z_pts - center).reshape(3)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis).reshape(3)

    # 计算AABB
    x_project = np.dot(vertices, x_axis)
    y_project = np.dot(vertices, y_axis)
    z_project = np.dot(vertices, z_axis)
    z_max_pts = vertices[np.argmax(z_project)]
    z_min_pts = vertices[np.argmin(z_project)]
    x_max_pts = vertices[np.argmax(x_project)]
    x_min_pts = vertices[np.argmin(x_project)]
    y_max_pts = vertices[np.argmax(y_project)]
    y_min_pts = vertices[np.argmin(y_project)]

    # 计算最大边界
    z_max = np.dot(z_max_pts - center, z_axis)
    z_min = np.dot(z_min_pts - center, z_axis)
    x_max = np.dot(x_max_pts - center, x_axis)
    x_min = np.dot(x_min_pts - center, x_axis)
    y_max = np.dot(y_max_pts - center, y_axis)
    y_min = np.dot(y_min_pts - center, y_axis)

    # 计算最大边界位移
    inv_x = x_min * x_axis
    inv_y = y_min * y_axis
    inv_z = z_min * z_axis
    x = x_max * x_axis
    y = y_max * y_axis
    z = z_max * z_axis

    # 绘制OBB
    verts = [
        center + x + y + z,
        center + inv_x + inv_y + inv_z,

        center + inv_x + inv_y + z,
        center + x + inv_y + inv_z,
        center + inv_x + y + inv_z,

        center + x + y + inv_z,
        center + x + inv_y + z,
        center + inv_x + y + z,

    ]

    faces = [
        [0, 6, 7],
        [6, 7, 2],
        [0, 6, 3],
        [0, 5, 3],
        [0, 7, 5],
        [4, 7, 5],
        [4, 7, 2],
        [1, 2, 4],
        [1, 2, 3],
        [2, 3, 6],
        [3, 5, 4],
        [1, 3, 4]

    ]
    R = np.vstack([x_axis, y_axis, z_axis]).T
    return verts, faces, R


def get_obb_box_max_min(x_pts: np.array,
                        z_pts: np.array,
                        z_max_pts: np.array,
                        z_min_pts: np.array,
                        x_max_pts: np.array,
                        x_min_pts: np.array,
                        y_max_pts: np.array,
                        y_min_pts: np.array,
                        center: np.array) -> Tuple[list, list, np.array]:
    """
     给定任意2个轴向交点及最大/最小点，返回定向包围框mesh

    Args:
        x_pts: x轴交点
        z_pts: z轴交点
        z_max_pts: 最大z顶点
        z_min_pts:最小z顶点
        x_max_pts:最大x顶点
        x_min_pts:最小x顶点
        y_max_pts:最大y顶点
        y_min_pts:最小y顶点
        center: 中心点

    Returns:
        包围框的顶点， 面片索引，3*3旋转矩阵

    """

    # 定义三个射线
    x_axis = np.array(x_pts - center).reshape(3)
    z_axis = np.array(z_pts - center).reshape(3)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis).reshape(3)

    # 计算最大边界
    z_max = np.dot(z_max_pts - center, z_axis)
    z_min = np.dot(z_min_pts - center, z_axis)
    x_max = np.dot(x_max_pts - center, x_axis)
    x_min = np.dot(x_min_pts - center, x_axis)
    y_max = np.dot(y_max_pts - center, y_axis)
    y_min = np.dot(y_min_pts - center, y_axis)

    # 计算最大边界位移
    inv_x = x_min * x_axis
    inv_y = y_min * y_axis
    inv_z = z_min * z_axis
    x = x_max * x_axis
    y = y_max * y_axis
    z = z_max * z_axis

    # 绘制OBB
    verts = [
        center + x + y + z,
        center + inv_x + inv_y + inv_z,

        center + inv_x + inv_y + z,
        center + x + inv_y + inv_z,
        center + inv_x + y + inv_z,

        center + x + y + inv_z,
        center + x + inv_y + z,
        center + inv_x + y + z,

    ]

    faces = [
        [0, 6, 7],
        [6, 7, 2],
        [0, 6, 3],
        [0, 5, 3],
        [0, 7, 5],
        [4, 7, 5],
        [4, 7, 2],
        [1, 2, 4],
        [1, 2, 3],
        [2, 3, 6],
        [3, 5, 4],
        [1, 3, 4]

    ]
    R = np.vstack([x_axis, y_axis, z_axis]).T
    return verts, faces, R


def create_voxels(vertices, resolution: int = 256):
    """
        通过顶点创建阵列方格体素
    Args:
        vertices: 顶点
        resolution:  分辨率

    Returns:
        返回 res**3 的顶点 , mc重建需要的缩放及位移

    Notes:
        v, f = mcubes.marching_cubes(data.reshape(256, 256, 256), 0)

        m=vedo.Mesh([v*scale+translation, f])


    """
    vertices = np.array(vertices)
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()

    # 使用np.mgrid生成网格索引
    i, j, k = np.mgrid[0:resolution, 0:resolution, 0:resolution]

    # 计算步长（即网格单元的大小）
    dx = (x_max - x_min) / resolution
    dy = (y_max - y_min) / resolution
    dz = (z_max - z_min) / resolution
    scale = np.array([dx, dy, dz])

    # 将索引转换为坐标值
    x = x_min + i * dx
    y = y_min + j * dy
    z = z_min + k * dz
    translation = np.array([x_min, y_min, z_min])

    verts = np.stack((x.ravel(), y.ravel(), z.ravel()), axis=-1)
    # log.info(verts.shape)
    # vedo.show(vedo.Points(verts[::30]),self.crown).close()
    return verts, scale, translation

def compute_face_normals(vertices, faces):
    """
    计算三角形网格中每个面的法线
    Args:
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面数组，形状为 (M, 3)，每个面由三个顶点索引组成
    Returns:
        面法线数组，形状为 (M, 3)
    """
    vertices = np.array(vertices)
    faces = np.array(faces)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)
    
    # 处理退化面（法线长度为0的情况）
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    eps = 1e-8
    norms = np.where(norms < eps, 1.0, norms)  # 避免除以零
    face_normals = face_normals / norms
    
    return face_normals

def compute_vertex_normals(vertices, faces):
    """
    计算三角形网格中每个顶点的法线
    Args:
        vertices: 顶点数组，形状为 (N, 3)
        faces: 面数组，形状为 (M, 3)，每个面由三个顶点索引组成
    Returns:
        顶点法线数组，形状为 (N, 3)
    """
    vertices = np.array(vertices)
    faces = np.array(faces)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # 计算未归一化的面法线（叉积的模长为两倍三角形面积）
    face_normals = np.cross(edge1, edge2)
    
    vertex_normals = np.zeros(vertices.shape)
    # 累加面法线到对应的顶点
    np.add.at(vertex_normals, faces.flatten(), np.repeat(face_normals, 3, axis=0))
    
    # 归一化顶点法线并处理零向量
    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    eps = 1e-8
    lengths = np.where(lengths < eps, 1.0, lengths)  # 避免除以零
    vertex_normals = vertex_normals / lengths
    
    return vertex_normals

def cut_mesh_point_loop(mesh,pts:vedo.Points,invert=False):
    """ 
    
    基于vtk+dijkstra实现的基于线的分割;
    
    线支持在网格上或者网格外；

    Args:
        mesh (_type_): 待切割网格
        pts (vedo.Points): 切割线
        invert (bool, optional): 选择保留外部. Defaults to False.

    Returns:
        _type_: 切割后的网格
    """
    
    # 强制关闭Can't follow edge错误弹窗
    vtk.vtkObject.GlobalWarningDisplayOff()
    selector = vtk.vtkSelectPolyData()
    selector.SetInputData(mesh.dataset)  
    selector.SetLoop(pts.dataset.GetPoints())
    selector.GenerateSelectionScalarsOn()
    selector.Update()
    if selector.GetOutput().GetNumberOfPoints()==0:
        #Can't follow edge
        selector.SetEdgeSearchModeToDijkstra()
        selector.Update()

    
    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(selector.GetOutput())
    clipper.SetInsideOut( not invert)
    clipper.SetValue(0.0)
    clipper.Update()
   

    cut_mesh = vedo.Mesh(clipper.GetOutput())
    vtk.vtkObject.GlobalWarningDisplayOn()
    return cut_mesh









def simplify_by_meshlab(vertices,faces, max_facenum: int = 30000) ->vedo.Mesh:
    """通过二次边折叠算法减少网格中的面数，简化模型。

    Args:
        mesh (pymeshlab.MeshSet): 输入的网格模型。
        max_facenum (int, optional): 简化后的目标最大面数，默认为 200000。

    Returns:
        pymeshlab.MeshSet: 简化后的网格模型。
    """
    import pymeshlab
    
    mesh = pymeshlab.MeshSet()
    mesh.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    mesh.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=max_facenum,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=3,
        preservenormal=True,
        preservetopology=True,
        autoclean=True
    )
    return vedo.Mesh(mesh.current_mesh())






def isotropic_remeshing_by_acvd(vedo_mesh, target_num=10000,clean=True):
    """
    对给定的 vedo 网格进行均质化处理，使其达到指定的目标面数。

    该函数使用 pyacvd 库中的 Clustering 类对输入的 vedo 网格进行处理。
    如果网格的顶点数小于等于目标面数，会先对网格进行细分，然后进行聚类操作，
    最终生成一个面数接近目标面数的均质化网格。

    Args:
        vedo_mesh (vedo.Mesh): 输入的 vedo 网格对象，需要进行均质化处理的网格。
        target_num (int, optional): 目标面数，即经过处理后网格的面数接近该值。
            默认为 10000。
        clean: 去除均匀化错误的点

    Returns:
        vedo.Mesh: 经过均质化处理后的 vedo 网格对象，其面数接近目标面数。

    Notes:
        该函数依赖于 pyacvd 和 pyvista 库，使用前请确保这些库已正确安装。
        
    """
    from pyacvd import Clustering
    from pyvista import wrap
    log.info(" Clustering target_num:{}".format(target_num))
    clus = Clustering(wrap(vedo_mesh.dataset))
    if vedo_mesh.npoints<=target_num:
        clus.subdivide(3)
    clus.cluster(target_num, maxiter=100, iso_try=10, debug=False)
    if clean:
        return vedo.Mesh(clus.create_mesh()).clean()
    else:
        return vedo.Mesh(clus.create_mesh())

def isotropic_remeshing_by_meshlab(mesh, target_edge_length=0.5, iterations=1)-> vedo.Mesh:
    """
    使用 PyMeshLab 实现网格均匀化。

    Args:
        mesh: 输入的网格对象 (pymeshlab.MeshSet)。
        target_edge_length: 目标边长比例 %。
        iterations: 迭代次数，默认为 1。

    Returns:
        均匀化后的网格对象。
    """
    import pymeshlab

    # 应用 Isotropic Remeshing 过滤器
    mesh.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        targetlen=pymeshlab.PercentageValue(target_edge_length),
        iterations=iterations,
    )

    # 返回处理后的网格
    return mesh

def fix_floater_by_meshlab(mesh,nbfaceratio=0.1,nonclosedonly=False) -> vedo.Mesh:
    """移除网格中的浮动小组件（小面积不连通部分）。

    Args:
        mesh (pymeshlab.MeshSet): 输入的网格模型。
        nbfaceratio (float): 面积比率阈值，小于该比率的部分将被移除。
        nonclosedonly (bool): 是否仅移除非封闭部分。

    Returns:
        pymeshlab.MeshSet: 移除浮动小组件后的网格模型。
    """

    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=nbfaceratio, nonclosedonly=nonclosedonly)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return mesh


def fix_invalid_by_meshlab(ms):
    """
    处理冗余元素，如合移除重复面和顶点等, 清理无效的几何结构，如折叠面、零面积面和未引用的顶点。

    Args:
        ms: pymeshlab.MeshSet 对象

    Returns:
        pymeshlab.MeshSet 对象
    """
    ms.apply_filter("meshing_remove_duplicate_faces")
    ms.apply_filter("meshing_remove_duplicate_vertices")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    ms.apply_filter("meshing_remove_folded_faces")
    ms.apply_filter("meshing_remove_null_faces")
    ms.apply_filter("meshing_remove_unreferenced_vertices")
    return ms



def fix_component_by_meshlab(ms):
    """
    移除低质量的组件，如小的连通分量,移除网格中的浮动小组件（小面积不连通部分）。

    Args:
        ms: pymeshlab.MeshSet 对象

    Returns:
        pymeshlab.MeshSet 对象
    """
    ms.apply_filter("meshing_remove_connected_component_by_diameter")
    ms.apply_filter("meshing_remove_connected_component_by_face_number")
    
    return ms

def fix_topology_by_meshlab(ms):
    """
    修复拓扑问题，如 T 型顶点、非流形边和非流形顶点，并对齐不匹配的边界。

    Args:
        ms: pymeshlab.MeshSet 对象

    Returns:
        pymeshlab.MeshSet 对象
    """
    ms.apply_filter("meshing_remove_t_vertices")
    ms.apply_filter("meshing_repair_non_manifold_edges")
    ms.apply_filter("meshing_repair_non_manifold_vertices")
    ms.apply_filter("meshing_snap_mismatched_borders")
    return ms




def labels_mapping(old_vertices,old_faces, new_vertices, old_labels,fast=True):
    """
    将原始网格的标签属性精确映射到新网格
    
    参数:
        old_mesh(vedo) : 原始网格对象
        new_mesh(vedo): 重网格化后的新网格对象
        old_labels (np.ndarray): 原始顶点标签数组，形状为 (N,) 
    
    返回:
        new_labels (np.ndarray): 映射后的新顶点标签数组，形状为 (M,)
    """
    if len(old_labels) != len(old_vertices):
        raise ValueError(f"标签数量 ({len(old_labels)}) 必须与原始顶点数 ({len(old_vertices)}) 一致")

    if fast:
        tree= KDTree( old_vertices)
        _,idx = tree.query(new_vertices,workers=-1)
        return old_labels[idx]
        
    else:
        import trimesh
        old_mesh  = trimesh.Trimesh(old_vertices,old_faces)
        # 步骤1: 查询每个新顶点在原始网格上的最近面片信息
        closest_points, distances, tri_ids = trimesh.proximity.closest_point(old_mesh, new_vertices)
        # 步骤2: 计算每个投影点的重心坐标
        tri_vertices = old_mesh.faces[tri_ids]
        tri_points = old_mesh.vertices[tri_vertices]
        # 计算重心坐标 (M,3)
        bary_coords = trimesh.triangles.points_to_barycentric(
            triangles=tri_points, 
            points=closest_points
        )
        # 步骤3: 确定最大重心坐标对应的顶点
        max_indices = np.argmax(bary_coords, axis=1)
        # 根据最大分量索引选择顶点编号
        nearest_vertex_indices = tri_vertices[np.arange(len(max_indices)), max_indices]
        # 步骤4: 映射标签
        new_labels = np.array(old_labels)[nearest_vertex_indices]
        return new_labels






class BestKFinder:
    def __init__(self, points, labels):
        """
        初始化类，接收点云网格数据和对应的标签
        
        Args:
            points (np.ndarray): 点云数据，形状为 (N, 3)
            labels (np.ndarray): 点云标签，形状为 (N,)
        """
        self.points =  np.array(points)
        self.labels = np.array(labels).reshape(-1)

    def calculate_boundary_points(self, k):
        """
        计算边界点
        :param k: 最近邻点的数量
        :return: 边界点的标签数组
        """
        points = self.points
        tree = KDTree(points)
        _, near_points = tree.query(points, k=k,workers=-1)
        # 确保 near_points 是整数类型
        near_points = near_points.astype(int)
        labels_arr = self.labels[near_points]
        # 将 labels_arr 转换为整数类型
        labels_arr = labels_arr.astype(int)
        label_counts = np.apply_along_axis(lambda x: np.bincount(x).max(), 1, labels_arr)
        label_ratio = label_counts / k
        bdl_ratio = 0.8  # 假设的边界点比例阈值
        bd_labels = np.zeros(len(points))
        bd_labels[label_ratio < bdl_ratio] = 1
        return bd_labels

    def evaluate_boundary_points(self, bd_labels):
        """
        评估边界点的分布合理性
        这里简单使用边界点的数量占比作为评估指标
        :param bd_labels: 边界点的标签数组
        :return: 评估得分
        """
        boundary_ratio = np.sum(bd_labels) / len(bd_labels)
        # 假设理想的边界点比例在 0.1 - 0.2 之间
        ideal_ratio = 0.15
        score = 1 - np.abs(boundary_ratio - ideal_ratio)
        return score

    def find_best_k(self, k_values):
        """
        找出最佳的最近邻点大小
        
        :param k_values: 待测试的最近邻点大小列表
        :return: 最佳的最近邻点大小
        """
        best_score = -1
        best_k = None
        for k in k_values:
            bd_labels = self.calculate_boundary_points(k)
            score = self.evaluate_boundary_points(bd_labels)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k



class LabelUpsampler:
    def __init__(self,
                 classifier_type='gbdt', 
                 knn_params =  {'n_neighbors':3},
                 gbdt_params={'n_estimators': 100, 'max_depth': 5}
                 ):
        """
        标签上采样，用于将简化后的标签映射回原始网格/点云

        Args:
            classifier_type : str, optional (default='gbdt')
                分类器类型，支持 'knn', 'gbdt', 'hgbdt', 'rfc'
        
            knn_params : dict, optional
                KNN分类器参数,默认 {'n_neighbors': 3}
            
            gbdt_params : dict, optional
                GBDT/HGBDT/RFC分类器参数,默认 {'n_estimators': 100, 'max_depth': 5}
        
        """
        self.gbdt_params = gbdt_params
        self.knn_params = knn_params
        self.classifier_type = classifier_type.lower()
        self.clf =None
        
        # 初始化组件
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self._init_classifier()
        
    def _init_classifier(self):
        """初始化内置分类器"""
        if self.classifier_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.clf = KNeighborsClassifier(**self.knn_params)
        elif self.classifier_type == 'gbdt':
            from sklearn.ensemble import GradientBoostingClassifier
            self.clf = GradientBoostingClassifier(**self.gbdt_params)
        elif self.classifier_type == "hgbdt":
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.clf = HistGradientBoostingClassifier(**self.gbdt_params)
        elif self.classifier_type == "rfc":
            from sklearn.ensemble import RandomForestClassifier
            self.clf = RandomForestClassifier(**self.gbdt_params)
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}。"
                             f"支持的类型: ['knn', 'gbdt', 'hgbdt', 'rfc']")


    def fit(self, train_features,  train_labels):
        """
        训练模型: 建议：
        点云： 按照[x,y,z,nx,ny,nz,cv] # 顶点坐标+顶点法线+曲率+其他特征
        网格： 按照[bx,by,bz,fnx,fny,fny] # 面片重心坐标+面片法线+其他特征
        """
        # 特征标准化
        self.scaler.fit(train_features)
        self.clf.fit(self.scaler.transform(train_features), train_labels)
        
    def predict(self, query_features):
        """
        预测标签，输入特征应与训练特征一一对应；
        
        """
        # 预测
        labels = self.clf.predict(self.scaler.transform(query_features))
        return labels


class GraphCutWithMesh:
    import trimesh
    def __init__(self, tri_mesh:trimesh.Trimesh, softmax_labels, smooth_factor=None, keep_label=True):
        """
        基于图切优化器,支持顶点/面片级别优化

        Args:
            tri_mesh (trimesh.Trimesh): trimesh.Trimesh格式的mesh
            softmax_labels (np.ndarray):  softmax后的概率矩阵，形状为：
                                        - 顶点模式：(Nv, class_num)
                                        - 面片模式：(Nf, class_num)
            smooth_factor (float, optional): 平滑强度系数，越大边界越平滑。默认值为 None，此时会自动计算。
            keep_label (bool, optional): 是否保持优化前后标签类别一致性，默认值为 True。

        """
        # 初始化
        self.smooth_factor = smooth_factor
        self.keep_label=keep_label
        self.labels = softmax_labels
        self.labels_type=None
        self.labels_classes=None
        self.mesh = tri_mesh
        self._preprocess()

    def _preprocess(self):
        """预处理"""
        """检测输入类型并验证形状"""
        n_vertices = self.mesh.vertices.shape[0]
        n_faces = self.mesh.faces.shape[0]
        labels_shape = self.labels.shape
        self.mesh.fix_normals()

        if labels_shape[0] == n_vertices:
            if len(labels_shape)>1:
                self.labels_type = 0
                self.labels_classes = labels_shape[1]
                log.info(f"检测到输入标签类型: 顶点标签的概率矩阵 (Nv, class_num),类别数:{self.labels_classes}")
            else:
                self.labels_type = 1

                raise ValueError("只支持顶点标签的概率矩阵 (Nv, class_num")
            # 提供顶点级信息
            self.vertices=self.mesh.vertices
            self.normals =self.mesh.vertex_normals
            self.adj  = [np.array(list(adj)) for adj in self.mesh.vertex_neighbors]


        elif labels_shape[0] == n_faces:
            if len(labels_shape)>1:
                self.labels_type = 2
                self.labels_classes = labels_shape[1]
                log.info(f"检测到输入标签类型: 面片标签的概率矩阵 (Nf, class_num),类别数:{self.labels_classes}")
            else:
                self.labels_type = 3
                raise ValueError("只支持面片标签的概率矩阵 (Nf, class_num")


            # 提供面片级信息
            self.vertices=self.mesh.triangles_center
            self.normals = self.mesh.face_normals
            face_adjacency = self.mesh.face_adjacency
            # 使用字典和集合来构建邻接关系(避免后续去重)
            face_adj = [set() for _ in range(len(self.mesh.faces))]
            # 批量添加邻接关系
            for f1, f2 in face_adjacency:
                face_adj[f1].add(f2)
                face_adj[f2].add(f1)
            self.adj = [np.sort(list(adj_set)) for adj_set in face_adj]
        else:
            raise ValueError("标签样本维度应与顶点数或面片数一致")



        # 获取势能
        self.unaries,  self.pairwise,  self.edges_weight = self.get_energy()

        # 自动参数优化强度计算 #1.2s
        if self.smooth_factor is None:
            # 取中值作为根据
            weights_raw = self.edges_weight[:, 2]
            unary_median = np.median(np.abs(self.unaries))
            weight_median = np.median(weights_raw) if weights_raw.size else 1.0
            self.smooth_factor= min([unary_median / max(weight_median, 1e-6)*0.8 ,1e4])#经验值



        log.info("Ready: "
                 f"vertex={n_vertices} "
                 f"face={n_faces} "
                 f"type={self.labels_type} "
                 f"n_class={self.labels_classes} "
                 f"smooth_factor={self.smooth_factor:.2f} "
                 f"keep_label={self.keep_label} ")



    def get_weights(self):
        # 获取边权
        """计算顶点间边权"""
        edges_weight = []
        for i, neighbors in enumerate(self.adj):
            for j in neighbors:
                if j <= i:
                    continue  # 避免重复

                ni, nj = self.normals[i], self.normals[j]
                if self.labels_type < 2:
                    theta = np.arccos(np.clip(np.dot(ni, nj), -1.0, 1.0))
                else:
                    theta = np.arccos(np.clip(np.dot(ni, nj)/np.linalg.norm(ni)/np.linalg.norm(nj), -1.0, 1.0))
                distance = np.linalg.norm(self.vertices[i] - self.vertices[j])


                # 边权计算
                theta = max(theta, 1e-6)  # 防止除零
                weight = -np.log10(theta/np.pi) * distance
                if theta < np.pi/2:
                    weight = 10 * weight
                # 按照点1,点2, 1与2的权重存储
                edges_weight.append([i, j,weight])
        return np.array(edges_weight, dtype=np.float32)

    def get_energy(self):
        # 获取一元势能和二元势能
        # 计算一元势能（原始标签）
        prob = self.labels
        prob = np.clip(prob, 1e-6, 1.0)

        unaries = (-100 * np.log10(prob)).astype(np.int32)

        # 建立图结构邻接表
        pairwise = (1 - np.eye(self.labels_classes, dtype=np.int32))

        # 二元势能 (边权)
        edges_weight = self.get_weights()
        return unaries, pairwise, edges_weight


    def refine(self):
        """执行优化并返回优化后的标签索引"""
        try:
            from pygco import cut_from_graph
            if self.keep_label:
                optimized_labels = None
                terminate_after_next = False  # 标记是否在下一次迭代后终止
                max_iterations = 10
                for i  in  range(max_iterations):
                    # 强化边权
                    new_edges_weight = self.edges_weight
                    new_edges_weight[:, 2] = (self.edges_weight[:, 2] * self.smooth_factor)
                    print("gco")
                    refine_labels = cut_from_graph( new_edges_weight.astype(np.int32), self.unaries, self.pairwise)
                    unique_count =len(np.unique(refine_labels))
                    if optimized_labels is None:
                        optimized_labels =refine_labels
                    if terminate_after_next and unique_count== self.labels_classes:
                        break  # 执行了额外的一次优化，终止循环
                    if unique_count== self.labels_classes:
                        optimized_labels =refine_labels
                        self.smooth_factor*=1.5
                        log.info(f"当前smooth_factor={self.smooth_factor},优化中({i+1}/10)....")
                    elif unique_count== 1:
                        self.smooth_factor*=0.6
                        log.info(f"当前smooth_factor={self.smooth_factor},优化中({i+1}/10)....")
                        terminate_after_next = True  # 标记下次迭代后终止
                        optimized_labels = None
                    else:
                        # 优化结束
                        break
            else:
                # 按照给定值优化一次
                self.edges_weight[:, 2] = (self.edges_weight[:, 2] * self.smooth_factor)
                optimized_labels = cut_from_graph(self.edges_weight.astype(np.int32), self.unaries, self.pairwise)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"图切优化失败: {str(e)}") from e

        return optimized_labels







class UnifiedLabelRefiner:
    def __init__(self, vertices, faces, labels, class_num, smooth_factor=None, temperature=None):
        """
        统一多标签优化器，支持顶点/面片概率输入
        
        Args:
            vertices (np.ndarray): 顶点坐标数组，形状 (Nv, 3)
            faces (np.ndarray):    面片索引数组，形状 (Nf, 3)
            labels (np.ndarray):   初始标签，可以是类别索引（一维）(n,) 或概率矩阵，形状为：
                                        - 顶点模式：(Nv, class_num) 
                                        - 面片模式：(Nf, class_num)
            class_num (int):       总类别数量（必须等于labels.shape[1]）
            smooth_factor (float): 边权缩放因子，默认自动计算
            temperature (float):   标签软化温度（None表示不软化）
        """
        # 输入验证
        if len(labels.shape) == 1:
            num_samples = labels.shape[0]
            self.labels = np.zeros((num_samples, class_num), dtype=np.float32)
            self.labels[np.arange(num_samples), labels] = 1.0
        else:
            self.labels = labels.astype(np.float32)
   
        
        # 构建trimesh对象
        import trimesh
        self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 检测输入类型
        self.class_num = class_num
        self.label_type = self._detect_label_type()
        
        # 预处理几何特征
        self._precompute_geometry()
        
        # 初始化参数
        self.smooth_factor = smooth_factor
        self.temperature = temperature

    def _detect_label_type(self):
        """检测输入类型并验证形状"""
        n_vertices = len(self.mesh.vertices)
        n_faces = len(self.mesh.faces)
        
        if self.labels.shape[0] == n_vertices:
            assert self.labels.shape == (n_vertices, self.class_num), \
                "顶点概率矩阵形状应为({}, {})".format(n_vertices, self.class_num)
            return 'vertex'
        
        if self.labels.shape[0] == n_faces:
            assert self.labels.shape == (n_faces, self.class_num), \
                "面片概率矩阵形状应为({}, {})".format(n_faces, self.class_num)
            return 'face'
        
        raise ValueError("概率矩阵的样本维度应与顶点数或面片数一致")

    def _precompute_geometry(self):
        """预计算几何特征"""
        # 顶点级特征
        self.mesh.fix_normals()
        self.vertex_normals = self.mesh.vertex_normals.copy()
        self.vertex_adj = [np.array(list(adj)) for adj in self.mesh.vertex_neighbors]
        
        # 面片级特征
        self.face_normals = self.mesh.face_normals.copy()
        self.face_centers = self.mesh.triangles_center.copy()
        self.face_adj = self._compute_face_adjacency()

    def _compute_face_adjacency(self):
        """计算面片邻接关系"""
        face_adj = []
        for i, face in enumerate(self.mesh.faces):
            # 查找共享两个顶点的面片
            shared = np.sum(np.isin(self.mesh.faces, face), axis=1)
            adj_faces = np.where(shared == 2)[0]
            # 排除自身并排序防止重复
            face_adj.append(adj_faces[adj_faces > i])
        return face_adj

    def refine(self):
        """执行优化并返回优化后的标签索引"""
        # 概率软化处理
        processed_prob = self._process_probability()
        
        # 计算unary势能
        unaries = (-100 * np.log10(processed_prob)).astype(np.int32)
        
        # 自动参数计算
        if self.smooth_factor is None:
            edges_raw = self._compute_edges(scale=1.0)
            weights_raw = edges_raw[:, 2]
            unary_median = np.median(np.abs(unaries))
            weight_median = np.median(weights_raw) if weights_raw.size else 1.0
            self.smooth_factor= min([unary_median / max(weight_median, 1e-6)*0.8 ,1e4])#经验值
        
        # 构建图结构并优化
        pairwise = (1 - np.eye(self.class_num, dtype=np.int32))

    
        from pygco import cut_from_graph
        optimized_labels = None
        terminate_after_next = False  # 标记是否在下一次迭代后终止
        for i  in  range(10):
            # 计算边权重
            edges = self._compute_edges(self.smooth_factor)
            refine_labels = cut_from_graph(edges, unaries, pairwise)
            unique_count =len(np.unique(refine_labels))
            if optimized_labels is None:
                optimized_labels =refine_labels
            if terminate_after_next and unique_count== self.class_num:
                break  # 执行了额外的一次优化，终止循环
            if unique_count==  self.class_num:
                optimized_labels =refine_labels
                self.smooth_factor*=1.5
                log.info(f"当前smooth_factor={self.smooth_factor},优化中({i+1}/10)....")
            elif unique_count== 1:
                self.smooth_factor*=0.6
                log.info(f"当前smooth_factor={self.smooth_factor},优化中({i+1}/10)....")
                terminate_after_next = True  # 标记下次迭代后终止
                optimized_labels = None
            else:
                # 优化结束
                break
            
        return optimized_labels #cut_from_graph(edges, unaries, pairwise)

    def _process_probability(self):
        """概率矩阵后处理"""
        prob = np.clip(self.labels, 1e-6, 1.0)
        
        # 温度软化
        if self.temperature is not None and self.temperature != 1.0:
            prob = np.exp(np.log(prob) / self.temperature)
            prob /= prob.sum(axis=1, keepdims=True)
        
        return prob

    def _compute_edges(self,scale=1.0):
        """根据类型计算边权"""
        if self.label_type == 'vertex':
            return self._compute_vertex_edges(scale)
        return self._compute_face_edges(scale)

    def _compute_vertex_edges(self,scale):
        """计算顶点间边权"""
        edges = []
        for i, neighbors in enumerate(self.vertex_adj):
            for j in neighbors:
                if j <= i:
                    continue  # 避免重复
                
                # 计算几何特征
                ni, nj = self.vertex_normals[i], self.vertex_normals[j]
                theta = np.arccos(np.clip(np.dot(ni, nj), -1.0, 1.0))
                dist = np.linalg.norm(self.mesh.vertices[i] - self.mesh.vertices[j])
                
                # 边权计算
                weight = self._calculate_edge_weight(theta, dist)
                edges.append([i, j, int(weight * scale)])
        
        return np.array(edges, dtype=np.int32)

    def _compute_face_edges(self,scale):
        """计算面片间边权"""
        edges = []
        for i, neighbors in enumerate(self.face_adj):
            for j in neighbors:
                # 计算面片特征
                ni, nj = self.face_normals[i], self.face_normals[j]
                theta = np.arccos(np.clip(np.dot(ni, nj)/np.linalg.norm(ni)/np.linalg.norm(nj), -1.0, 1.0))
                dist = np.linalg.norm(self.face_centers[i] - self.face_centers[j])
                
                # 边权计算（放大权重）
                weight = self._calculate_edge_weight(theta, dist) 
                edges.append([i, j, int(weight*scale)])
        
        return np.array(edges, dtype=np.int32)

    @staticmethod
    def _calculate_edge_weight(theta, distance):
        """统一边权计算公式"""
        theta = max(theta, 1e-6)  # 防止除零
        if theta > np.pi/2:
            return -np.log10(theta/np.pi) * distance
        return -10 * np.log10(theta/np.pi) * distance















@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def furthestsampling_jit(xyz: np.ndarray, offset: np.ndarray, new_offset: np.ndarray) -> np.ndarray:
    """使用并行批次处理的最远点采样算法实现
    
    该方法将输入点云划分为多个批次，每个批次独立进行最远点采样。通过维护最小距离数组，
    确保每次迭代选择距离已选点集最远的新点，实现高效采样。

    Args:
        xyz (np.ndarray): 输入点云坐标，形状为(N, 3)的C连续float32数组
        offset (np.ndarray): 原始点云的分段偏移数组，表示每个批次的结束位置。例如[1000, 2000]表示两个批次
        new_offset (np.ndarray): 采样后的分段偏移数组，表示每个批次的目标采样数。例如[200, 400]表示每批采200点

    Returns:
        np.ndarray: 采样点索引数组，形状为(total_samples,)，其中total_samples = new_offset[-1]

    Notes:
        实现特点:
        - 使用Numba并行加速，支持多核并行处理不同批次
        - 采用平方距离计算避免开方运算
        - 每批次独立初始化距离数组，避免跨批次干扰
        - 自动处理边界情况（空批次或零采样批次）

        典型调用流程:
        >>> n_total = 10000
        >>> offset = np.array([1000, 2000, ..., 10000], dtype=np.int32)
        >>> new_offset = np.array([200, 400, ..., 2000], dtype=np.int32)
        >>> sampled_indices = furthestsampling_jit(xyz, offset, new_offset)
    """
    # 确保输入为C连续的float32数组
    total_samples = new_offset[-1]
    indices = np.empty(total_samples, dtype=np.int32)
    
    # 并行处理每个批次
    for bid in prange(len(new_offset)):
        # 确定批次边界
        if bid == 0:
            n_start, n_end = 0, offset[0]
            m_start, m_end = 0, new_offset[0]
        else:
            n_start = offset[bid-1]
            n_end = offset[bid]
            m_start = new_offset[bid-1]
            m_end = new_offset[bid]
        
        batch_size = n_end - n_start
        sample_size = m_end - m_start
        
        if batch_size == 0 or sample_size == 0:
            continue
        
        # 提取当前批次的点坐标（三维）
        batch_xyz = xyz[n_start:n_end]
        x = batch_xyz[:, 0]  # x坐标数组
        y = batch_xyz[:, 1]  # y坐标数组
        z = batch_xyz[:, 2]  # z坐标数组
        
        # 初始化最小距离数组
        min_dists = np.full(batch_size, np.finfo(np.float32).max, dtype=np.float32)
        
        # 首点选择批次内的第一个点
        current_local_idx = 0
        indices[m_start] = n_start + current_local_idx  # 转换为全局索引
        
        # 初始化最新点坐标
        last_x = x[current_local_idx]
        last_y = y[current_local_idx]
        last_z = z[current_local_idx]
        
        # 主采样循环
        for j in range(1, sample_size):
            max_dist = -1.0
            best_local_idx = 0
            
            # 遍历所有点更新距离并寻找最大值
            for k in range(batch_size):
                # 计算到最新点的平方距离
                dx = x[k] - last_x
                dy = y[k] - last_y
                dz = z[k] - last_z
                dist = dx*dx + dy*dy + dz*dz
                
                # 更新最小距离
                if dist < min_dists[k]:
                    min_dists[k] = dist
                
                # 跟踪当前最大距离
                if min_dists[k] > max_dist:
                    max_dist = min_dists[k]
                    best_local_idx = k
            
            # 更新当前最优点的索引和坐标
            current_local_idx = best_local_idx
            indices[m_start + j] = n_start + current_local_idx  # 转换为全局索引
            last_x = x[current_local_idx]
            last_y = y[current_local_idx]
            last_z = z[current_local_idx]
    
    return indices

def farthest_point_sampling(vertices: np.ndarray, n_sample: int = 2000, auto_seg: bool = False, n_batches: int = 10) -> np.ndarray:
    """
    最远点采样，支持自动分批处理
    
    根据参数配置，自动决定是否将输入点云分割为多个批次进行处理。当处理大规模数据时，
    建议启用auto_seg以降低内存需求并利用并行加速。

    Args:
        vertices (np.ndarray): 输入点云坐标，形状为(N, 3)的浮点数组
        n_sample (int, optional): 总采样点数，当auto_seg=False时生效。默认2000
        auto_seg (bool, optional): 是否启用自动分批处理(提速，但会丢失全局距离信息)。默认False
        n_batches (int, optional): 自动分批时的批次数量。默认10

    Returns:
        np.ndarray: 采样点索引数组，形状为(n_sample,)

    Raises:
        ValueError: 当输入数组维度不正确时抛出

    Notes:
        典型场景:
        - 小规模数据（如5万点以下）: auto_seg=False，单批次处理
        - 大规模数据（如百万级点）: auto_seg=True，分10批处理，每批采样2000点

        示例:
        >>> vertices = np.random.rand(100000, 3).astype(np.float32)
        >>> # 自动分10批，每批采2000点
        >>> indices = farthest_point_sampling(vertices, auto_seg=True)
        >>> # 单批采5000点
        >>> indices = farthest_point_sampling(vertices, n_sample=5000)
    """
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("输入点云必须是形状为(N, 3)的二维数组")
    xyz =np.ascontiguousarray(vertices, dtype=np.float32) 
    n_total = xyz.shape[0]
    if auto_seg:
         # 计算批次采样数分配
        base_samples = n_sample // n_batches
        remainder = n_sample % n_batches
        # 创建采样数数组，前remainder个批次多采1点
        batch_samples = [base_samples + 1 if i < remainder else base_samples 
                        for i in range(n_batches)]
        # 生成偏移数组（累加形式）
        new_offset = np.cumsum(batch_samples).astype(np.int32)
        # 原始点云分批偏移（均匀分配）
        batch_size = n_total // n_batches
        offset = np.array([batch_size*(i+1) for i in range(n_batches)], dtype=np.int32)
        offset[-1] = n_total  # 最后一批包含余数点
        
    else:
        offset = np.array([n_total], dtype=np.int32)
        new_offset = np.array([n_sample], dtype=np.int32)
    return  furthestsampling_jit(xyz,offset,new_offset)


def farthest_point_sampling_by_open3d(vertices: np.ndarray, n_sample: int = 2000,device ="CPU:0") -> np.ndarray:
    """
    基于Open3D的最远点采样算法，返回采样点的索引数组

   该函数利用Open3D库的高效实现，从输入的点云中按最远点策略采样指定数量的点，
   并返回这些采样点在原始点云中的索引，便于后续还原采样前的点云数据。

   Args:
       vertices: 输入点云数据，形状为[N, 3]的numpy数组，其中N为点的数量，3对应xyz坐标
       n_sample: 期望采样的点数量，默认值为2000
       device: 计算设备，可选"CPU:0"或"CUDA:1"等，默认使用CPU

   Returns:
       采样点的索引数组，形状为[n_sample]的numpy数组，元素为原始点云的索引值

   Raises:
       若输入点云数量小于n_sample，可能会抛出Open3D内部异常
       若设备指定无效（如CUDA不可用时指定"CUDA:1"），会抛出设备初始化错误
   """
    import open3d as o3d
    device = o3d.core.Device(device)
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions =o3d.core.Tensor(np.ascontiguousarray(vertices,dtype=np.float32), dtype, device)
    # 用索引代替标签，方便还原
    pcd.point.labels = o3d.core.Tensor(np.arange(len(vertices)) ,o3d.core.int32, device)
    downpcd_farthest = pcd.farthest_point_down_sample(n_sample)
    idx= downpcd_farthest.point.labels.cpu().numpy()
    return idx

    


def add_base(vd_mesh,value_z=-20,close_base=True,return_strips=False):
    """给网格边界z方向添加底座

    Args:
        vd_mesh (_type_):vedo.mesh
        value_z (int, optional): 底座长度. Defaults to -20.
        close_base (bool, optional): 底座是否闭合. Defaults to True.
        return_strips (bool, optional): 是否返回添加的网格. Defaults to False.

    Returns:
        _type_: 添加底座的网格
    """
    
    # 开始边界
    boundarie_start = vd_mesh.clone().boundaries()
    boundarie_start =boundarie_start.generate_delaunay2d(mode="fit").boundaries()
    # TODO:补充边界损失
    # 底座边界
    boundarie_end= boundarie_start.copy()
    boundarie_end.vertices[...,2:]=value_z
    strips = boundarie_start.join_with_strips(boundarie_end)
    merge_list=[vd_mesh,strips]
    if return_strips:
        return strips
    if close_base:
        merge_list.append(boundarie_end.generate_delaunay2d(mode="fit"))
    out_mesh = vedo.merge(merge_list).clean()
    return out_mesh


def equidistant_mesh(mesh, d=-0.01,merge=True):
    """

    此函数用于创建一个与输入网格等距的新网格，可选择将新网格与原网格合并。


    Args:
        mesh (vedo.Mesh): 输入的三维网格对象。
        d (float, 可选): 顶点偏移的距离，默认为 -0.01。负值表示向内偏移，正值表示向外偏移。
        merge (bool, 可选): 是否将原网格和偏移后的网格合并，默认为 True。

    Returns:
        vedo.Mesh 或 vedo.Assembly: 如果 merge 为 True，则返回合并后的网格；否则返回偏移后的网格。
    """
    mesh.compute_normals().clean() 
    cells = np.asarray(mesh.cells)
    original_vertices = mesh.vertices
    vertex_normals = mesh.vertex_normals
    pts_id =mesh.boundaries(return_point_ids=True)
    
    # 创建边界掩码
    boundary_mask = np.zeros(len(original_vertices), dtype=bool)
    boundary_mask[pts_id] = True
    
    # 仅对非边界顶点应用偏移
    pts = original_vertices.copy()
    pts[~boundary_mask] += vertex_normals[~boundary_mask] * d
    
    # 构建新网格
    offset_mesh = vedo.Mesh([pts, cells]).clean()
    if merge:
        return vedo.merge([mesh,offset_mesh])
    else:
        return offset_mesh
    

def voxel2array(grid_index_array, voxel_size=32):
    """
    将 voxel_grid_index 数组转换为固定大小的三维数组。

    该函数接收一个形状为 (N, 3) 的 voxel_grid_index 数组，
    并将其转换为形状为 (voxel_size, voxel_size, voxel_size) 的三维数组。
    其中，原 voxel_grid_index 数组中每个元素代表三维空间中的一个网格索引，
    在转换后的三维数组中对应位置的值会被设为 1，其余位置为 0。

    Args:
        grid_index_array (numpy.ndarray): 形状为 (N, 3) 的数组，
            通常从 open3d 的 o3d.voxel_grid.get_voxels() 方法获取，
            表示三维空间中每个体素的网格索引。
        voxel_size (int, optional): 转换后三维数组的边长，默认为 32。

    Returns:
        numpy.ndarray: 形状为 (voxel_size, voxel_size, voxel_size) 的三维数组，
            其中原 voxel_grid_index 数组对应的网格索引位置值为 1，其余为 0。

    Example:
        ```python
        # 获取 grid_index_array
        voxel_list = voxel_grid.get_voxels()
        grid_index_array = list(map(lambda x: x.grid_index, voxel_list))
        grid_index_array = np.array(grid_index_array)
        voxel_grid_array = voxel2array(grid_index_array, voxel_size=32)
        grid_index_array = array2voxel(voxel_grid_array)
        pointcloud_array = grid_index_array  # 0.03125 是体素大小
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud_array)
        o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=0.05)
        o3d.visualization.draw_geometries([pcd, cc, o3d_voxel])
        ```
    """
    array_voxel = np.zeros((voxel_size, voxel_size, voxel_size))
    array_voxel[grid_index_array[:, 0], grid_index_array[:, 1], grid_index_array[:, 2]] = 1
    return array_voxel

    
def array2voxel(voxel_array):
    """
        将固定大小的三维数组转换为 voxel_grid_index 数组。
        该函数接收一个形状为 (voxel_size, voxel_size, voxel_size) 的三维数组，
        找出其中值为 1 的元素的索引，将这些索引组合成一个形状为 (N, 3) 的数组，
        类似于从 open3d 的 o3d.voxel_grid.get_voxels () 方法获取的结果。
        
    Args:
        voxel_array (numpy.ndarray): 形状为 (voxel_size, voxel_size, voxel_size) 的三维数组，数组中值为 1 的位置代表对应的体素网格索引。
    
    Returns:
    
        numpy.ndarray: 形状为 (N, 3) 的数组，表示三维空间中每个体素的网格索引，类似于从 o3d.voxel_grid.get_voxels () 方法获取的结果。
    
    Example:
    
        ```python
        
        # 获取 grid_index_array
        voxel_list = voxel_grid.get_voxels()
        grid_index_array = list(map(lambda x: x.grid_index, voxel_list))
        grid_index_array = np.array(grid_index_array)
        voxel_grid_array = voxel2array(grid_index_array, voxel_size=32)
        grid_index_array = array2voxel(voxel_grid_array)
        pointcloud_array = grid_index_array  # 0.03125 是体素大小
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud_array)
        o3d_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pc, voxel_size=0.05)
        o3d.visualization.draw_geometries([pcd, cc, o3d_voxel])
                
        
        ```
    
    """
    x, y, z = np.where(voxel_array == 1)
    index_voxel = np.vstack((x, y, z))
    grid_index_array = index_voxel.T
    return grid_index_array









def fill_hole_with_center(mesh,boundaries,return_vf=False):
    """
        用中心点方式强制补洞

    Args:
        mesh (_type_): vedo.Mesh
        boundaries:vedo.boundaries
        return_vf: 是否返回补洞的mesh


    """
    vertices = mesh.vertices.copy()
    cells = mesh.cells

    # 获取孔洞边界的顶点坐标
    boundaries = boundaries.join(reset=True)
    if not boundaries:
        return mesh  # 没有孔洞
    pts_coords = boundaries.vertices

    # 将孔洞顶点坐标转换为原始顶点的索引
    hole_indices = []
    for pt in pts_coords:
        distances = np.linalg.norm(vertices - pt, axis=1)
        idx = np.argmin(distances)
        if distances[idx] < 1e-6:
            hole_indices.append(idx)
        else:
            raise ValueError("顶点坐标未找到")

    n = len(hole_indices)
    if n < 3:
        return mesh  # 无法形成面片

    # 计算中心点并添加到顶点
    center = np.mean(pts_coords, axis=0)
    new_vertices = np.vstack([vertices, center])
    center_idx = len(vertices)

    # 生成新的三角形面片
    new_faces = []
    for i in range(n):
        v1 = hole_indices[i]
        v2 = hole_indices[(i + 1) % n]
        new_faces.append([v1, v2, center_idx])
        
    if return_vf:
        return vedo.Mesh([new_vertices, new_faces]).clean().compute_normals()
    # 合并面片并创建新网格
    updated_cells = np.vstack([cells, new_faces])
    new_mesh = vedo.Mesh([new_vertices, updated_cells])
    return new_mesh.clean().compute_normals()






def collision_depth(mesh1, mesh2) -> float:
    """计算两个网格间的碰撞深度或最小间隔距离。

    使用VTK的带符号距离算法检测碰撞状态：
    - 正值：两网格分离，返回值为最近距离
    - 零值：表面恰好接触
    - 负值：发生穿透，返回值为最大穿透深度（绝对值）

    Args:
        mesh1 (vedo.Mesh): 第一个网格对象，需包含顶点数据
        mesh2 (vedo.Mesh): 第二个网格对象，需包含顶点数据

    Returns:
        float: 带符号的距离值，符号表示碰撞状态，绝对值表示距离量级
        
    Raises:
        RuntimeError: 当VTK计算管道出现错误时抛出

    Notes:
        1. 当输入网格顶点数>1000时会产生性能警告
        2. 返回float('inf')表示计算异常或无限远距离

    """
    # 性能优化提示
    if mesh1.npoints > 1000 or mesh2.npoints > 1000:
        log.info("[性能警告] 检测到高精度网格(顶点数>1000)，建议执行 mesh.decimate(n=500) 进行降采样")

    try:
        # 初始化VTK距离计算器
        distance_filter = vtk.vtkDistancePolyDataFilter()
        distance_filter.SetInputData(0, mesh1.dataset)
        distance_filter.SetInputData(1, mesh2.dataset)
        distance_filter.SignedDistanceOn()
        distance_filter.Update()

        # 提取距离数据
        distance_array = distance_filter.GetOutput().GetPointData().GetScalars("Distance")
        if not distance_array:
            return float('inf')
            
        return distance_array.GetRange()[0]
        
    except Exception as e:
        raise RuntimeError(f"VTK距离计算失败: {str(e)}") from e
    
    
    
    
def compute_curvature_by_meshlab(ms):
    """
    使用 MeshLab 计算网格的曲率和顶点颜色。

    该函数接收一个顶点矩阵和一个面矩阵作为输入，创建一个 MeshLab 的 MeshSet 对象，
    并将输入的顶点和面添加到 MeshSet 中。然后，计算每个顶点的主曲率方向，
    最后获取顶点颜色矩阵和顶点曲率数组。

    Args:
        ms: pymeshlab格式mesh;

    Returns:
        - vertex_colors (numpy.ndarray): 顶点颜色矩阵，形状为 (n, 3)，其中 n 是顶点的数量。
            每个元素的范围是 [0, 255]，表示顶点的颜色。
        - vertex_curvature (numpy.ndarray): 顶点曲率数组，形状为 (n,)，其中 n 是顶点的数量。
            每个元素表示对应顶点的曲率。
        - new_vertex (numpy.ndarray): 新的顶点数组，形状为 (n,)，其中 n 是顶点的数量。
        

    """
    ms.compute_curvature_principal_directions_per_vertex()
    curr_ms = ms.current_mesh()
    vertex_colors =curr_ms.vertex_color_matrix()*255
    vertex_curvature=curr_ms.vertex_scalar_array()
    new_vertex  =curr_ms.vertex_matrix()
    return vertex_colors,vertex_curvature,new_vertex


def compute_curvature_by_igl(v,f,method="Mean"):
    """
    用igl计算平均曲率并归一化

    Args:
        v: 顶点;
        f: 面片:
        method:返回曲率类型

    Returns:
        - vertex_curvature (numpy.ndarray): 顶点曲率数组，形状为 (n,)，其中 n 是顶点的数量。
            每个元素表示对应顶点的曲率。
            
    Notes:
    
        输出: PD1 (主方向1), PD2 (主方向2), PV1 (主曲率1), PV2 (主曲率2)
        
        pd1 : #v by 3 maximal curvature direction for each vertex
        pd2 : #v by 3 minimal curvature direction for each vertex
        pv1 : #v by 1 maximal curvature value for each vertex
        pv2 : #v by 1 minimal curvature value for each vertex


    """
    try:
        import igl
    except ImportError:
        log.info("请安装igl, pip install libigl>=2.6.1")
    PD1, PD2, PV1, PV2,_  = igl.principal_curvature(v, f)

    if "Gaussian" in method:
        # 计算高斯曲率（Gaussian Curvature）
        K = PV1 * PV2
    elif "Mean" in method:
        # 计算平均曲率（Mean Curvature）
        K = 0.5 * (PV1 + PV2)
    else:
        K=[PD1, PD2, PV1, PV2]
    return K


def harmonic_by_igl(v,f,map_vertices_to_circle=True):
    """
    谐波参数化后的2D网格

    Args:
        v (_type_): 顶点
        f (_type_): 面片
        map_vertices_to_circle: 是否映射到圆形（正方形)

    Returns:
        uv,v_p: 创建参数化后的2D网格,3D坐标
        
    Note:
    
        ```
         
        # 创建空间索引
        uv_kdtree = KDTree(uv)
        
        # 初始化可视化系统
        plt = Plotter(shape=(1, 2), axes=False, title="Interactive Parametrization")
        
        # 创建网格对象
        mesh_3d = Mesh([v, f]).cmap("jet", calculate_curvature(v, f)).lighting("glossy")
        mesh_2d = Mesh([v_p, f]).wireframe(True).cmap("jet", calculate_curvature(v, f))
        
        # 存储选中标记
        markers_3d = []
        markers_2d = []

        def on_click(event):
            if not event.actor or event.actor not in [mesh_2d, None]:
                return
            if not hasattr(event, 'picked3d') or event.picked3d is None:
                return
            
            try:
                # 获取点击坐标
                uv_click = np.array(event.picked3d[:2])
                
                # 查找最近顶点
                _, idx = uv_kdtree.query(uv_click)
                v3d = v[idx]
                uv_point = uv[idx]  # 获取对应2D坐标
                
                
                # 创建3D标记（使用球体）
                marker_3d = Sphere(v3d, r=0.1, c='cyan', res=12)
                markers_3d.append(marker_3d)
                
                # 创建2D标记（使用大号点）
                marker_2d = Point(uv_point, c='magenta', r=10, alpha=0.8)
                markers_2d.append(marker_2d)
                
                # 更新视图
                plt.at(0).add(marker_3d)
                plt.at(1).add(marker_2d)
                plt.render()
                
            except Exception as e:
                log.info(f"Error processing click: {str(e)}")

        plt.at(0).show(mesh_3d, "3D Visualization", viewup="z")
        plt.at(1).show(mesh_2d, "2D Parametrization").add_callback('mouse_click', on_click)
        plt.interactive().close()
            
        
        ``` 
        
    """
    try:
        import igl
    except ImportError:
        log.info("请安装igl, pip install libigl")
    v=np.array(v,dtype=np.float32)
    # 正方形边界映射）
    def map_to_square(bnd):
        n = len(bnd)
        quarter = n // 4
        uv = np.zeros((n, 2))
        for i in range(n):
            idx = i % quarter
            side = i // quarter
            t = idx / (quarter-1)
            if side == 0:   uv[i] = [1, t]
            elif side == 1: uv[i] = [1-t, 1]
            elif side == 2: uv[i] = [0, 1-t]
            else:           uv[i] = [t, 0]
        return uv
    try:
        # 参数化
        bnd = igl.boundary_loop(f)
        if map_vertices_to_circle:
            bnd_uv = igl.map_vertices_to_circle(v, bnd)  # 圆形参数化
        else:
            bnd_uv = map_to_square(bnd)                # 正方形参数化
        uv = igl.harmonic(v, f, bnd, bnd_uv, 1)
    except Exception as e:
        log.info(f"生成错误，请检测连通体数量，{e}")
    # 创建参数化后的2D网格（3D坐标）
    v_p = np.hstack([uv, np.zeros((uv.shape[0], 1))])
    
    return uv,v_p










def hole_filling_by_Radial(boundary_coords):
    """
    参考 
    
    [https://www.cnblogs.com/shushen/p/5759679.html]
    
    实现的最小角度法补洞法；

    Args:
        boundary_coords (_type_): 有序边界顶点

    Returns:
        v,f: 修补后的曲面
        
        
    Note:
        ```python 
        
        # 创建带孔洞的简单网格
        s = vedo.load(r"J10166160052_16.obj")
        # 假设边界点即网格边界点
        boundary =vedo.Spline((s.boundaries().join(reset=True).vertices),res=100)
        # 通过边界点进行补洞
        filled_mesh =vedo.Mesh(hole_filling(boundary.vertices))
        # 渲染补洞后的曲面
        vedo.show([filled_mesh,boundary,s.alpha(0.8)], bg='white').close()
        
        ```
    
    """
    # 初始化顶点列表和边界索引
    vertex_list = np.array(boundary_coords.copy())
    boundary = list(range(len(vertex_list)))  # 存储顶点在vertex_list中的索引
    face_list = []

    while len(boundary) >= 3:
        # 1. 计算平均边长
        avg_length = 0.0
        n_edges = len(boundary)
        for i in range(n_edges):
            curr_idx = boundary[i]
            next_idx = boundary[(i+1)%n_edges]
            avg_length += np.linalg.norm(vertex_list[next_idx] - vertex_list[curr_idx])
        avg_length /= n_edges

        # 2. 寻找最小内角顶点在边界列表中的位置
        min_angle = float('inf')
        min_idx = 0  # 默认取第一个顶点
        for i in range(len(boundary)):
            prev_idx = boundary[(i-1)%len(boundary)]
            curr_idx = boundary[i]
            next_idx = boundary[(i+1)%len(boundary)]
            
            v1 = vertex_list[prev_idx] - vertex_list[curr_idx]
            v2 = vertex_list[next_idx] - vertex_list[curr_idx]
            # 检查向量长度避免除以零
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            if v1_norm == 0 or v2_norm==0:
                continue  # 跳过无效顶点
            cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
            angle = np.arccos(np.clip(cos_theta, -1, 1))
            if angle < min_angle:
                min_angle = angle
                min_idx = i  # 记录边界列表中的位置

        # 3. 获取当前处理的三个顶点索引
        curr_pos = min_idx
        prev_pos = (curr_pos - 1) % len(boundary)
        next_pos = (curr_pos + 1) % len(boundary)
        
        prev_idx = boundary[prev_pos]
        curr_idx = boundary[curr_pos]
        next_idx = boundary[next_pos]

        # 计算前驱和后继顶点的距离
        dist = np.linalg.norm(vertex_list[next_idx] - vertex_list[prev_idx])

        # 4. 根据距离决定添加三角形的方式
        if dist < 2 * avg_length:
            # 添加单个三角形
            face_list.append([prev_idx, curr_idx, next_idx])
            # 从边界移除当前顶点
            boundary.pop(curr_pos)
        else:
            # 创建新顶点并添加到顶点列表
            new_vertex = (vertex_list[prev_idx] + vertex_list[next_idx]) / 2
            vertex_list = np.vstack([vertex_list, new_vertex])
            new_idx = len(vertex_list) - 1

            # 添加两个三角形
            face_list.append([prev_idx, curr_idx, new_idx])
            face_list.append([curr_idx, next_idx, new_idx])

            # 更新边界：替换当前顶点为新顶点
            boundary.pop(curr_pos)
            boundary.insert(curr_pos, new_idx)

    return vertex_list, face_list








class A_Star:
    def __init__(self,vertices, faces):
        """
        使用A*算法在三维三角网格中寻找最短路径
        
        参数：
        vertices: numpy数组，形状为(N,3)，表示顶点坐标
        faces: numpy数组，形状为(M,3)，表示三角形面的顶点索引
        
        """
        self.adj=self.build_adjacency(faces)
        self.vertices = vertices
        

    def build_adjacency(self,faces):
        """构建顶点的邻接表"""
        from collections import defaultdict
        adj = defaultdict(set)
        for face in faces:
            for i in range(3):
                u = face[i]
                v = face[(i + 1) % 3]
                adj[u].add(v)
                adj[v].add(u)
        return {k: list(v) for k, v in adj.items()}

    def run(self,start_idx, end_idx, vertex_weights=None):
        """
        使用A*算法在三维三角网格中寻找最短路径
        
        参数：
        start_idx: 起始顶点的索引
        end_idx: 目标顶点的索引
        vertex_weights: 可选，形状为(N,)，顶点权重数组，默认为None
        
        返回：
        path: 列表，表示从起点到终点的顶点索引路径，若不可达返回None
        """
        import heapq
        end_coord = self.vertices[end_idx]
        
        # 启发式函数（当前顶点到终点的欧氏距离）
        def heuristic(idx):
            return np.linalg.norm(self.vertices[idx] - end_coord)
        
        # 优先队列：(f, g, current_idx)
        open_heap = []
        heapq.heappush(open_heap, (heuristic(start_idx), 0, start_idx))
        
        # 记录各顶点的g值和父节点
        g_scores = {start_idx: 0}
        parents = {}
        closed_set = set()
        
        while open_heap:
            current_f, current_g, current_idx = heapq.heappop(open_heap)
            
            # 若当前节点已处理且有更优路径，跳过
            if current_idx in closed_set:
                if current_g > g_scores.get(current_idx, np.inf):
                    continue
            # 找到终点，回溯路径
            if current_idx == end_idx:
                path = []
                while current_idx is not None:
                    path.append(current_idx)
                    current_idx = parents.get(current_idx)
                return path[::-1]
            
            closed_set.add(current_idx)
            
            # 遍历邻接顶点
            for neighbor in self.adj.get(current_idx, []):
                if neighbor in closed_set:
                    continue
                
                # 计算移动代价
                distance = np.linalg.norm(self.vertices[current_idx] - self.vertices[neighbor])
                if vertex_weights is not None:
                    cost = distance *vertex_weights[neighbor] 
                else:
                    cost = distance
                
                tentative_g = current_g + cost
                
                # 更新邻接顶点的g值和父节点
                if tentative_g < g_scores.get(neighbor, np.inf):
                    parents[neighbor] = current_idx
                    g_scores[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))
        
        # 开放队列空，无路径
        return None




    
    
    
    
    
    
    
    
    



class MeshRandomWalks:
    def __init__(self, vertices, faces, face_normals=None):
        """
        随机游走分割网格
        
        参考：https://www.cnblogs.com/shushen/p/5144823.html
        
        Args:
            vertices: 顶点坐标数组，形状为(N, 3)
            faces: 面片索引数组，形状为(M, 3)
            face_normals: 可选的面法线数组，形状为(M, 3)
            
            
        Note:
        
            ```python
            
                # 加载并预处理网格
                mesh = vedo.load(r"upper_jaws.ply")
                mesh.compute_normals()
                
                # 创建分割器实例
                segmenter = MeshRandomWalks(
                    vertices=mesh.points,
                    faces=mesh.faces(),
                    face_normals=mesh.celldata["Normals"]
                )
                
                head = [1063,3571,1501,8143]
                tail = [7293,3940,8021]
                
                # 执行分割
                labels, unmarked = segmenter.segment(
                    foreground_seeds=head,
                    background_seeds=tail
                )
                p1 = vedo.Points(mesh.points[head],r=20,c="red")
                p2 = vedo.Points(mesh.points[tail],r=20,c="blue")
                # 可视化结果
                mesh.pointdata["labels"] = labels
                mesh.cmap("jet", "labels")
                vedo.show([mesh,p1,p2], axes=1, viewup='z').close()
            ```
        """
        self.vertices = np.asarray(vertices)
        self.faces = np.asarray(faces, dtype=int)
        self.face_normals = face_normals
        
        # 自动计算面法线（如果未提供）
        if self.face_normals is None:
            self.face_normals = self._compute_face_normals()
        
        # 初始化其他属性
        self.edge_faces = None
        self.edge_weights = None
        self.W = None       # 邻接矩阵
        self.D = None       # 度矩阵
        self.L = None       # 拉普拉斯矩阵
        self.labels = None  # 顶点标签
        self.marked = None  # 标记点掩码

    def _compute_face_normals(self):
        """计算每个面片的单位法向量"""
        normals = []
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            vec1 = v1 - v0
            vec2 = v2 - v0
            normal = np.cross(vec1, vec2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            else:
                normal = np.zeros(3)  # 处理退化面片
            normals.append(normal)
        return np.array(normals)

    def _compute_edge_face_map(self):
        """构建边到面片的映射关系"""

        from collections import defaultdict
        edge_map = defaultdict(list)
        for fid, face in enumerate(self.faces):
            for i in range(3):
                v0, v1 = sorted([face[i], face[(i+1)%3]])
                edge_map[(v0, v1)].append(fid)
        self.edge_faces = edge_map

    def _compute_edge_weights(self):
        """基于面片法线计算边权重"""
        self._compute_edge_face_map()
        self.edge_weights = {}
        
        for edge, fids in self.edge_faces.items():
            if len(fids) != 2:
                continue  # 只处理内部边
                
            # 获取相邻面法线
            n1, n2 = self.face_normals[fids[0]], self.face_normals[fids[1]]
            
            # 计算角度差异权重
            cos_theta = np.dot(n1, n2)
            eta = 1.0 if cos_theta < 0 else 0.2
            d = eta * (1 - abs(cos_theta))
            self.edge_weights[edge] = np.exp(-d)

    def _build_adjacency_matrix(self):
        """构建顶点邻接权重矩阵"""
        from scipy.sparse import csr_matrix, lil_matrix

        n = len(self.vertices)
        self.W = lil_matrix((n, n))
        
        for (v0, v1), w in self.edge_weights.items():
            self.W[v0, v1] = w
            self.W[v1, v0] = w
        
        self.W = self.W.tocsr()  # 转换为压缩格式提高效率

    def _build_laplacian_matrix(self):
        """构建拉普拉斯矩阵 L = D - W"""
        from scipy.sparse import csr_matrix
        degrees = self.W.sum(axis=1).A.ravel()
        self.D = csr_matrix((degrees, (range(len(degrees)), range(len(degrees)))))
        self.L = self.D - self.W

    def segment(self, foreground_seeds, background_seeds, vertex_weights=None):
        """
        执行网格分割
        
        参数:
            foreground_seeds: 前景种子点索引列表
            background_seeds: 背景种子点索引列表
            vertex_weights: 可选的顶点权重矩阵（稀疏矩阵）
        
        返回:
            labels: 顶点标签数组 (0: 背景，1: 前景)
            unmarked: 未标记顶点的布尔掩码
        """
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import csr_matrix
        # 初始化标签数组
        n = len(self.vertices)
        self.labels = np.full(n, -1, dtype=np.float64)
        self.labels[foreground_seeds] = 1.0
        self.labels[background_seeds] = 0.0

        # 处理权重矩阵
        if vertex_weights is not None:
            self.W = vertex_weights
        else:
            if not self.edge_weights:
                self._compute_edge_weights()
            if self.W is None:
                self._build_adjacency_matrix()
        
        # 构建拉普拉斯矩阵
        self._build_laplacian_matrix()

        # 分割问题求解
        self.marked = self.labels != -1
        L_uu = self.L[~self.marked, :][:, ~self.marked]
        L_ul = self.L[~self.marked, :][:, self.marked]
        rhs = -L_ul @ self.labels[self.marked]

        # 求解并应用阈值
        L_uu_reg = L_uu + 1e-9 * csr_matrix(np.eye(L_uu.shape[0])) #防止用户选择过少造成奇异值矩阵
        try:
            x = spsolve(L_uu_reg, rhs)
        except:
            # 使用最小二乘法作为备选方案
            x = np.linalg.lstsq(L_uu_reg.toarray(), rhs, rcond=None)[0]
        self.labels[~self.marked] = (x > 0.5).astype(int)
        
        return self.labels.astype(int), ~self.marked
    
    
    
    
    
    
    
    

def mesh2sdf(v, f, size=64):
    """
    体素化网格，该函数适用于非水密网格（带孔的网格）、自相交网格、具有非流形几何体的网格以及具有方向不一致的面的网格。

    Args:
        v (array-like): 网格的顶点数组。
        f (array-like): 网格的面数组。
        size (int, optional): 体素化的大小，默认为 64。

    Returns:
        array: 体素化后的数组。

    Raises:
        ImportError: 如果未安装 'mesh-to-sdf' 库，会提示安装。
    """
    import trimesh
    try:
        from mesh_to_sdf import mesh_to_voxels
    except ImportError:
        log.info("请安装依赖库：pip install mesh-to-sdf")

    mesh = trimesh.Trimesh(v, f)

    voxels = mesh_to_voxels(mesh, size, pad=True)
    return voxels


def sample_sdf_mesh(v, f, number_of_points=200000):
    """
    在曲面附近不均匀地采样 SDF 点，该函数适用于非水密网格（带孔的网格）、自相交网格、具有非流形几何体的网格以及具有方向不一致的面的网格。
    这是 DeepSDF 论文中提出和使用的方法。

    Args:
        v (array-like): 网格的顶点数组。
        f (array-like): 网格的面数组。
        number_of_points (int, optional): 采样点的数量，默认为 200000。

    Returns:
        tuple: 包含采样点数组和对应的 SDF 值数组的元组。

    Raises:
        ImportError: 如果未安装 'mesh-to-sdf' 库，会提示安装。
    """
    import trimesh
    try:
        from mesh_to_sdf import sample_sdf_near_surface
    except ImportError:
        log.info("请安装依赖库：pip install mesh-to-sdf")

    mesh = trimesh.Trimesh(v, f)

    points, sdf = sample_sdf_near_surface(mesh, number_of_points=number_of_points)
    return points, sdf
    
    
    
def resample_mesh(vertices, faces, density=1, num_samples=None):
    """在由顶点和面定义的网格表面上进行点云重采样。
    
    1. 密度模式：根据单位面片面积自动计算总采样数
    2. 指定数量模式：直接指定需要采样的总点数

    该函数使用向量化操作高效地在网格表面进行均匀采样，采样密度由单位面积点数决定。
    采样策略基于重心坐标系，采用分层随机抽样方法。

    注意：
        零面积三角形会被自动跳过，因为不会分配采样点。

    参考实现：
        https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/

    Args:
        vertices (numpy.ndarray): 网格顶点数组，形状为(V, 3)，V表示顶点数量
        faces (numpy.ndarray): 三角形面片索引数组，形状为(F, 3)，数据类型应为整数
        density (float, 可选): 每单位面积的采样点数，默认为1
        num_samples (int, 可选): 指定总采样点数，若提供则忽略density参数

    Returns:
        numpy.ndarray: 重采样后的点云数组，形状为(N, 3)，N为总采样点数

    Notes:
        采样点生成公式（重心坐标系）：
            P = (1 - √r₁)A + √r₁(1 - r₂)B + √r₁ r₂ C
        其中：
        - r₁, r₂ ∈ [0, 1) 为随机数
        - A, B, C 为三角形顶点
        - 该公式可确保在三角形表面均匀采样

        算法流程：
        1. 计算每个面的面积并分配采样点数
        2. 通过随机舍入处理总点数误差
        3. 使用向量化操作批量生成采样点

    References:
        [1] Barycentric coordinate system - https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    """
    # 计算每个面的法向量并计算面的面积
    vec_cross = np.cross(
        vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
        vertices[faces[:, 1], :] - vertices[faces[:, 2], :],
    )
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    
    if num_samples is not None:
        n_samples = num_samples
        # 按面积比例分配采样数
        ratios = face_areas / face_areas.sum()
        n_samples_per_face = np.random.multinomial(n_samples, ratios)
    else:
        # 计算需要采样的总点数
        n_samples = (np.sum(face_areas) * density).astype(int)
        # face_areas = face_areas / np.sum(face_areas)

        # 为每个面分配采样点数
        # 首先，过度采样点并去除多余的点
        # Bug 修复由 Yangyan (yangyan.lee@gmail.com) 完成
        n_samples_per_face = np.ceil(density * face_areas).astype(int)
        
    
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # 创建一个包含面索引的向量
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample


    # 生成随机数
    r = np.random.rand(n_samples, 2)
    faces_samples = faces[sample_face_idx]
    A = vertices[faces_samples[:, 0]]
    B = vertices[faces_samples[:, 1]]
    C = vertices[faces_samples[:, 2]]

    # 使用重心坐标公式计算采样点
    P = (
        (1 - np.sqrt(r[:, 0:1])) * A
        + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B
        + np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    )
    
    # # 随机采样
    # if num_samples is not None:
    #     idx = np.random.choice(len(P), num_samples,replace=False)
    #     P=P[idx]

    return P
    
    
    
    
    
def subdivide_loop_by_trimesh(
    vertices,
    faces,
    iterations=5,
    max_face_num=100000,
    face_mask=None,
):
    """
    
    对给定的顶点和面片进行 Loop 细分。

    Args:
        vertices (array-like): 输入的顶点数组，形状为 (n, 3)，其中 n 是顶点数量。
        faces (array-like): 输入的面片数组，形状为 (m, 3)，其中 m 是面片数量。
        iterations (int, optional): 细分的迭代次数，默认为 5。
        max_face_num (int, optional): 细分过程中允许的最大面片数量，达到此数量时停止细分，默认为 100000。
        face_mask (array-like, optional): 面片掩码数组，用于指定哪些面片需要进行细分，默认为 None。

    Returns:
        tuple: 包含细分后的顶点数组、细分后的面片数组和面片掩码数组的元组。

    Notes:
        以下是一个示例代码，展示了如何使用该函数：
        ```python
        # 1. 获取每个点的最近表面点及对应面
        face_indices = set()
        kdtree = cKDTree(mesh.vertices)
        for p in pts:
            # 查找半径2mm内的顶点
            vertex_indices = kdtree.query_ball_point(p, r=1.0)
            for v_idx in vertex_indices:
                # 获取包含这些顶点的面片
                faces = mesh.vertex_faces[v_idx]
                faces = faces[faces != -1]  # 去除无效索引
                face_indices.update(faces.tolist())
        face_indices = np.array([[i] for i in list(face_indices)])
        new_vertices, new_face, _ = subdivide_loop(v, f, face_mask=face_indices)
        ```
    
    
    """
    import trimesh
    current_v = np.asarray(vertices)
    current_f = np.asarray(faces)
    if face_mask is not None:
        face_mask = np.asarray(face_mask).reshape(-1)

    for _ in range(iterations):
        current_v, current_f,face_mask_dict=trimesh.remesh.subdivide(current_v,current_f,face_mask, return_index=True)
        face_mask = np.asarray(np.concatenate(list(face_mask_dict.values()))).reshape(-1)
        # 检查停止条件
        if len(current_f)>max_face_num:
            log.info(f"subdivide: {len(current_f)} >{ max_face_num},break")
            break
        
    return current_v, current_f,face_mask



def angle_axis_np(angle, axis):
    """
    计算绕给定轴旋转指定角度的旋转矩阵。

    Args:
        angle (float): 旋转角度（弧度）。
        axis (np.ndarray): 旋转轴，形状为 (3,) 的 numpy 数组。

    Returns:
        np.array: 3x3 的旋转矩阵，数据类型为 np.float32。
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                        [u[2], 0.0, -u[0]],
                                        [-u[1], u[0], 0.0]])
    R =cosval * np.eye(3)+ sinval * cross_prod_mat+ (1.0 - cosval) * np.outer(u, u)
    return R





def detect_boundary_points(points, labels, config=None):
    """
    基于局部标签一致性的边界点检测函数
    
    Args:
        points (np.ndarray): 点云坐标，形状为 (N, 3)
        labels (np.ndarray): 点云标签，形状为 (N,)
        config (dict): 配置参数，包含:
            - knn_k: KNN查询的邻居数（默认40）
            - bdl_ratio: 边界判定阈值（默认0.8）
            
    Returns:
        np.ndarray: 边界点掩码，形状为 (N,)，边界点为True，非边界点为False
    """
    from sklearn.neighbors import KDTree
    from scipy.stats import mode
    labels = labels.reshape(-1)
    # 设置默认配置
    default_config = {
        "knn_k": 40,
        "bdl_ratio": 0.8
    }
    if config:
        default_config.update(config)
    config = default_config
    
    # 构建KD树
    tree = KDTree(points, leaf_size=2)
    
    # 查询k近邻索引
    near_points_indices = tree.query(points, k=config["knn_k"], return_distance=False)
    
    # 获取邻居标签
    neighbor_labels = np.asarray(labels[near_points_indices], dtype=np.int32)  # 形状: (N, knn_k)
    
    # 统计每个点的邻居中主要标签的出现次数
    # def count_dominant_label(row):
    #     return np.bincount(row).max() if len(row) > 0 else 0
    # label_counts = np.apply_along_axis(count_dominant_label, axis=1, arr=neighbor_labels)
    if neighbor_labels.size == 0:
        label_counts = np.zeros(len(points), dtype=int)
    else:
        label_counts = mode(neighbor_labels, axis=1, keepdims=False).count

    # 计算主要标签比例并生成边界掩码
    label_ratio = label_counts / config["knn_k"]
    boundary_mask = label_ratio < config["bdl_ratio"]
    # print(neighbor_labels.shape,np.unique(neighbor_labels))
    # print(f"标签比例范围: [{label_ratio.min():.2f}, {label_ratio.max():.2f}]")
    # print(f"边界点数量: {boundary_mask.sum()}")
    
    return boundary_mask






def cut_mesh_with_meshlib(v: np.ndarray, f: np.ndarray, loop_points,
                          get_bigger_part: bool = False, smooth_boundary: bool = False) -> tuple:
    """沿指定的点环切割网格并返回选定的部分

    给定的点环投影到网格表面，创建闭合轮廓，沿此轮廓切割网格，
    并返回网格的较大或较小部分。可选择对切割边界进行平滑处理。

    Args:
        v: 输入网格的顶点坐标，形状为 (N, 3)
        f: 输入网格的面索引，形状为 (M, 3)
        loop_points: 定义切割环的3D点列表，每个点为 [x, y, z],，形状为 (B, 3)
        get_bigger_part: 如果为True，返回切割后较大的部分；否则返回较小的部分
        smooth_boundary: 如果为True，对切割边界进行平滑处理

    Returns:
        tuple: 包含:
            kept_mesh_v: 切割后网格的顶点坐标，形状为 (P, 3)
            kept_mesh_f: 切割后网格的面索引，形状为 (Q, 3)
            removed_mesh_v: 其他网格的顶点坐标，形状为 (P, 3)
            removed_mesh_f: 其他网格的面索引，形状为 (Q, 3)

    Raises:
        RuntimeError: 如果切割操作失败或产生无效结果

    Example:
         kept_mesh_v,kept_mesh_f,removed_mesh_v,removed_mesh_f = cut_mesh(vertices, faces, margin_points, get_bigger_part=True, smooth_boundary=True)
    """

    import meshlib.mrmeshnumpy as mrmeshnumpy
    from meshlib.mrmeshpy import (smoothRegionBoundary, edgeCurvMetric,
                                  fillContourLeftByGraphCut, Mesh, Vector3f,
                                  findProjection, convertMeshTriPointsToClosedContour,
                                  cutMesh)
    from importlib.metadata import version
    assert version("meshlib")=='3.0.6.229'
    # 验证输入数组格式
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("顶点数组必须是 (N, 3) 的形状")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("面索引数组必须是 (M, 3) 的形状")
    if len(loop_points) < 3:
        raise ValueError("切割环必须包含至少3个点")

    # 从输入的顶点和面创建网格
    mesh_clone = mrmeshnumpy.meshFromFacesVerts(f, v)

    # 将环上的点投影到网格表面
    tri_points = []
    for p in loop_points:
        v3 = Vector3f(p[0], p[1], p[2])
        projection = findProjection(v3, mesh_clone)
        tri_points.append(projection.mtp)

    # 从投影点创建闭合轮廓
    contour = convertMeshTriPointsToClosedContour(mesh_clone, tri_points)


    # 沿轮廓切割网格
    cut_result = cutMesh(mesh_clone, [contour])

    # 使用图割方法选择网格部分
    edge_path = cut_result.resultCut[0]
    one_part = fillContourLeftByGraphCut(
        mesh_clone.topology,
        edge_path,
        edgeCurvMetric(mesh_clone)
    )

    # 如果需要，平滑边界
    if smooth_boundary:
        smoothRegionBoundary(mesh_clone, one_part)

    # 确定要保留的部分
    other_part = mesh_clone.topology.getValidFaces() - one_part
    # 计算两个部分的面积
    area_one = mesh_clone.area(one_part)
    area_other = mesh_clone.area(other_part)
    # one_part_bool_np =mrmeshnumpy.getNumpyBitSet(one_part)
    # 根据面积选择保留部分
    if get_bigger_part:
        kept_part = one_part if area_one > area_other else other_part
        removed_part = other_part if kept_part == one_part else one_part
    else:
        kept_part = one_part if area_one < area_other else other_part
        removed_part = other_part if kept_part == one_part else one_part

    # 创建输出网格
    kept_mesh = Mesh()
    kept_mesh.addPartByMask(mesh_clone, kept_part)
    # 清理未使用的顶点，优化网格
    kept_mesh.pack()

    # 其他网格
    removed_mesh = Mesh()
    removed_mesh.addPartByMask(mesh_clone, removed_part)
    removed_mesh.pack()

    # 提取numpy数组
    kept_mesh_v = mrmeshnumpy.getNumpyVerts(kept_mesh)
    kept_mesh_f = mrmeshnumpy.getNumpyFaces(kept_mesh.topology)
    removed_mesh_v = mrmeshnumpy.getNumpyVerts(removed_mesh)
    removed_mesh_f = mrmeshnumpy.getNumpyFaces(removed_mesh.topology)

    return kept_mesh_v,kept_mesh_f,removed_mesh_v,removed_mesh_f


def line_project_mesh(v: np.ndarray, f: np.ndarray, loop_points,gen_new_edge=False):
    """
    将输入的3D点环投影到网格表面，根据参数决定是生成新的切割边还是仅提取投影轮廓。

    Args:
        v (np.ndarray): 网格顶点数组，形状为(N, 3)的浮点数组。
        f (np.ndarray): 网格面索引数组，形状为(M, 3)的整数数组。
        loop_points (iterable): 3D点列表/数组，定义要投影到网格上的环状路径。
        gen_new_edge (bool, optional): 是否生成新切割边。默认为True。
            True: 在网格上生成新边并分割网格
            False: 仅提取投影轮廓

    Returns:
        tuple: 包含四个元素的元组:
            - res_pts (list): 投影轮廓的3D点列表，每个点为[x, y, z]
            - res_pts_idx (list): 新生成边的顶点索引列表(仅当gen_new_edge=True时有效)
            - res_v (np.ndarray): 处理后网格顶点数组(仅当gen_new_edge=True时返回新网格)
            - res_f (np.ndarray): 处理后网格面索引数组(仅当gen_new_edge=True时返回新网格)
    """

    import meshlib.mrmeshnumpy as mrmeshnumpy
    from meshlib.mrmeshpy import ( Vector3f, findProjection, convertMeshTriPointsToClosedContour, extractMeshContours,cutMesh,func_float_from_Id_EdgeTag)
    # 验证输入数组格式
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("顶点数组必须是 (N, 3) 的形状")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("面索引数组必须是 (M, 3) 的形状")
    if len(loop_points) < 3:
        raise ValueError("切割环必须包含至少3个点")

    # 从输入的顶点和面创建网格
    mesh_clone = mrmeshnumpy.meshFromFacesVerts(f, v)

    # 将环上的点投影到网格表面
    tri_points = []
    for p in loop_points:
        v3 = Vector3f(p[0], p[1], p[2])
        projection = findProjection(v3, mesh_clone)
        tri_points.append(projection.mtp)



    # 从投影点创建闭合轮廓
    contour = convertMeshTriPointsToClosedContour(mesh_clone, tri_points)

    res_v = v
    res_f = f
    res_pts_idx = []
    res_pts = []

    if gen_new_edge:
        # 基于投影生成新边
        cut_result = cutMesh(mesh_clone, [contour])
        res_v ,res_f =  mrmeshnumpy.getNumpyVerts(mesh_clone), mrmeshnumpy.getNumpyFaces(mesh_clone.topology)
        for i in cut_result.resultCut[0]:
            res_pts_idx.append(int(i))
    else:
        for pts in extractMeshContours([contour])[0]:
            res_pts.append([pts.x,pts.y,pts.z])


    return res_pts,res_pts_idx,res_v ,res_f




def mesh_uv_wrap(mesh):
    """对3D网格进行UV展开处理，使用xatlas库生成优化的UV坐标。

    该函数接收一个trimesh网格或场景对象，将其转换为单个网格后，
    使用xatlas算法进行参数化处理以生成UV坐标，最终返回带有UV信息的网格。

    Args:
        mesh (trimesh.Trimesh or trimesh.Scene or str): 输入的3D网格或mesh路径或场景对象。
            如果是场景对象，会先合并为单个网格。

    Returns:
        trimesh.Trimesh: 带有生成的UV坐标的网格对象，顶点和面可能经过重新索引。

    Raises:
        ValueError: 当输入网格的面数超过500,000,000时抛出，不支持过大的网格处理。

    Note:
        处理过程中会使用xatlas.parametrize()进行UV展开，这可能会重新组织顶点和 faces索引。
        生成的UV坐标会存储在网格的visual.uv属性中，可用于纹理映射等后续处理。
    """
    # 局部导入模块，避免在不需要该功能时加载依赖
    import trimesh
    import xatlas
    # 如果输入是路径，则合并为单个网格
    if isinstance(mesh, str):
        mesh =trimesh.load_mesh(mesh)

    # 如果输入是场景，则合并为单个网格
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # 检查网格面数是否在支持范围内
    if len(mesh.faces) > 500_000_000:  # 使用下划线提高可读性
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    # 使用xatlas进行UV参数化
    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    # 更新网格的顶点、面和UV坐标
    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh


def normal2color(normal):
    """
    将normal转换成颜色
    Args:
        normal: 网格的法线

    Returns:
        (0-255)颜色

    """
    rgb = (normal*0.5 + 0.5)*255.0
    return rgb


def depth2color(depth: np.ndarray,bg_color=None) -> np.ndarray:
    """
    将深度值转换为彩色图像

    Args:
        depth: 深度值

    Returns:
        彩色深度图像，形状为(H, W, 3)
    """
    import matplotlib.colors as mcolors
    from matplotlib import colormaps
    cmap = colormaps['jet']
    norm = mcolors.Normalize(vmin=depth.min(), vmax=depth.max())
    value = norm(depth) # 0~1
    rgb = cmap(value)[:, :, :3]
    if bg_color is not  None:
        # 替换最大值区域为指定颜色
        rgb[value>1-1e-6] = bg_color
    return (rgb * 255).astype(np.uint8)