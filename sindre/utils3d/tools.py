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

                                                    __----~~~~~~~~~~~------___
                                   .  .   ~~//====......          __--~ ~~         江城子 . 程序员之歌
                   -.            \_|//     |||\\  ~~~~~~::::... /~
                ___-==_       _-~o~  \/    |||  \\            _/~~-           十年生死两茫茫，写程序，到天亮。
        __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~                    千行代码，Bug何处藏。
    _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /                   纵使上线又怎样，朝令改，夕断肠。
  .~       .~       |   \\ -_    /  /-   /   ||      \   /
 /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /                     领导每天新想法，天天改，日日忙。
 |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\                          相顾无言，惟有泪千行。
          '         ~-|      /|    |-~\~~       __--~~                        每晚灯火阑珊处，夜难寐，加班狂。
                      |-~~-_/ |    |   ~\_   _-~            /\
                           /  \     \__   \/~                \__
                       _--~ _/ | .-~~____--~-/                  ~~==.
                      ((->/~   '.|||' -_|    ~~-/ ,              . _||
                                 -_     ~\      ~~---l__i__i__i--~~_/
                                 _-~-__   ~)  \--______________--~~
                               //.-~~~-~_--~- |-------~~~~~~~~
                                      //.-~~~--\
                              神兽保佑                                 永无BUG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
__author__ = 'sindre'
import json
import vedo
import numpy as np
from typing import *
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree
from scipy.linalg import eigh
import vtk
import pymeshlab


def labels2colors(labels:np.array):
    """
    将labels转换成颜色标签
    Args:
        labels: numpy类型,形状(N)对应顶点的标签；

    Returns:
        RGBA颜色标签;
    """

    colormap = [
        [230, 25, 75, 255],
        [60, 180, 75, 255],
        [255, 225, 25, 255],
        [67, 99, 216, 255],
        [245, 130, 49, 255],
        [66, 212, 244, 255],
        [240, 50, 230, 255],
        [250, 190, 212, 255],
        [70, 153, 144, 255],
        [220, 190, 255, 255],
        [154, 99, 36, 255],
        [255, 250, 200, 255],
        [128, 0, 0, 255],
        [170, 255, 195, 255],
        [0, 0, 117, 255],
        [169, 169, 169, 255],
        [255, 255, 255, 255],
        [0, 0, 0, 255],
        [255, 10, 0, 255],
        [0, 255, 10, 255],
        [0, 9, 255, 255],

    ]

    color_labels= np.zeros((len(labels),4))
    for i in np.unique(labels):
        color = colormap[int(i) % len(colormap)]
        idx_i = np.argwhere(labels == i).reshape(-1)
        color_labels[idx_i] = color

    return color_labels


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
        vertices: 牙颌三角网格
        faces: 面片标签
        face_labels: 顶点标签

    Returns:
        顶点属性

    """

    # 获取三角网格的顶点标签
    vertex_labels = np.zeros(len(vertices))
    for i in range(len(faces)):
        for vertex_id in faces[i]:
            vertex_labels[vertex_id] = face_labels[i]

    return vertex_labels.astype(np.int32)


def tooth_labels_to_color(data: Union[np.array, list]) -> list:
    """
        将牙齿标签转换成RGBA颜色

    Notes:
        只支持以下标签类型：

            upper_dict = [0, 18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]

            lower_dict = [0, 48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

    Args:
        data: 属性

    Returns:
        colors: 对应属性的RGBA类型颜色

    """

    colormap_hex = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#42d4f4', '#f032e6', '#fabed4', '#469990',
                    '#dcbeff',
                    '#9A6324', '#fffac8', '#800000', '#aaffc3', '#000075', '#a9a9a9', '#ffffff', '#000000'
                    ]
    hex2rgb= lambda h: list(int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    colormap = [ hex2rgb(h) for h in colormap_hex]
    upper_dict = [0, 18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    lower_dict = [0, 48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]

    if max(data) in upper_dict:
        colors = [colormap[upper_dict.index(data[i])] for i in range(len(data))]
    else:
        colors = [colormap[lower_dict.index(data[i])] for i in range(len(data))]
    return colors


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
    vedo_mesh = mesh.clone().decimate(n=5000).clean()
    vertices = vedo_mesh.points()

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


def apply_pac_transform(vertices: np.array, transform: np.array) -> np.array:
    """
        对pca获得4*4矩阵进行应用

    Args:
        vertices: 顶点
        transform: 4*4 矩阵

    Returns:
        变换后的顶点

    """

    # 在每个顶点的末尾添加一个维度为1的数组，以便进行齐次坐标转换
    vertices = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=-1)
    vertices = vertices @ transform.T
    # 移除结果中多余的维度，只保留前3列，即三维坐标
    vertices = vertices[..., :3]

    return vertices


def restore_pca_transform(vertices: np.array, transform: np.array) -> np.array:
    """
        根据提供的顶点及矩阵，进行逆变换(还原应用矩阵之前的状态）

    Args:
        vertices: 顶点
        transform: 4*4变换矩阵

    Returns:
        还原后的顶点坐标

    """
    # 得到转换矩阵的逆矩阵
    inv_transform = np.linalg.inv(transform.T)

    # 将经过转换后的顶点坐标乘以逆矩阵
    vertices_restored = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=-1) @ inv_transform

    # 去除齐次坐标
    vertices_restored = vertices_restored[:, :3]

    # 最终得到还原后的顶点坐标 vertices_restored
    return vertices_restored


def rotation_crown(near_mesh: vedo.Mesh, jaw_mesh: vedo.Mesh) -> vedo.Mesh:
    """
        调整单冠的轴向

    Tip:
        1.通过连通域分割两个邻牙;

        2.以邻牙质心为确定x轴；

        3.通过找对颌最近的点确定z轴方向;如果z轴方向上有mesh，则保持原样，否则将z轴取反向;

        4.输出调整后的牙冠


    Args:
        near_mesh: 两个邻牙组成的mesh
        jaw_mesh: 两个邻牙的对颌

    Returns:
        变换后的单冠mesh

    """
    vertices = near_mesh.points()
    # 通过左右邻牙中心指定x轴
    m_list = near_mesh.split()
    center_vec = m_list[0].center_of_mass() - m_list[1].center_of_mass()
    user_xaxis = center_vec / np.linalg.norm(center_vec)

    # 通过找对颌最近的点确定z轴方向
    jaw_mesh = jaw_mesh.split()[0]
    jaw_near_point = jaw_mesh.closest_point(vertices.mean(0))
    jaw_vec = jaw_near_point - vertices.mean(0)
    user_zaxis = jaw_vec / np.linalg.norm(jaw_vec)

    components = PCA(n_components=3).fit(vertices).components_
    xaxis, yaxis, zaxis = components

    # debug
    # arrow_user_zaxis = vedo.Arrow(vertices.mean(0), user_zaxis*5+vertices.mean(0), c="blue")
    # arrow_zaxis = vedo.Arrow(vertices.mean(0), zaxis*5+vertices.mean(0), c="red")
    # arrow_xaxis = vedo.Arrow(vertices.mean(0), user_xaxis*5+vertices.mean(0), c="green")
    # vedo.show([arrow_user_zaxis,arrow_zaxis,arrow_xaxis,jaw_mesh.split()[0], vedo.Point(jaw_near_point,r=12,c="black"),vedo.Point(vertices.mean(0),r=20,c="red5"),vedo.Point(m_list[0].center_of_mass(),r=24,c="green"),vedo.Point(m_list[1].center_of_mass(),r=24,c="green"),near_mesh], axes=3)
    # print(np.dot(user_zaxis, zaxis))

    if np.dot(user_zaxis, zaxis) < 0:
        # 如果z轴方向上有mesh，则保持原样，否则将z轴取反向
        zaxis = -zaxis
    yaxis = np.cross(user_xaxis, zaxis)
    components = np.stack([user_xaxis, yaxis, zaxis], axis=0)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = components
    transform[:3, 3] = - components @ vertices.mean(0)

    # 渲染
    new_m = vedo.Mesh([apply_pac_transform(near_mesh.points(), transform), near_mesh.faces()])
    return new_m


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
    print(center)

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
    # print(verts.shape)
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
        invert (bool, optional): 选择保留最大/最小模式. Defaults to False.

    Returns:
        _type_: 切割后的网格
    """
    
    # 去除不相关的联通体
    regions = mesh.split()
    
    def batch_closest_dist(vertices, curve_pts):
        # 将曲线点集转为矩阵（n×3）
        curve_matrix = np.array(curve_pts)
        # 计算顶点到曲线点的所有距离（矩阵运算）
        dist_matrix = np.linalg.norm(vertices[:, np.newaxis] - curve_matrix, axis=2)
        return np.min(dist_matrix, axis=1)

    # 计算各区域到曲线的最近距离
    min_dists = [np.min(batch_closest_dist(r.vertices, pts.vertices)) for r in regions]
    mesh = regions[np.argmin(min_dists)]
    
    # 切割网格并设置EdgeSearchMode
    selector = vtk.vtkSelectPolyData()
    selector.SetInputData(mesh.dataset)  # 直接获取VTK数据
    selector.SetLoop(pts.dataset.GetPoints())
    selector.GenerateSelectionScalarsOff()
    selector.SetEdgeSearchModeToDijkstra()  # 设置搜索模式
    if invert:
        selector.SetSelectionModeToLargestRegion()
    selector.SetSelectionModeToSmallestRegion()
    selector.Update()
    
    cut_mesh = vedo.Mesh(selector.GetOutput())
    return cut_mesh


def cut_mesh_point_loop_crow(mesh,pts):
    
    """ 
    
    基于vtk+dijkstra实现的基于线的牙齿冠分割;
    
    线支持在网格上或者网格外；

    Args:
        mesh (_type_): 待切割网格
        pts (vedo.Points): 切割线
        invert (bool, optional): 选择保留最大/最小模式. Defaults to False.

    Returns:
        _type_: 切割后的网格
    """
    # 去除不相关的联通体
    regions = mesh.split()
    
    def batch_closest_dist(vertices, curve_pts):
        # 将曲线点集转为矩阵（n×3）
        curve_matrix = np.array(curve_pts)
        # 计算顶点到曲线点的所有距离（矩阵运算）
        dist_matrix = np.linalg.norm(vertices[:, np.newaxis] - curve_matrix, axis=2)
        return np.min(dist_matrix, axis=1)

    # 计算各区域到曲线的最近距离
    min_dists = [np.min(batch_closest_dist(r.vertices, pts.vertices)) for r in regions]
    mesh =regions[np.argmin(min_dists)]
    
    
    # 切割网格并设置EdgeSearchMode
    selector = vtk.vtkSelectPolyData()
    selector.SetInputData(mesh.dataset)  
    selector.SetLoop(pts.dataset.GetPoints())
    selector.GenerateSelectionScalarsOff()
    selector.SetEdgeSearchModeToDijkstra()  # 设置搜索模式
    if np.min(min_dists)<0.1:
        print("mesh已经被裁剪")
        selector.SetSelectionModeToClosestPointRegion()
    else:
        selector.SetSelectionModeToSmallestRegion()
    selector.Update()
    cut_mesh = vedo.vedo.Mesh(selector.GetOutput()).clean()
    return cut_mesh


def reduce_face_by_meshlab(mesh: vedo.Mesh, max_facenum: int = 200000) ->vedo.Mesh:
    """通过二次边折叠算法减少网格中的面数，简化模型。

    Args:
        mesh (pymeshlab.MeshSet): 输入的网格模型。
        max_facenum (int, optional): 简化后的目标最大面数，默认为 200000。

    Returns:
        pymeshlab.MeshSet: 简化后的网格模型。
    """
    import pymeshlab
    
    ms =vedo.vedo2meshlab(ms)
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
    return vedo.meshlab2vedo(ms)


def remove_floater_by_meshlab(mesh: vedo.Mesh) -> vedo.Mesh:
    """移除网格中的浮动小组件（小面积不连通部分）。

    Args:
        mesh (pymeshlab.MeshSet): 输入的网格模型。

    Returns:
        pymeshlab.MeshSet: 移除浮动小组件后的网格模型。
    """
    import pymeshlab
    
    ms =vedo.vedo2meshlab(ms)
    mesh.apply_filter("compute_selection_by_small_disconnected_components_per_face",
                      nbfaceratio=0.005)
    mesh.apply_filter("compute_selection_transfer_face_to_vertex", inclusive=False)
    mesh.apply_filter("meshing_remove_selected_vertices_and_faces")
    return vedo.meshlab2vedo(ms)


def isotropic_remeshing_pymeshlab(mesh: vedo.Mesh, target_edge_length, iterations=10)-> vedo.Mesh:
    """
    使用 PyMeshLab 实现网格均匀化。

    Args:
        mesh: 输入的网格对象 (pymeshlab.MeshSet)。
        target_edge_length: 目标边长。
        iterations: 迭代次数，默认为 10。

    Returns:
        均匀化后的网格对象。
    """
    
    import pymeshlab
    
    ms =vedo.vedo2meshlab(ms)
    # 应用 Isotropic Remeshing 过滤器
    mesh.apply_filter(
        "meshing_isotropic_explicit_remeshing",
        targetlen=pymeshlab.AbsoluteValue(target_edge_length),
        iterations=iterations,
        preserveboundary=True,
        preservenormal=True
    )

    # 返回处理后的网格
    return vedo.meshlab2vedo(ms)



def optimize_mesh_by_meshlab(ms: vedo.Mesh)-> vedo.Mesh:
    
    """
    使用 PyMeshLab 实现一键优化网格。
    
    ```
    
    Merge Close Vertices：合并临近顶点
    Merge Wedge Texture Coord：合并楔形纹理坐标
    Remove Duplicate Faces：移除重复面
    Remove Duplicate Vertices：移除重复顶点
    Remove Isolated Folded Faces by Edge Flip：通过边翻转移除孤立的折叠面
    Remove Isolated pieces (wrt diameter)：移除孤立部分（相对于直径）
    Remove Isolated pieces (wrt Face Num.)：移除孤立部分（相对于面数）
    Remove T-Vertices：移除 T 型顶点
    Remove Unreferenced Vertices：移除未引用的顶点
    Remove Vertices wrt Quality：根据质量移除顶点
    Remove Zero Area Faces：移除零面积面
    Repair non Manifold Edges：修复非流形边
    Repair non Manifold Vertices by splitting：通过拆分修复非流形顶点
    Snap Mismatched Borders ：对齐不匹配的边界 
    
    
    ```
    
    
    
    

    Args:
        mesh: 输入的网格对象 (pymeshlab.MeshSet)。

    Returns:
        优化后的网格对象。
    """
    import pymeshlab
    
    ms =vedo.vedo2meshlab(ms)
    # 1. 合并临近顶点
    ms.apply_filter("meshing_merge_close_vertices", threshold=pymeshlab.AbsoluteValue(0.001))

    # 2. 合并楔形纹理坐标
    ms.apply_filter("apply_texcoord_merge_per_wedge")  

    # 3. 移除重复面
    ms.apply_filter("meshing_remove_duplicate_faces")

    # 4. 移除重复顶点
    ms.apply_filter("meshing_remove_duplicate_vertices")

    # 5. 通过边翻转移除孤立的折叠面
    ms.apply_filter("meshing_remove_folded_faces")  

    # 6. 移除孤立部分（基于直径）
    ms.apply_filter("meshing_remove_connected_component_by_diameter")

    # 7. 移除孤立部分（基于面数）
    ms.apply_filter("meshing_remove_connected_component_by_face_number", mincomponentsize=10)

    # 8. 移除 T 型顶点（文档中无直接对应过滤器）
    ms.apply_filter("meshing_remove_t_vertices") 

    # 9. 移除未引用的顶点
    ms.apply_filter("meshing_remove_unreferenced_vertices")

    # 10. 根据质量移除顶点（需自定义质量阈值）
    ms.apply_filter("meshing_remove_vertices_by_scalar", maxqualitythr=0.5)

    # 11. 移除零面积面
    ms.apply_filter("meshing_remove_null_faces")

    # 12. 修复非流形边
    ms.apply_filter("meshing_repair_non_manifold_edges")

    # 13. 通过拆分修复非流形顶点
    ms.apply_filter("meshing_repair_non_manifold_verticess")

    # 15. 对齐不匹配的边界
    ms.apply_filter("meshing_snap_mismatched_borders", threshold=pymeshlab.AbsoluteValue(0.001))
    
    
    return vedo.meshlab2vedo(ms)





