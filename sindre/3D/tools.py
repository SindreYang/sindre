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



try:
    import json
    import trimesh
    import vedo
    import numpy as np
    from typing import Union
    from sklearn.decomposition import PCA

except ImportError:
    raise ImportError("请安装：pip install numpy vedo trimesh scikit-learn")


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
    colormap = [trimesh.visual.color.hex_to_rgba(i) for i in colormap_hex]
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


def apply_pac_transform(vertices: np.array, transform: np.array)->np.array:
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


def save_np_json(output_path: str, obj: any) -> None:
    """
    保存np形式的json

    Args:
        output_path: 保存路径
        obj: 保存对象


    """

    with open(output_path, 'w') as fp:
        json.dump(obj, fp, cls=NpEncoder)
