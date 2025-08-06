"""


专注于牙颌mesh的特殊实现

"""
__author__ = 'sindre'
import vedo
import numpy as np
from typing import *
try:
    from sklearn.decomposition import PCA
except ImportError:
    pass
from sindre.utils3d.algorithm import apply_transform,cut_mesh_point_loop,  subdivide_loop_by_trimesh

def convert_fdi2idx(labels):
    """
    
    将口腔牙列的FDI编号(11-18,21-28/31-38,41-48)转换为(1-16),只支持单颌:
    上颌：
    - 右上(11-18) → 1-8
    - 左上(21-28) → 9-16

    下颌：
    - 左下(31-38) → 1-8
    - 右下(41-48) → 9-16
    - 0或小于0    → 0

                  1 9
                2    10
              3        11
            4            12
          5                13
        6                    14
      7                        15
    8                            16
    """
    labels=np.array(labels)
    if labels.max()>30:
        labels -= 20
    labels[labels//10==1] %= 10
    labels[labels//10==2] = (labels[labels//10==2]%10) + 8
    labels[labels<0] = 0
    return labels




def convert_labels2color(data: Union[np.array, list]) -> list:
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



def transform_crown(near_mesh: vedo.Mesh, jaw_mesh: vedo.Mesh) -> vedo.Mesh:
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
    new_m = vedo.Mesh([apply_transform(near_mesh.points(), transform), near_mesh.faces()])
    return new_m




def cut_mesh_point_loop_crow(mesh,pts,error_show=True,invert=True):
    
    """ 
    
    实现的基于线的牙齿冠分割;

    Args:
        mesh (_type_): 待切割网格
        pts (vedo.Points/Line): 切割线
        error_show(bool, optional): 裁剪失败是否进行渲染. Defaults to True.
        invert(bool): 是否取反；

    Returns:
        _type_: 切割后的网格
    """
    if len(pts.vertices)>30:
        s = int(len(pts.vertices)/30)
        pts = vedo.Points(pts.vertices[::s])
        

    # 计算各区域到曲线的最近距离,去除不相关的联通体
    def batch_closest_dist(vertices, curve_pts):
        curve_matrix = np.array(curve_pts)
        dist_matrix = np.linalg.norm(vertices[:, np.newaxis] - curve_matrix, axis=2)
        return np.sum(dist_matrix, axis=1)
    regions = mesh.split()
    min_dists = [np.min(batch_closest_dist(r.vertices, pts.vertices)) for r in regions]
    mesh =regions[np.argmin(min_dists)]
    
    

    c1 = cut_mesh_point_loop(mesh,pts,invert=not invert)
    c2 = cut_mesh_point_loop(mesh,pts,invert=invert)



    # 可能存在网格错误造成的洞,默认执行补洞
    c1_num = len(c1.fill_holes().split()[0].boundaries().split())
    c2_num = len(c2.fill_holes().split()[0].boundaries().split())

    
    
    # 牙冠只能有一个开口
    if c1_num==1:
        cut_mesh=c1
    elif  c2_num==1:
        cut_mesh=c2
    else:
        print("裁剪失败,请检查分割线,尝试pts[::3]进行采样输入")
        if error_show:
            print(f"边界1:{c1_num},边界2：{c2_num}")
            vedo.show([(c1),(c2)],N=2).close()
        return None
    
    return cut_mesh


def cut_with_ribbon(mesh:vedo.Mesh, pts):
    """
    使用点序列切割网格
    
    参数:
    pts: 切割点序列 (k, 3)
   
    返回:
    new_v: 切割后顶点
    new_f: 切割后面
    """
    mesh.compute_normals()
    vn = np.array(mesh.vertex_normals)
    pns = []
    ksp = vedo.KSpline(pts, closed=True)
    ptsk=ksp.vertices-mesh.center_of_mass()
    v = np.zeros_like(ptsk)
    tol = mesh.diagonal_size()/2
    for i in range(len(pts)):
        iclos = mesh.closest_point(pts[i], return_point_id=True)
        pns.append(vn[iclos])
    for i in range(len(ptsk)-1):
        vi = vedo.cross(ptsk[i],  ptsk[i+1])
        v[i] = vi/vedo.mag(vi)
    vmean = np.mean(v, axis=0)


    
    rib1 = vedo.Ribbon(pts - 0.1*vedo.vector(pns), pts +0.1*vedo.vector(pns))
    rib2 = vedo.Ribbon(ksp.vertices-tol*vmean,ksp.vertices+tol*vmean)
    mesh.cut_with_mesh(rib1)
    mesh.cut_with_mesh(rib2)
    # DEBUG
    # show(mesh, line,rib1.bc('green').alpha(0.5),rib2.bc('bule').alpha(1), axes=1).close()
    return mesh


def subdivide_with_pts(v, f, line_pts, r=0.15, iterations=3, method="mid"):
    """
    对给定的网格和线点集进行局部细分。

    Args:
        v (array-like): 输入网格的顶点数组。
        f (array-like): 输入网格的面数组。
        line_pts (array-like): 线的点集数组。
        r (float, optional): 查找线点附近顶点的半径，默认为 0.15.
        method (str, optional): 细分方法，可选值为 "mid"（中点细分）或其他值（对应 ls3_loop 细分），默认为 "mid"。

    Returns:
        - new_vertices (np.ndarray): 细分后的顶点数组;
        
        - new_face (np.ndarray): 细分后的面数组;

    Notes:
        ```python
        # 闭合线可能在曲面上，曲面内，曲面外
        line = Line(pts)
        mesh = isotropic_remeshing_by_acvd(mesh)
        v, f = np.array(mesh.vertices), np.array(mesh.cells)
        new_vertices, new_face = subdivide_with_pts(v, f, pts)

        show([(Mesh([new_vertices, new_face]).c("green"), Line(pts, lw = 2, c = "red")),
             (Mesh([v, f]).c("pink"), Line(pts, lw = 2, c = "red"))], N = 2).close()
        ```
        
    """
    
    
    from scipy.spatial import cKDTree
    import pymeshlab
    v ,f,pts =np.array(v),np.array(f),np.array(line_pts)
    # 获取每个点的最近表面点及对应面
    face_indices = set()
    kdtree = cKDTree(v)
    vertex_indices = kdtree.query_ball_point(pts, r=r)
    
    #合并所有邻近顶点并去重
    all_vertex_indices = np.unique(np.hstack(vertex_indices)).astype(np.int32)
    face_mask = np.any(np.isin(f, all_vertex_indices), axis=1)
    
    # 局部细分
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f,f_scalar_array=face_mask))
    ms.compute_selection_by_condition_per_face(condselect="fq == 1")
    if method == "mid":
        ms.meshing_surface_subdivision_midpoint(
            iterations=iterations,
            threshold=pymeshlab.PercentageValue(1e-4),
            selected=True  
        )
    else:
        ms.meshing_surface_subdivision_ls3_loop(
            iterations=iterations,
            threshold=pymeshlab.PercentageValue(1e-4),
            selected=True  
        )
    current_mesh = ms.current_mesh()
    new_vertices = current_mesh.vertex_matrix()
    new_faces = current_mesh.face_matrix()
    return new_vertices,new_faces




def deform_and_merge_mesh(
        mesh_path: str,
        base_mesh_path: str,
        ref_direction: np.ndarray = np.array([0, 0, -1]),
        angle_threshold: float = 30,
        boundary_samples: int = 200,
        non_boundary_samples: int = 500,
        boundary_radius: float = 1.0,
        seed: int = 1024,
):
    """
    将网格变形并与基础网格合并,用于将AI生成闭合冠裁剪并拟合到基座上；

    该函数：
        1. 基于参考方向的角度阈值处理输入网格
        2. 识别边界区域
        3. 生成变形控制点
        4. 将边界区域变形以匹配基础网格
        5. 将变形后的网格与基础网格合并


    Notes:

    '''
        # 自定义参考方向向量
        sm = SindreMesh(r"J10177170088_UpperJaw_gen.ply")
        custom_direction = np.array(sm.vertices[42734] - sm.vertices[48221])


        result_mesh = deform_and_merge_mesh(
            mesh_path=r"J10177170088_UpperJaw_gen.ply",
            base_mesh_path=r"17.ply",
            ref_direction=custom_direction,
            angle_threshold=30,
            boundary_samples=200,
            non_boundary_samples=500
        )

        result_mesh.write("merged_result.ply")
        show(result_mesh, axes=1).close()
    '''

    Args:
        mesh_path: 主网格PLY文件路径
        base_mesh_path: 基础网格PLY文件路径
        ref_direction: 参考方向向量 (默认 [0,0,1])
        angle_threshold: 用于面片过滤的角度阈值 (度)
        boundary_samples: 边界点采样数量
        non_boundary_samples: 非边界点采样数量
        boundary_radius: 边界区域识别半径
        seed: 随机种子 (确保结果可重现)

    Returns:

        vedo.Mesh: 合并并清理后的网格
    """
    from vedo import Mesh,  merge
    from scipy.spatial import KDTree
    import numpy as np
    import random
    # 加载网格
    sm = Mesh(mesh_path)
    sm_base = Mesh(base_mesh_path)

    # 设置随机种子确保结果可重现
    random.seed(seed)
    np.random.seed(seed)

    # 1. 基于法线角度阈值过滤面片
    # 归一化参考方向
    ref_normal = ref_direction / (np.linalg.norm(ref_direction) + 1e-8)

    # 获取面法线并归一化
    sm.compute_normals()
    face_normals = sm.celldata["Normals"]
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1)[:, None]

    # 计算面法线与参考方向的夹角余弦值
    cos_angles = np.dot(face_normals, ref_normal)
    angle_rad = np.radians(angle_threshold)

    # 创建面片掩码：角度小于阈值的面片保留
    faces_mask = cos_angles > np.cos(angle_rad)

    # 提取并清理新网格
    new_sm = Mesh([sm.vertices, np.array(sm.cells)[~faces_mask]])
    new_sm = new_sm.extract_largest_region().clean()

    # 2. 提取边界点
    new_sm_boundary = new_sm.boundaries().join(reset=True).vertices
    sm_base_boundary = sm_base.boundaries().join(reset=True).vertices

    # 3. 创建变形控制点
    sources, targets = [], []

    # 为基准网格边界创建KDTree加速搜索
    base_kdtree = KDTree(sm_base_boundary)

    # 采样边界点
    if len(new_sm_boundary) > boundary_samples:
        sampled_indices = random.sample(range(len(new_sm_boundary)), boundary_samples)
    else:
        sampled_indices = range(len(new_sm_boundary))

    for idx in sampled_indices:
        p = new_sm_boundary[idx]
        sources.append(p)
        # 在基础网格边界中找到最近点
        targets.append(sm_base_boundary[base_kdtree.query(p)[1]])

    # 4. 识别边界区域
    cell_centers = new_sm.cell_centers(True).vertices
    kdtree = KDTree(cell_centers)

    # 找到所有边界点半径内的单元
    boundary_cell_ids = set()
    for p in new_sm_boundary:
        cell_ids = kdtree.query_ball_point(p, boundary_radius)
        boundary_cell_ids.update(cell_ids)

    boundary_cell_ids = np.array(list(boundary_cell_ids))
    non_boundary_cell_ids = np.setdiff1d(np.arange(new_sm.ncells), boundary_cell_ids)

    # 提取非边界区域顶点
    non_boundary_region = new_sm.extract_cells(non_boundary_cell_ids)
    non_boundary_vertices = non_boundary_region.vertices

    # 添加非边界固定点 (保持形状)
    if len(non_boundary_vertices) > non_boundary_samples:
        sampled_vertices = non_boundary_vertices[
            np.random.choice(len(non_boundary_vertices), non_boundary_samples, replace=False)
        ]
    else:
        sampled_vertices = non_boundary_vertices

    sources.extend(sampled_vertices.tolist())
    targets.extend(sampled_vertices.tolist())

    # 5. 使用控制点变形整个网格
    warped_mesh = new_sm.clone().warp(sources, targets)

    # 6. 与基础网格合并并清理
    merged = merge(warped_mesh, sm_base)
    merged.smooth(niter=10,boundary=True)  # 平滑融合边界
    merged.clean()  # 清理无效元素

    return merged





def cut_crown_with_meshlib(mesh: vedo.Mesh, margin_points: np.ndarray) -> Tuple[vedo.Mesh, vedo.Mesh]:
    """使用边缘点分割牙冠网格模型，返回保留部分和移除部分。

    该函数通过以下步骤实现牙冠分割：
    1. 在输入网格中定位距离边缘线最近的连通区域
    2. 使用边缘线切割该区域
    3. 根据边界距离验证切割结果
    4. 合并剩余网格组件

    参数：
        mesh: vedo.Mesh对象，表示待分割的牙冠网格模型
        margin_points: np.ndarray数组，形状为(N,3)的边缘点集

    返回：
        Tuple[vedo.Mesh, vedo.Mesh]:
            第一个Mesh为保留部分（牙冠主体）
            第二个Mesh为移除部分（牙龈区域）

    异常：
        当边缘点与网格的最小距离超过1mm时触发断言错误
    """
    from scipy.spatial import KDTree
    from vedo import Mesh, merge
    import numpy as np
    from sindre.utils3d.algorithm import cut_mesh_with_meshlib

    # 建立边缘点的KDTree用于空间搜索
    margin_tree = KDTree(margin_points)
    dist_min = float("inf")  # 初始化最小距离为无穷大
    mesh_min = None  # 存储距离边缘最近的网格组件
    mesh_idx = None  # 存储目标组件的索引
    get_bigger_part = False  # 标记是否需要保留较大区域

    # 分割网格为连通组件
    split_mesh = mesh.split()

    # 遍历所有连通组件寻找目标切割区域
    for i, mesh_i in enumerate(split_mesh):
        npts = mesh_i.npoints
        # 仅处理足够大的组件（避免小碎片）
        if npts > 1000:
            # 计算组件顶点到边缘点的最小距离
            distances, _ = margin_tree.query(mesh_i.vertices)
            min_component_dist = np.min(distances)

            # 更新最近组件信息
            if min_component_dist < dist_min:
                dist_min = min_component_dist
                mesh_min = mesh_i
                mesh_idx = i
                # 如果组件小于原始网格一半，标记为拼合网格
                if npts < mesh.npoints * 0.5:
                    get_bigger_part = True

    # 验证边缘点与网格的贴合程度
    assert dist_min < 1, f"边缘线与网格距离过大({dist_min:.2f}mm)，请检查输入数据"

    # 使用meshlib进行网格切割（假设已实现）
    kept_mesh_v, kept_mesh_f, removed_mesh_v, removed_mesh_f = cut_mesh_with_meshlib(
        mesh_min.vertices,
        np.array(mesh_min.cells),
        margin_points,
        get_bigger_part=get_bigger_part
    )

    # 构建切割后的网格对象
    keep_mesh = Mesh([kept_mesh_v, kept_mesh_f])
    remove_mesh = Mesh([removed_mesh_v, removed_mesh_f])


    # 验证切割结果方向正确性

    # 法线是否一致
    mesh.compute_normals()
    keep_mesh.compute_normals()
    remove_mesh.compute_normals()
    main_normal = np.mean(mesh.vertex_normals, axis=0)
    # 计算平均法线与主流方向的点积（范围[-1,1]，1为同向，-1为反向）
    keep_mesh_normal_product =sum( np.dot(keep_mesh.vertex_normals, main_normal))
    remove_mesh_normal_product = sum(np.dot(remove_mesh.vertex_normals, main_normal))
    print(keep_mesh_normal_product,remove_mesh_normal_product)
    if keep_mesh_normal_product>0 and remove_mesh_normal_product>0:
        # 已分模检测，计算边界到边缘点的最大距离
        keep_mesh_boundary = keep_mesh.boundaries().extract_largest_region()
        keep_boundary_dists, _ = margin_tree.query(keep_mesh_boundary.vertices)
        max_keep_dist = np.max(keep_boundary_dists)
        remove_mesh_mesh_boundary = remove_mesh.boundaries().extract_largest_region()
        remove_dists, _ = margin_tree.query(remove_mesh_mesh_boundary.vertices)
        max_remove_dist = np.max(remove_dists)
        print(max_remove_dist,max_keep_dist)
        if max_remove_dist < max_keep_dist:
            keep_mesh, remove_mesh = Mesh([removed_mesh_v, removed_mesh_f]), Mesh([kept_mesh_v, kept_mesh_f])

    elif remove_mesh_normal_product>0 and keep_mesh_normal_product<0:
        keep_mesh, remove_mesh = Mesh([removed_mesh_v, removed_mesh_f]), Mesh([kept_mesh_v, kept_mesh_f])
    elif remove_mesh_normal_product<0 and keep_mesh_normal_product>0:
        pass
    else:
        print(f"无法判断裁剪区域:{remove_mesh_normal_product},{keep_mesh_normal_product}")

    # 合并其他组件与移除部分
    other_components = [split_mesh[i] for i in range(len(split_mesh)) if i != mesh_idx]
    other_mesh = merge(other_components + [remove_mesh])

    return keep_mesh, other_mesh