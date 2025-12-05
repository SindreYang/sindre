
import numpy as np
from numba import njit, prange
from sindre.general.logs import CustomLogger
log = CustomLogger(logger_name="sample").get_logger()
__all__ = ["sample_mesh_sdf",
           "sample_mesh_sharp_edges",
           "sample_mesh_surface",
           "sample_pcd_farthest",
           "sample_pcd_farthest_open3d"]



def sample_mesh_sharp_edges(vertices, faces, num=20000, angle_threshold=10.0):
    """
    从网格的尖锐边缘采样点云（含法线）;

    从网格的尖锐边缘采样点云,支持上采样;（阈值为角度，单位：度）
    https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/f8db63096c8282cb27354314d896feba5ba6ff8a/hy3dgen/shapegen/surface_loaders.py#L40
    Args:
        vertices (array-like): 网格的顶点数组。
        faces (array-like): 网格的面数组。
        num (int): 采样点数，默认16384
        angle_threshold (float): 尖锐度角度阈值（范围0~180度），值越大识别的尖锐边越“钝”，默认10度

    Returns:
        samples (np.ndarray): (num, 3) 采样点坐标
        normals (np.ndarray): (num, 3) 采样点法线

    """
    import trimesh
    # 参数校验：确保角度阈值在合理范围
    if not (0 <= angle_threshold <= 180):
        raise ValueError(f"角度阈值angle_threshold必须在0~180度之间，当前值为{angle_threshold}")
    mesh =trimesh.Trimesh(vertices=vertices, faces=faces)

    # 将角度阈值转换为余弦值（顶点法线与面法线的点积阈值）
    cos_threshold = np.cos(np.radians(angle_threshold))

    # 提取网格基础数据
    V = np.asarray(mesh.vertices)  # 顶点坐标 (Nv, 3)
    N = np.asarray(mesh.face_normals)  # 面法线 (Nf, 3)
    VN = np.asarray(mesh.vertex_normals)  # 顶点法线 (Nv, 3)
    F = np.asarray(mesh.faces)  # 面索引 (Nf, 3)

    # 计算顶点的尖锐度指标：顶点法线与所属面法线的最小点积
    VN2 = np.ones(V.shape[0])  # 初始化尖锐度指标为1（最平滑）
    for i in range(3):  # 遍历面的三个顶点
        face_vertex_idx = F[:, i]  # 当前面的第i个顶点索引
        # 顶点法线与对应面法线的点积
        dot_product = np.sum(VN[face_vertex_idx] * N, axis=-1)
        # 更新顶点的最小点积（保留最尖锐的情况）
        current_vn2 = VN2[face_vertex_idx]
        VN2[face_vertex_idx] = np.minimum(current_vn2, dot_product)

    # 筛选尖锐顶点（点积小于角度对应的余弦值 → 夹角大于阈值）
    sharp_mask = VN2 < cos_threshold

    # 提取所有唯一边（避免重复边影响权重）
    edges = mesh.edges_unique  # trimesh直接提供唯一边 (Ne, 2)
    edge_a, edge_b = edges[:, 0], edges[:, 1]

    # 筛选尖锐边（两个顶点均为尖锐顶点）
    sharp_edge_mask = sharp_mask[edge_a] & sharp_mask[edge_b]
    sharp_edges = edges[sharp_edge_mask]  # 尖锐边索引 (Nse, 2)

    # 处理无尖锐边的情况
    if len(sharp_edges) == 0:
        print(f"警告：在角度阈值{angle_threshold}度下未检测到尖锐边，返回网格表面均匀采样")
        samples, face_idx = mesh.sample(num, return_index=True)
        normals = mesh.face_normals[face_idx]
        return samples, normals

    # 提取尖锐边的顶点坐标和法线
    sharp_verts_a = V[sharp_edges[:, 0]]  # (Nse, 3)
    sharp_verts_b = V[sharp_edges[:, 1]]  # (Nse, 3)
    sharp_verts_an = VN[sharp_edges[:, 0]]  # (Nse, 3)
    sharp_verts_bn = VN[sharp_edges[:, 1]]  # (Nse, 3)

    # 计算每条尖锐边的长度作为采样权重（避免除以0）
    edge_lengths = np.linalg.norm(sharp_verts_b - sharp_verts_a, axis=-1)
    edge_lengths = np.maximum(edge_lengths, 1e-8)  # 防止零长度边
    weights = edge_lengths / np.sum(edge_lengths)  # 归一化权重

    # 按权重随机选择边（防止索引越界）
    random_indices = np.searchsorted(weights.cumsum(), np.random.rand(num))
    random_indices = np.clip(random_indices, 0, len(weights)-1)

    # 在选中的边上线性插值采样
    w = np.random.rand(num, 1)  # 插值权重 (num, 1)
    samples = w * sharp_verts_a[random_indices] + (1 - w) * sharp_verts_b[random_indices]
    normals = w * sharp_verts_an[random_indices] + (1 - w) * sharp_verts_bn[random_indices]

    # 归一化法线（确保单位长度）
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)

    return samples, normals



def sample_mesh_sdf(vertices, faces, number_of_points=200000):
    """
    从网格曲面附近采样SDF点（支持非水密/自相交网格）

    在曲面附近不均匀地采样 SDF 点，该函数适用于非水密网格（带孔的网格）、自相交网格、具有非流形几何体的网格以及具有方向不一致的面的网格。
    这是 DeepSDF 论文中提出和使用的方法。

    Args:
        vertices (array-like): 网格的顶点数组。
        faces (array-like): 网格的面数组。
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

    mesh = trimesh.Trimesh(vertices, faces)

    points, sdf = sample_sdf_near_surface(mesh, number_of_points=number_of_points)
    return points, sdf




def sample_mesh_surface(vertices, faces,normals=None, density=1, num_samples=None):
    """
    网格表面均匀重采样（基于重心坐标）

    在由顶点和面定义的网格表面上进行点云重采样。

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
        normals (numpy.ndarray): 法线数组，形状为(F, 3)/(V, 3)
        density (float, 可选): 每单位面积的采样点数，默认为1
        num_samples (int, 可选): 指定总采样点数，若提供则忽略density参数

    Returns:
        采样点数组, 面索引数组, 顶点索引数组

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
    vertices = np.array(vertices,dtype=np.float64)
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

    # 分配每个面的采样点数
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # 生成采样点对应的面索引
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc : acc + _n_sample] = face_idx
        acc += _n_sample


    # 生成重心坐标随机数
    r = np.random.rand(n_samples, 2)
    faces_samples = faces[sample_face_idx]
    A = vertices[faces_samples[:, 0]]
    B = vertices[faces_samples[:, 1]]
    C = vertices[faces_samples[:, 2]]

    # 使用重心坐标公式计算采样点
    sqrt_r1 = np.sqrt(r[:, 0:1])
    sample_pts = (1 - sqrt_r1) * A + sqrt_r1 * (1 - r[:, 1:]) * B + sqrt_r1 * r[:, 1:] * C



    if normals is not None:
        if len(normals) == len(vertices):
            # 顶点法线
            A_norm = normals[faces_samples[:, 0]]
            B_norm = normals[faces_samples[:, 1]]
            C_norm = normals[faces_samples[:, 2]]
            sample_normals = (
                    (1 - sqrt_r1) * A_norm
                    + sqrt_r1 * (1 - r[:, 1:]) * B_norm
                    + sqrt_r1 * r[:, 1:] * C_norm
            )
            sample_pts_normals = sample_normals / np.linalg.norm(sample_normals, axis=1, keepdims=True)  # 重新归一化
            return  sample_pts, sample_pts_normals
        elif len(normals) == len(faces):
            # 面片法线
            sample_face_normals=normals[sample_face_idx]
            return  sample_pts, sample_face_normals
        else:
            raise ValueError(f"{len(normals)} 长度只能等于顶点/面片长度")
    else:
        return sample_pts






@njit(nogil=True, fastmath=True, cache=True, parallel=True)
def _farthest_pcd_jit(xyz: np.ndarray, offset: np.ndarray, new_offset: np.ndarray) -> np.ndarray:
    """
    最远点采样的Numba并行实现（内部核心函数）

    使用并行批次处理的最远点采样算法实现

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


def sample_pcd_farthest(vertices: np.ndarray, n_sample: int = 2000, auto_seg: bool = False, n_batches: int = 10) -> np.ndarray:
    """
    点云最远点采样（Numba并行实现）

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
    return  _farthest_pcd_jit(xyz,offset,new_offset)


def sample_pcd_farthest_open3d(vertices: np.ndarray, n_sample: int = 2000,device ="CPU:0") -> np.ndarray:
    """
    点云最远点采样（Open3D高效实现）

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