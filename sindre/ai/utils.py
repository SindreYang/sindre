import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Optional, Dict
from sindre.general.logs import CustomLogger
log= CustomLogger(logger_name="ai_utils").get_logger()

def set_global_seeds(seed: int = 1024,cudnn_enable: bool = False) -> None:
    """
    设置全局随机种子，确保Python、NumPy、PyTorch等环境的随机数生成器同步，提升实验可复现性。

    Args:
        seed (int): 要使用的基础随机数种子，默认值为1024。
        cudnn_enable (bool): 是否将CuDNN设置为确定性模式，启用后可能会影响性能但提高可复现性，默认值为False。
    """
    # 设置Python内置的随机数生成器
    import random,os
    random.seed(seed)
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置NumPy的随机数生成器
    np.random.seed(seed)
    # 尝试设置PyTorch的随机数生成器
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if cudnn_enable:
                # 控制CuDNN的确定性和性能之间的平衡
                torch.backends.cudnn.deterministic = True
                # 禁用CuDNN的自动寻找最优算法
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    log.info(f"全局随机种子已设置为 {seed} | CuDNN确定性模式: {'启用' if cudnn_enable else '禁用'}")


def save_checkpoint(
        path,
        network: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: float,
        curr_iter: int,
        multi_gpu: bool = False,
        extra_info: Optional[Dict] = None
) -> None:
    """
    保存模型状态、优化器状态、当前迭代次数和损失值

    Args:
        path: 包含模型保存路径等参数的配置对象
        network: 神经网络模型
        optimizer: 优化器
        loss: 当前损失值
        curr_iter: 当前迭代次数
        multi_gpu: 是否使用多GPU训练
        extra_info: 可选的额外信息字典，用于保存其他需要的信息
    """
    try:


        # 获取已保存模型loss




        # 获取模型状态字典
        if multi_gpu and torch.cuda.device_count() > 1:
            net_dict = network.module.state_dict()
        else:
            net_dict = network.state_dict()

        # 创建保存字典
        save_dict = {
            "state_dict": net_dict,
            "optimizer": optimizer.state_dict(),
            "curr_iter": curr_iter,
            "loss": loss,
        }

        # 添加额外信息
        if extra_info is not None:
            save_dict.update(extra_info)
        # 确保保存目录存在
        save_dir = os.path.dirname(path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # 保存模型
        torch.save(save_dict, path)
        log.info(f"Save model path: {path},损失: {loss}, epoch: {curr_iter}")

    except Exception as e:
        log.error(f"Failed to save model: {str(e)}", exc_info=True)
        raise

def load_checkpoint(
        path: str,
        net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        load_optimizer: bool = True,
        strict: bool = True,
        check_shape: bool = True,
        map_location: Optional[str] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int, float, Dict]:
    """
    加载模型状态，可以支持部分参数加载

    加载策略:
    - strict==True: 只有名称和形状完全一致的参数才会被加载；
    - strict==False且check_shape==True: 仅加载名称存在且形状匹配的参数；
    - strict==False且check_shape==False: 加载所有名称匹配的参数，不检查形状；

    Args:
        path: 模型文件路径
        net: 要加载参数的神经网络模型
        optimizer: 优化器，如果需要加载优化器状态
        load_optimizer: 是否加载优化器状态
        strict: 是否严格匹配模型参数
        check_shape: 是否检查参数形状匹配
        map_location: 指定设备映射，例如"cpu"或"cuda:0"

    Returns:
        加载了参数的模型、优化器、最后迭代次数、最后损失值、额外信息字典
    """
    try:
        # 检查模型文件是否存在
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        # 加载模型数据
        log.info(f"加载模型: {path}")
        checkpoint = torch.load(path, map_location=map_location)
        model_state, checkpoint_state = net.state_dict(), checkpoint["state_dict"]

        # 处理参数匹配
        if check_shape and not strict:
            filtered = {}
            for k in checkpoint_state:
                if k in model_state and checkpoint_state[k].shape == model_state[k].shape:
                    filtered[k] = checkpoint_state[k]
                elif k in model_state:
                    log.warning(f"参数形状不匹配，跳过: {k} "
                                f"({checkpoint_state[k].shape} vs {model_state[k].shape})")
            net.load_state_dict(filtered, strict=False)
        else:
            net.load_state_dict(checkpoint_state, strict=strict)

        # 加载优化器状态
        if optimizer is not None and load_optimizer:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                log.info("优化器状态已加载")
        # 获取额外信息
        curr_iter = checkpoint.get("curr_iter", 0)
        loss = checkpoint.get("loss", 0.0)
        known_keys = {"state_dict", "optimizer", "curr_iter", "loss"}
        extra_info = {k: v for k, v in checkpoint.items() if k not in known_keys}
        log.info(f"模型加载完成，最后迭代次数: {curr_iter}, 最后损失值: {loss:.6f}")
        return net, optimizer, curr_iter, loss, extra_info
    except Exception as e:
        log.error(f"加载模型失败: {str(e)}", exc_info=True)
        raise






















def square_distance(src, dst):
    """
    计算每两个点之间的欧几里得距离。

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    根据索引从点云中提取点
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    从点云中采样npoint个最远点
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    从点云中查询每个查询点指定半径范围内的点，并返回固定数量的采样点索引

    对于每个查询点，该函数会找出所有在指定半径范围内的原始点，
    如果找到的点数量少于nsample，则用第一个找到的点进行填充。

    Args:
        radius (float): 局部区域的半径阈值
        nsample (int): 每个局部区域内的最大采样点数
        xyz (torch.Tensor): 所有原始点的坐标，形状为 [B, N, 3]
            B: 批次大小，N: 原始点数量，3: xyz坐标
        new_xyz (torch.Tensor): 查询点的坐标，形状为 [B, S, 3]
            S: 查询点数量

    Returns:
        torch.Tensor: 分组后的点索引，形状为 [B, S, nsample]
            每个查询点对应nsample个在半径范围内的点索引，
            不足时用第一个有效点索引填充

    Raises:
        无显式抛出，但如果输入维度不匹配或设备错误会有异常信息打印
    """
    try:
        # 获取设备信息，确保所有操作在同一设备上进行
        device = xyz.device
        B, N, C = xyz.shape  # 解析批次大小、点数量和坐标维度
        _, S, _ = new_xyz.shape  # 解析查询点数量
        # 初始化索引矩阵，每个查询点对应所有原始点的索引
        group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        # 计算查询点与所有原始点之间的平方距离
        sqrdists = square_distance(new_xyz, xyz)
        # 将距离大于半径平方的点索引标记为N（超出原始点范围的无效值）
        group_idx[sqrdists > radius **2] = N
        # 对索引按距离排序并取前nsample个点
        group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
        # 生成填充掩码：用每个查询点的第一个有效点索引填充无效值
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
        mask = group_idx == N
        group_idx[mask] = group_first[mask]

    except Exception as e:
        print(f"查询球点过程中发生错误: {e}")

    return group_idx

def query_knn(nsample, xyz, new_xyz, include_self=True):
    """Find k-NN of new_xyz in xyz"""
    pad = 0 if include_self else 1
    sqrdists = square_distance(new_xyz, xyz)  # B, S, N
    sorted_dist, indices = torch.sort(sqrdists, dim=-1, descending=False)
    idx = indices[:, :, pad: nsample+pad]
    #sdist = sorted_dist[:,:,pad: nsample+pad]
    return idx.int()



def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    对点云进行采样并分组，通过最远点采样(FPS)选择中心点，
    然后对每个中心点在指定半径范围内进行球查询，形成局部区域分组。

    该函数首先从输入点云中采样出npoint个中心点，然后为每个中心点
    查找其半径范围内的nsample个邻近点，最后将这些局部区域的坐标和特征
    进行组合和归一化处理。

    Args:
        npoint (int): 要采样的中心点数量
        radius (float): 球查询的半径范围
        nsample (int): 每个局部区域内最多采样的点数量
        xyz (torch.Tensor): 输入点云的坐标数据，形状为 [B, N, 3]
            B: 批次大小，N: 输入点的总数，3: xyz坐标维度
        points (torch.Tensor or None): 输入点云的特征数据，形状为 [B, N, D]
            D: 每个点的特征维度，如果为None则只使用坐标信息
        returnfps (bool, optional): 是否返回最远点采样的索引，默认为False

    Returns:
        根据returnfps参数不同，返回不同的结果组合：
        - 当returnfps=False时：
            new_xyz (torch.Tensor): 采样出的中心点坐标，形状为 [B, npoint, 3]
            new_points (torch.Tensor): 分组后的点特征（包含归一化坐标），
                形状为 [B, npoint, nsample, 3+D]（若points不为None）或 [B, npoint, nsample, 3]（若points为None）
        - 当returnfps=True时：
            new_xyz, new_points, grouped_xyz, fps_idx: 包含原始分组坐标和FPS索引

    依赖函数:
        farthest_point_sample: 用于从点云中采样最远点
        query_ball_point: 用于查询每个中心点半径范围内的点
        index_points: 用于根据索引从点云中提取点
    """
    B, N, C = xyz.shape  # 解析批次大小、点数量和坐标维度
    S = npoint  # 采样中心点数量
    # 使用最远点采样选择中心点
    fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
    # 对每个中心点进行球查询，获取局部区域内的点索引
    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
    # 根据索引提取局部区域内的点坐标
    grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]
    # 计算局部坐标相对于中心点的偏移（归一化）
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, nsample, 3]
    # 组合局部坐标偏移和特征
    if points is not None:
        # 提取局部区域内的点特征
        grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
        # 拼接归一化坐标和特征
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
    else:
        # 如果没有特征，仅使用归一化坐标
        new_points = grouped_xyz_norm  # [B, npoint, nsample, 3]
    # 根据需要返回额外信息
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points



def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points





def pca_with_svd(data,eps=1e-6):
    """PCA预测旋转正交矩阵 
    `
    def pca_with_svd(data, n_components=3):
        # 数据中心化
        mean = torch.mean(data, dim=0)
        centered_data = data - mean
        # 执行 SVD
        _, _, v = torch.linalg.svd(centered_data, full_matrices=False)
        # 提取前 n_components 个主成分
        components = v[:n_components]
        return components
    `

    """
    identity = torch.eye(data.size(-1), device=data.device) * eps 
    cov = torch.matmul(data.transpose(-2, -1), data) / (data.size(-2) - 1)
    cov_reg = cov + identity
    _, _, v = torch.linalg.svd(cov_reg, full_matrices=False)
    rotation = v.transpose(1,2)  
    det = torch.det(rotation)    # 确保右手坐标系
    new_last_column = rotation[:, :, -1] * det.unsqueeze(-1)
    rotation = torch.cat([rotation[:, :, :-1], new_last_column.unsqueeze(-1)], dim=-1)
    return rotation



def sdf2mesh_by_diso(sdf,diffdmc=None ,deform=None,return_quads=False, normalize=True,isovalue=0 ,invert=True):
    """用pytorch方式给，sdf 转换成 mesh"""
    device = sdf.device
    try:
        from diso import DiffDMC
    except ImportError:
        print("请安装 pip install diso")
    if diffdmc is None:
        diffdmc =DiffDMC(dtype=torch.float32).to(device)
    if invert:
        sdf*=-1
    v, f = diffdmc(sdf, deform, return_quads=return_quads, normalize=normalize, isovalue=isovalue) 
    return v,f



def occ2mesh_by_pytorch3d(occ,isovalue=0 ):
    """用pytorch3d方式给，sdf 转换成 mesh"""
    from pytorch3d.ops import cubify
    from pytorch3d.structures import Meshes
    meshes = cubify(occ, isovalue)
    return meshes





def detect_boundary(points, labels, config=None):
    """
    基于局部标签一致性的边界点检测函数（PyTorch版本）

    Args:
        points (torch.Tensor): 点云坐标，形状为 (N, 3)
        labels (torch.Tensor): 点云标签，形状为 (N,)
        config (dict): 配置参数，包含:
            - knn_k: KNN查询的邻居数（默认40）
            - bdl_ratio: 边界判定阈值（默认0.8）

    Returns:
        torch.Tensor: 边界点掩码，形状为 (N,)，边界点为True，非边界点为False
    """
    # 设置默认配置
    default_config = {
        "knn_k": 40,
        "bdl_ratio": 0.8
    }
    if config:
        default_config.update(config)
    config = default_config
    k = config["knn_k"]
    # 计算所有点对之间的欧氏距离
    dist = torch.cdist(points, points)
    # 获取k近邻索引（包括自身）
    _, indices = torch.topk(dist, k=k, largest=False, dim=1)
    # 获取邻居标签
    neighbor_labels = labels[indices]  # 形状: (N, k)
    # 计算每个点的众数标签及其出现次数
    # 将标签转换为one-hot编码以便于计算
    num_classes = int(labels.max() + 1)
    one_hot_labels = F.one_hot(neighbor_labels, num_classes).float()  # (N, k, C)
    # 统计每个类别在邻居中的出现次数
    class_counts = one_hot_labels.sum(dim=1)  # (N, C)
    # 找到每个点的众数标签的出现次数
    max_counts, _ = class_counts.max(dim=1)  # (N,)
    # 计算主要标签比例并生成边界掩码
    label_ratio = max_counts / k
    boundary_mask = label_ratio < config["bdl_ratio"]
    return boundary_mask



def knn_by_dgcnn(x, k):
    """使用DGCNN风格的KNN实现，通过矩阵运算高效计算最近邻点

    该方法通过矩阵运算而非显式计算所有点对距离来确定每个点的k个最近邻，
    具有内存效率高和计算速度快的特点。

    优点：
    - 内存占用为 O(Nk)
    - 使用矩阵运算，避免了显式计算所有点对之间的距离
    - 计算的是平方距离（避免开方运算），效率更高
    - 内存效率较高，不需要存储完整的距离矩阵

    Args:
        x (torch.Tensor): 输入点云数据，形状为 (batch_size, num_dims, num_points)
        k (int): 需要查找的最近邻数量

    Returns:
        torch.Tensor: 每个点的k个最近邻索引，形状为 (batch_size, num_points, k)
    """
    # 计算内积项: (batch_size, num_points, num_points)
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    # 计算每个点的平方和: (batch_size, 1, num_points)
    xx = torch.sum(x **2, dim=1, keepdim=True)
    # 计算 pairwise 平方距离: (batch_size, num_points, num_points)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    # 获取每个点的k个最近邻索引
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx