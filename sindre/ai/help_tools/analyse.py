
import  torch
def pca_color_by_feat(feat, brightness=1.25, center=True):
    """
    通过PCA将高维特征转换为RGB颜色，用于可视化。

    该函数使用主成分分析(PCA)对输入特征进行降维，
    组合前6个主成分生成3维颜色向量，并将其归一化到[0, 1]范围，
    适用于作为RGB颜色值进行点云等数据的可视化。

    Args:
        feat (torch.Tensor): 输入的高维特征张量。
            形状应为(num_points, feature_dim)，其中num_points是点的数量，
            feature_dim是每个特征的维度。
        brightness (float, 可选): 颜色亮度的缩放因子。
            值越高，整体颜色越明亮。默认值为1.25。
        center (bool, 可选): 在执行PCA之前是否对特征进行中心化（减去均值）。
            默认值为True。

    Returns:
         torch.Tensor: 归一化到[0, 1]范围的RGB颜色值。
            形状为(num_points, 3)，每行代表(R, G, B)三个通道的颜色值。

    """
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color
