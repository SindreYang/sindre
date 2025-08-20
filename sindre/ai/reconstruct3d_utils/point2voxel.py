"""
结合gridsample和conv3d处理点云数据 - anchor的文章 - 知乎
https://zhuanlan.zhihu.com/p/1894179743487743623

核心实现思路
    体素化预处理：将无序点云映射到规则网格
    特征提取：提取体素局部几何特征
    动态网格采样：预测偏移量进行特征重采样
    网格重建：生成规则拓扑结构


关键实现说明
    体素化模块：
    使用3D直方图统计点云分布
    归一化坐标到[-1,1]区间
    输出体素密度图
    动态采样器：
    通过3D卷积预测每个体素的偏移量
    对基础网格进行非线性形变
    grid_sample实现特征重采样
    网格生成：
    解码器输出体素占用概率
    结合Marching Cubes算法生成三角面片（需配合PyMCubes等库）

性能优化技巧
    稀疏体素处理：
    # 使用稀疏张量加速计算
    sparse_voxel = voxel.to_sparse()
    多尺度特征融合：
    # 添加多尺度采样路径
    self.sampler2 = DynamicGridSampler(64, 3)
    自适应网格密度：
    # 根据点密度调整网格分辨率
    adaptive_grid_size = int(points.std() * 64)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sindre.ai.utils import sdf2mesh_by_diso

class PointCloudVoxelizer(nn.Module):
    def __init__(self, grid_size=64):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, points):
        """
        输入：点云 [B, N, 3]
        输出：体素特征 [B, C, D, H, W]
        """
        # 归一化到[-1,1]区间
        points_min = points.min(dim=1, keepdim=True)[0]
        points_max = points.max(dim=1, keepdim=True)[0]
        normalized_points = (points - points_min) / (points_max - points_min + 1e-6) * 2 - 1

        # 体素化（3D直方图）
        voxel_features = []
        for b in range(points.size(0)):
            hist = torch.histogramdd(
                input=normalized_points[b], 
                bins=self.grid_size,
                range=[-1,1, -1,1, -1,1]
            ).hist
            voxel_features.append(hist)
        return torch.stack(voxel_features, dim=0).unsqueeze(1)  # [B,1,D,H,W]

class DynamicGridSampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset_conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 3, 3, padding=1)  # 预测x,y,z偏移量
        )

    def forward(self, voxel_feats):
        """
        输入：体素特征 [B,C,D,H,W]
        输出：变形网格特征 [B,C,D,H,W]
        """
        # 生成基础网格
        B, C, D, H, W = voxel_feats.size()
        grid = self._create_base_grid(D, H, W, voxel_feats.device)  # [D,H,W,3]

        # 预测动态偏移量
        offset = self.offset_conv(voxel_feats)  # [B,3,D,H,W]
        offset = offset.permute(0,2,3,4,1)  # [B,D,H,W,3]

        # 应用偏移量
        deformed_grid = grid + offset * 0.1  # 控制偏移范围
        deformed_grid = deformed_grid.clamp(-1, 1)  # 确保在合法范围

        # 重采样特征
        sampled_feats = F.grid_sample(
            input=voxel_feats, 
            grid=deformed_grid,
            padding_mode='border',
            align_corners=False
        )

        return sampled_feats

    def _create_base_grid(self, D, H, W, device):
        z = torch.linspace(-1, 1, D, device=device)
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid([z, y, x], indexing='ij')
        return torch.stack((grid_x, grid_y, grid_z), dim=-1)  # [D,H,W,3]

class PointCloud2Mesh(nn.Module):
    def __init__(self, grid_size=64):
        super().__init__()
        self.voxelizer = PointCloudVoxelizer(grid_size)
        self.sampler = DynamicGridSampler(1, 3)
        self.decoder = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 1, 3, padding=1),
            nn.Sigmoid()  # 输出0-1的占用概率
        )

        self.diffdmc =None

    def forward(self, points):
        voxel = self.voxelizer(points)
        deformed_feats = self.sampler(voxel)
        occupancy = self.decoder(deformed_feats)
        return occupancy  # 体素占用概率

    def get_mesh(self,sdf,diffdmc=None ,deform=None,return_quads=False, normalize=True,isovalue=0):
        device= sdf.device
        if self.diffdmc is None:
            try:
                from diso import DiffDMC
            except ImportError:
                raise ("请安装 pip install diso")
            self.diffdmc =DiffDMC(dtype=torch.float32).to(device)
        v, f = diffdmc(sdf, deform, return_quads=return_quads, normalize=normalize, isovalue=isovalue) 
        return v,f

# 使用示例
if __name__ == "__main__":
    # 生成随机点云 [B, N, 3]
    points = torch.rand(2, 1024, 3) * 2 -1  # 范围[-1,1]

    model = PointCloud2Mesh(grid_size=64)
    occupancy = model(points)
    print(occupancy.shape)

    # Marching Cubes生成网格 (需配合外部库实现)
    # vertices, faces = marching_cubes(occupancy.squeeze().detach().numpy())