import torch
import torch.nn as nn
import torch.nn.functional as F

from sindre.ai.resnet_utils.block import ResnetBlockFC
from sindre.ai.unet_utils.unet import UNet

try:
    from torch_scatter import scatter_mean, scatter_max
except ImportError:
    print("请安装：pip install torch_scatter")

"""
基于PointNet的3D点云特征提取网络，结合ResNet块和UNet结构进行特征学习。
支持多平面特征聚合（xz/xy/yz平面）和3D网格特征生成。
"""

class ConvPointnet(nn.Module):
    """
    基于PointNet的编码器网络，每个点使用ResNet块进行特征提取。

    Args:
        c_dim (int): 潜在代码c的维度（默认512）
        dim (int): 输入点的维度（默认3）
        hidden_dim (int): 网络隐藏层维度（默认128）
        scatter_type (str): 局部池化时的特征聚合方式（'max'或'mean'）
        unet (bool): 是否使用UNet结构处理平面特征（默认True）
        unet_kwargs (dict): UNet参数配置（默认深度4，合并方式concat）
        plane_resolution (int): 平面特征分辨率（默认64）
        plane_type (str): 特征类型（'xz'单平面，['xz','xy','yz']三平面，['grid']3D网格）
        padding (float): 坐标归一化时的填充系数（默认0.1）
        n_blocks (int): ResNet块数量（默认5）
        inject_noise (bool): 是否注入噪声（默认False）
    """
    def __init__(self, c_dim=512, dim=3, hidden_dim=128, scatter_type='max', 
                 unet=True, unet_kwargs={"depth": 4, "merge_mode": "concat", "start_filts": 32}, 
                 plane_resolution=64, plane_type=['xz', 'xy', 'yz'], padding=0.1, n_blocks=5,
                 inject_noise=False):
        super().__init__()
        self.c_dim = c_dim

        # 坐标特征编码层
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        
        # ResNet块序列
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for _ in range(n_blocks)
        ])
        
        # 潜在代码生成层
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        # UNet平面特征处理模块
        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        self.reso_plane = plane_resolution
        self.plane_type = plane_type
        self.padding = padding

        # 池化操作选择
        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean

    def generate_plane_features(self, p, c, plane='xz'):
        """
        生成指定平面的特征图

        Args:
            p (Tensor): 输入点云 (B, T, 3)
            c (Tensor): 点云特征 (B, T, c_dim)
            plane (str): 平面类型 ('xz','xy','yz')

        Returns:
            Tensor: 平面特征图 (B, c_dim, reso, reso)
        """
        # 坐标归一化到[0,1]
        xy = self.normalize_coordinate(p.clone(), plane=plane, padding=self.padding)
        # 坐标转索引
        index = self.coordinate2index(xy, self.reso_plane)

        # 初始化平面特征张量
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        # 特征聚合
        c = c.permute(0, 2, 1)  # B x c_dim x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x c_dim x reso^2
        # 重塑为二维特征图
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane)

        # UNet特征处理
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane 

    def forward(self, p, query):
        """
        前向传播主函数

        Args:
            p (Tensor): 输入点云 (B, T, 3)
            query (Tensor): 查询点坐标 (B, Q, 3)

        Returns:
            Tensor: 聚合后的特征 (B, Q, c_dim)
        """
        batch_size, T, D = p.size()

        # 多平面坐标预处理
        coord = {}
        index = {}
        if 'xz' in self.plane_type:
            coord['xz'] = self.normalize_coordinate(p.clone(), plane='xz', padding=self.padding)
            index['xz'] = self.coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = self.normalize_coordinate(p.clone(), plane='xy', padding=self.padding)
            index['xy'] = self.coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = self.normalize_coordinate(p.clone(), plane='yz', padding=self.padding)
            index['yz'] = self.coordinate2index(coord['yz'], self.reso_plane)

        # 坐标编码
        net = self.fc_pos(p)

        # ResNet块处理
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            # 局部池化
            pooled = self.pool_local(coord, index, net)
            # 特征融合
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        # 生成潜在特征
        c = self.fc_c(net)
        
        # 多平面特征聚合
        plane_feat_sum = 0
        if 'xz' in self.plane_type:
            fea_xz = self.generate_plane_features(p, c, plane='xz')
            plane_feat_sum += self.sample_plane_feature(query, fea_xz, 'xz')
        if 'xy' in self.plane_type:
            fea_xy = self.generate_plane_features(p, c, plane='xy')
            plane_feat_sum += self.sample_plane_feature(query, fea_xy, 'xy')
        if 'yz' in self.plane_type:
            fea_yz = self.generate_plane_features(p, c, plane='yz')
            plane_feat_sum += self.sample_plane_feature(query, fea_yz, 'yz')

        return plane_feat_sum.transpose(2,1)

    def coordinate2index(self, x, reso):
        """
        将归一化坐标转换为网格索引

        Args:
            x (Tensor): 归一化坐标 (B, T, 2)
            reso (int): 网格分辨率

        Returns:
            Tensor: 网格索引 (B, 1, T)
        """
        x = (x * reso).long()
        index = x[:, :, 0] + reso * x[:, :, 1]
        return index[:, None, :]

    def pool_local(self, xy, index, c):
        """
        局部特征池化操作

        Args:
            xy (dict): 各平面归一化坐标
            index (dict): 各平面网格索引
            c (Tensor): 点云特征 (B, T, c_dim)

        Returns:
            Tensor: 池化后的特征 (B, T, c_dim)
        """
        bs, fea_dim = c.size(0), c.size(2)
        c_out = 0
        for key in xy.keys():
            # 特征聚合
            fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # 特征还原
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def sample_plane_feature(self, query, plane_feature, plane):
        """
        从平面特征图采样特征值

        Args:
            query (Tensor): 查询点坐标 (B, Q, 3)
            plane_feature (Tensor): 平面特征图 (B, c_dim, reso, reso)
            plane (str): 平面类型

        Returns:
            Tensor: 采样后的特征 (B, Q, c_dim)
        """
        # 坐标归一化到[0,1]
        xy = self.normalize_coordinate(query.clone(), plane=plane, padding=self.padding)
        # 转换为GridSample格式
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0  # 归一化到[-1, 1]
        # 双线性插值
        sampled_feat = F.grid_sample(
            plane_feature, vgrid, 
            padding_mode='border', 
            align_corners=True, 
            mode='bilinear'
        ).squeeze(-1)
        return sampled_feat

    def normalize_coordinate(self, p, padding=0.1, plane='xz'):
        """
        坐标归一化到[0,1]范围

        Args:
            p (Tensor): 输入坐标 (B, T, 3)
            padding (float): 填充系数
            plane (str): 平面类型

        Returns:
            Tensor: 归一化坐标 (B, T, 2)
        """
        if plane == 'xz':
            xy = p[:, :, [0, 2]]
        elif plane == 'xy':
            xy = p[:, :, [0, 1]]
        else:
            xy = p[:, :, [1, 2]]

        # 归一化处理
        xy_new = xy / (1 + padding)
        xy_new = xy_new + 0.5

        # 边界处理
        xy_new[xy_new >= 1] = 1 - 1e-6
        xy_new[xy_new < 0] = 0.0
        return xy_new

# UNet实现
    
if __name__ == "__main__":
    model = ConvPointnet(c_dim=256, plane_resolution=32, n_blocks=3)
    p = torch.randn(2, 100, 3)  # 批次大小2，100个点，3维坐标
    query = torch.randn(2, 50, 3)  # 50个查询点
    output = model(p, query)
    print(output.shape)
    assert output.shape == (2, 50, 256), "输出形状错误"