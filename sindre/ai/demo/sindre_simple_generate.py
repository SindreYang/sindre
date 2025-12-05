import os
import time
from typing import Union, Tuple, List, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat,rearrange
from skimage import measure
import numpy as np

import sindre.utils3d
from sindre import CustomLogger
from sindre.deploy.check_tools import timeit
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from sindre.ai.layers import CrossAttentionDecoder,FourierEmbedder
from sindre.utils3d import SindreMesh

log=CustomLogger("sindre_generate").get_logger()



def generate_test_data(device, batch_size=1, grid_size=256):
    """生成测试用的 3D 网格数据"""
    # 创建一个简单的球体形状的 SDF
    x = torch.linspace(-1, 1, grid_size, device=device)
    y = torch.linspace(-1, 1, grid_size, device=device)
    z = torch.linspace(-1, 1, grid_size, device=device)
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    # 创建球体 SDF (内部为负值，外部为正值) 球体SDF：中心在(-0.5,-0.5,-0.5)，半径0.5
    sphere_sdf = ((X+0.5)**2 + (Y+0.5)**2 + (Z+0.5)**2) - 0.5**2
    # 添加批次维度
    grid_logits = sphere_sdf.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    return grid_logits


class VoxelFeatureQuery(nn.Module):
    def __init__(self, voxel_size=256):
        """
        基于交叉注意力的体素查询模块

        """
        super().__init__()

        # 投影层：将坐标转换为位置编码
        self.pos_enc = FourierEmbedder()

        # 生成基础网格坐标
        self.register_buffer("base_grid", self._create_base_grid(voxel_size))

    def _create_base_grid(self, size):
        """创建基础网格坐标 [-1, 1]"""
        d = torch.linspace(-1, 1, size)
        h = torch.linspace(-1, 1, size)
        w = torch.linspace(-1, 1, size)
        grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')
        grid = torch.stack((grid_w, grid_h, grid_d), dim=-1)  # [D, H, W, 3]
        return grid

    def forward(self, latent,coords):
        """
        将多视图（64张图像）及 条件点云特征
        期望：
            通过这个两个特征生成符合位置的sdf
        输入:
            latent: 多视图特征 [B,64, H, W]
            coords: 空间点云位置特征 [ B,N,3]
        输出:
            voxel_features: 体素特征 [B, C ,voxel_size,voxel_size,voxel_size]
        """
        # 点云特征编码
        coords_enc = self.pos_enc(coords)

        # 映射到grid
        coords_feats = F.grid_sample(
            input=coords_enc,
            grid=self.base_grid,
            padding_mode='border',
            align_corners=False
        )


        # 多视图映射到grid







class SurfaceExtractorDMC(nn.Module):
    def __init__(self):
        super().__init__()
        from diso import DiffDMC
        self.dmc = DiffDMC(dtype=torch.float32).to(device)
        self.resolution=256

    def computer_loss(self,gen_mesh,trg_mesh):
        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(gen_mesh, 5000)
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        loss_edge = mesh_edge_loss(gen_mesh)
        loss_normal = mesh_normal_consistency(gen_mesh)
        loss_laplacian = mesh_laplacian_smoothing(gen_mesh, method="uniform")
        loss = loss_chamfer * 1.0 + loss_edge * 1.0 + loss_normal * 0.01 + loss_laplacian * 0.1
        return loss



    def forward(self, grid_logits):
        """

        Args:
            grid_logits: [b,res,res,res] 的occ/sdf
        Returns:

        """
        batchsize = grid_logits.shape[0]
        verts_list = []
        faces_list = []
        # 生成网格
        for bs in range(batchsize):
            verts, faces = self.dmc(grid_logits[bs], deform=None, return_quads=False, normalize=False)
            # 坐标转换：体素索引(0~255) → 归一化坐标(-1~1)  公式：normalized = (voxel_index / (resolution-1)) * 2 - 1
            verts = verts / (self.resolution - 1)  # 先缩放到0~1
            verts = verts * 2 - 1
            faces = torch.flip(faces, dims=[1])
            verts_list.append(verts)
            faces_list.append(faces)

        # 添加到list
        gen_mesh = Meshes(verts=verts_list, faces=faces_list)

        return gen_mesh



if __name__ == "__main__":
    os.environ["DISPLAY"]=":0"
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # 生成测试数据
    batch_size = 20
    grid_size = 256
    grid_logits = generate_test_data(device, batch_size, grid_size)
    print(f"测试数据形状: {grid_logits.shape},{grid_logits.max(),grid_logits.min()}")

    dmc = SurfaceExtractorDMC()
    gen_mesh = dmc(grid_logits)
    sm =SindreMesh(gen_mesh)
    print(sm)
    sm.show()



