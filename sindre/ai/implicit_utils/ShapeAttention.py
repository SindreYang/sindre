import argparse
import dataclasses
import os
import re
import sys
from typing import Callable, List, Union

import numpy as np
from skimage import measure
import torch
import torch.nn as nn
import trimesh
import vedo
from einops import repeat
from torch.utils.data import Dataset, DataLoader
from torch_cluster import fps
import torch.distributed as dist
from tqdm import tqdm

from sindre.ai.attention_utils.models.decoder import CrossAttentionDecoder, PointCrossAttentionEncoder
from sindre.ai.attention_utils.models.transformer import Transformer
from sindre.ai.embedder import FourierEmbedder
from sindre.ai.implicit_utils.utils import DiagonalGaussianDistribution, Latents2Meshes
from sindre.ai.pointcloud_utils.augment import *
from sindre.ai.utils import *
from sindre.lmdb import Reader

class MeshSharpSDFProcessor:
    """
    网格锐边与SDF（有符号距离函数）的采样处理类

    功能说明：
    1. 对输入网格进行归一化处理，验证网格水密性；
    2. 实现网格表面、锐边的点采样，以及空间点/近表面点的SDF计算；
    3. 对采样点执行最远点采样（FPS），生成最终的特征数组。
    """
    def __init__(self,vertices,faces,design_vertices,design_faces):
        """
        初始化网格SDF处理器

        Args:
            vertices (np.ndarray): 网格顶点数组，形状为(N, 3)，N为顶点数
            faces (np.ndarray): 网格面索引数组，形状为(M, 3)，M为面数

        Raises:
            AssertionError: 若输入网格非水密（watertight）时触发
        """
        # 采样数量配置（语义化变量名）
        self.surface_sample_count = 200000   # 表面采样总数
        self.sharp_sample_count = 200000     # 锐边采样总数
        self.space_sample_count = 200000     # 空间随机采样总数


        # 网格归一化与初始化
        import trimesh
        normalized_vertices,normalized_design_vertices = self.normalize_vertices(vertices,design_vertices)
        print(normalized_design_vertices.max(),normalized_design_vertices.min())
        self.jaw_mesh = trimesh.Trimesh(normalized_vertices, faces)
        self.design_mesh = trimesh.Trimesh(normalized_design_vertices, design_faces).split(only_watertight=True)[0]
        assert self.design_mesh.is_watertight, "设计冠必须是水密的（watertight）"
        assert len(self.design_mesh.vertices)>1000, "设计冠顶点必须大于1k"




        # 初始化CUDA加速的BVH树（用于SDF快速计算）
        import cubvh # pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
        self.bvh_tree = cubvh.cuBVH(normalized_design_vertices, design_faces)



    def normalize_vertices(self, vertices,design_vertices) :
        bb_min = vertices.min(axis=0)
        bb_max = vertices.max(axis=0)
        center = (bb_min + bb_max) / 2.0
        scale = 2.0 / (bb_max - bb_min).max()  # 最大轴长缩放到2（对应[-1,1]）
        normalized_vertices = (vertices - center) * scale
        normalized_design_vertices= (design_vertices - center) * scale*0.99
        return normalized_vertices,normalized_design_vertices

    def sample_surface_points(self,mesh) -> np.ndarray:
        """
        网格表面随机采样点，并拼接法向量生成特征数组

        Returns:
            np.ndarray: 表面采样特征数组，形状为(采样数, 6)，列顺序为(x,y,z,nx,ny,nz)
        """
        surface_points, face_indices = mesh.sample(
            self.surface_sample_count,
            return_index=True
        )
        face_normals =mesh.face_normals[face_indices]
        surface_feat = np.concatenate([surface_points, face_normals], axis=1)
        return surface_feat

    def sample_sharp_edges_points(self,mesh) -> np.ndarray:
        """
        网格锐边采样点，并拼接法向量生成特征数组

        Returns:
            np.ndarray: 锐边采样特征数组，形状为(采样数, 6)，列顺序为(x,y,z,nx,ny,nz)
        """
        from sindre.utils3d.sample import sample_mesh_sharp_edges
        sharp_points, sharp_normals = sample_mesh_sharp_edges(
            mesh.vertices,
            mesh.faces,
            self.sharp_sample_count
        )
        sharp_feat = np.concatenate([sharp_points, sharp_normals], axis=1)
        return sharp_feat



    def get_rand_points_sdf(self, surface_points: np.ndarray) -> np.ndarray:
        """
        计算空间随机点与表面近点的SDF，并生成特征数组

        Args:
            surface_points (np.ndarray): 表面采样点数组，形状为(N, 3)

        Returns:
            np.ndarray: 随机点SDF特征数组，形状为(总点数, 4)，列顺序为(x,y,z,sdf)
        """
        # 空间均匀采样点（范围略大于[-1,1]以覆盖边界）
        space_points = np.random.uniform(-1.05, 1.05, (self.space_sample_count, 3))

        # 表面近点采样（不同高斯噪声尺度）
        near_surface_points_list = [
            surface_points + np.random.normal(scale=0.001, size=surface_points.shape),
            surface_points + np.random.normal(scale=0.005, size=surface_points.shape),
            surface_points + np.random.normal(scale=0.015, size=surface_points.shape)
        ]
        near_surface_points = np.concatenate(near_surface_points_list, axis=0)

        # 合并所有点并计算SDF
        all_rand_points = np.concatenate([near_surface_points, space_points], axis=0)
        rand_sdf = self.bvh_tree.signed_distance(torch.tensor(all_rand_points))[0].cpu().numpy().reshape(-1, 1)
        rand_sdf_feat = np.concatenate([all_rand_points, rand_sdf], axis=1)

        return rand_sdf_feat

    def get_sharp_points_sdf(self, sharp_points: np.ndarray) -> np.ndarray:
        """
        计算锐边近表面点的SDF，并生成特征数组

        Args:
            sharp_points (np.ndarray): 锐边采样点数组，形状为(N, 3)

        Returns:
            np.ndarray: 锐边点SDF特征数组，形状为(总点数, 4)，列顺序为(x,y,z,sdf)
        """
        # 锐边近点采样（多种高斯噪声尺度）
        sharp_near_surface_points_list = [
            sharp_points + np.random.normal(scale=0.001, size=sharp_points.shape),
            sharp_points + np.random.normal(scale=0.005, size=sharp_points.shape),
            sharp_points + np.random.normal(scale=0.01, size=sharp_points.shape),
            sharp_points + np.random.normal(scale=0.015, size=sharp_points.shape),
            sharp_points + np.random.normal(scale=0.03, size=sharp_points.shape)
        ]
        sharp_near_surface_points = np.concatenate(sharp_near_surface_points_list, axis=0)

        # 计算SDF并拼接特征
        sharp_sdf = self.bvh_tree.signed_distance(torch.tensor(sharp_near_surface_points))[0].cpu().numpy().reshape(-1, 1)
        sharp_sdf_feat = np.concatenate([sharp_near_surface_points, sharp_sdf], axis=1)

        return sharp_sdf_feat

    def __call__(self) -> dict:
        """
        执行完整的网格采样与SDF计算流程

        Returns:
            dict: 包含各类采样特征的字典，键说明：
                - fps_surface_feat: FPS采样后的表面特征数组（float32）
                - fps_sharp_feat: FPS采样后的锐边特征数组（float32）
                - rand_sdf_feat: 随机点SDF特征数组（float32）
                - sharp_sdf_feat: 锐边近点SDF特征数组（float32）

        Raises:
            AssertionError: 若任意特征数组包含NaN值时触发（注：原逻辑assert np.isnan().all()为笔误，已修正为np.isnan().any()）
        """
        # 表面采样与处理
        surface_feat = self.sample_surface_points(self.jaw_mesh)
        # 锐边采样与处理
        sharp_feat = self.sample_sharp_edges_points(self.jaw_mesh)



        ##########  设计冠       ##########
        # 表面采样与处理
        design_surface_feat = self.sample_surface_points(self.design_mesh)
        # 锐边采样与处理
        design_sharp_feat = self.sample_sharp_edges_points(self.design_mesh)

        # 随机点SDF计算
        rand_sdf_feat = self.get_rand_points_sdf(design_surface_feat[..., :3])
        # 锐边点SDF计算
        sharp_sdf_feat = self.get_sharp_points_sdf(design_sharp_feat[..., :3])



        # 检查NaN值
        assert not np.isnan(surface_feat).any(), "表面采样特征包含NaN值"
        assert not np.isnan(sharp_feat).any(), "锐边采样特征包含NaN值"
        assert not np.isnan(rand_sdf_feat).any(), "随机点SDF特征包含NaN值"
        assert not np.isnan(sharp_sdf_feat).any(), "锐边点SDF特征包含NaN值"

        # 整理输出
        output_dict = {
            "surface_feat": surface_feat.astype(np.float16),
            "sharp_feat": sharp_feat.astype(np.float16),
            "rand_sdf_feat": rand_sdf_feat.astype(np.float16),
            "sharp_sdf_feat": sharp_sdf_feat.astype(np.float16),
        }

        return output_dict









class ShapeAttention(nn.Module):
    def __init__(self, pc_size = 81920 // 2,pc_sharpedge_size= 81920 //2,):
        super(ShapeAttention, self).__init__()
        self.out_dim =1 # 输出sdf/occ/tsdf/udf
        self.embed_dim=64 # 嵌入层维度
        self.num_latents=4096
        self.width =1024
        self.heads =16
        self.point_feats=3 # xyz + normal
        self.encoder_layers=8
        self.decoder_layers=16
        self.kv_bias=False
        self.embedder = FourierEmbedder(num_freqs=8,include_pi=False)

        self.pre_kl = nn.Linear(self.width, self.embed_dim * 2)
        self.post_kl = nn.Linear(self.embed_dim, self.width)
        
        self.shape_encoder = PointCrossAttentionEncoder(
            pc_size = pc_size,
            pc_sharpedge_size= pc_sharpedge_size,
            num_latents=self.num_latents,
            fourier_embedder=self.embedder,
            point_feats=4,
            width=self.width,
            heads=self.heads,
            layers=self.encoder_layers,
            qkv_bias=self.kv_bias,
        )




        self.transformer = Transformer(
                n_ctx=self.num_latents,
                width=self.width,
                layers=self.decoder_layers,
                heads=self.heads,
                qkv_bias=self.kv_bias,
        )
        self.query_decoder = CrossAttentionDecoder(out_channels=self.out_dim,
                                                   fourier_embedder=self.embedder,
                                                   width=self.width,
                                                   heads=self.heads,
                                                   qkv_bias=self.kv_bias,
                                                   )

    
    def get_latents(self,  *,surface_feat):
        pc, feats = surface_feat[:, :, :3], surface_feat[:, :, 3:]
        # 形状特征提取
        shape_latents,_ = self.shape_encoder(pc, feats)
        # 生成形状高斯分布采样
        moments = self.pre_kl(shape_latents) # 103，256，768 -》 103，256，128
        shape_posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)
        shape_kl_embed = shape_posterior.sample() # 1，768，6
        last_latents = self.post_kl(shape_kl_embed) # [B, num_latents, embed_dim] -> [B, num_latents, width]
        last_latents = self.transformer(last_latents)
        return last_latents,shape_posterior
    
    def query(self,queries,latents):
        logits = self.query_decoder(queries,latents).squeeze(-1)
        return logits
    
    def forward(self, *,surface_feat,queries):
        last_latents,shape_posterior = self.get_latents(surface_feat=surface_feat)
        # 查询logits
        logits = self.query_decoder(queries,last_latents).squeeze(-1)
        return logits,shape_posterior





class ShapeTrainLoss(nn.Module):
    def __init__(self,loss_type="occ",kl_weight=1e-4):
        super(ShapeTrainLoss, self).__init__()
        if loss_type=="occ":
            print("ShapeTrainLoss启用occ Loss（BCEWithLogits）")
            self.criteria = torch.nn.BCEWithLogitsLoss()
        else:
            print("ShapeValLoss启用MSE Loss（回归）")
            self.criteria = torch.nn.MSELoss()

        self.kl_weight=kl_weight
    def forward(self, *,
                logits,
                target,
                posterior=None,
                ):
        loss_logits=self.criteria(logits, target)
        loss_kl = torch.tensor(0.0, device=logits.device)
        if posterior is not None:
            loss_kl = posterior.kl(dims=(1, 2)).mean()*self.kl_weight

        loss = loss_logits + loss_kl
        return loss,loss_kl



class ShapeValLoss(nn.Module):
    def __init__(self,loss_type="occ"):
        super(ShapeValLoss,self).__init__()
        self.threshold = 0
        if loss_type=="occ":
            print("ShapeValLoss启用occ Loss（BCEWithLogits）")
            self.criteria = torch.nn.BCEWithLogitsLoss()
        else:
            print("ShapeValLoss启用MSE Loss（回归）")
            self.criteria = torch.nn.MSELoss()

    def forward(self,*, logits,target):
        #print(f"logits统计：min={logits.min().item()}, max={logits.max().item()}, mean={logits.mean().item()}")
        # print(f"target统计：min={target.min().item()}, max={target.max().item()}, mean={target.mean().item()}")
        # print(f"logits.shape={logits.shape}, target.shape={target.shape}")
        # 计算loss
        loss = self.criteria(logits, target)

        # 转换为0，1
        overall_labels = (target >= 0.0).float()  # target≥0为正类（1），否则负类（0）
        overall_pred = (logits >= self.threshold).float()  # logits≥阈值为正类

        # 全局准确率
        overall_correct = (overall_pred == overall_labels).float()
        overall_accuracy = overall_correct.mean()


        # 全局IoU
        intersection = (overall_pred * overall_labels).sum(dim=1)
        union = (overall_pred + overall_labels).gt(0).sum(dim=1)
        overall_iou = intersection / (union + 1e-5)
        overall_iou = overall_iou.mean()



        return loss,overall_accuracy, overall_iou



