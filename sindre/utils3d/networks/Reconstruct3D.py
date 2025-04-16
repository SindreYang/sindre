# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from typing import Tuple, List, Union, Optional
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from einops import  repeat
from sindre.utils3d.networks.embed_attention import *


class ShapeVAE(nn.Module):
    def __init__(
        self,
        *,
        num_latents: int,
        embed_dim: int,
        width: int,
        heads: int,
        num_decoder_layers: int,
        num_freqs: int = 8,
        include_pi: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        label_type: str = "binary",
        drop_path_rate: float = 0.0,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        self.post_kl = nn.Linear(embed_dim, width)

        self.transformer = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            drop_path_rate=drop_path_rate
        )

        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            label_type=label_type,
        )

        self.scale_factor = scale_factor
        self.latent_shape = (num_latents, embed_dim)

    def forward(self, latents):
        latents = self.post_kl(latents)
        latents = self.transformer(latents)
        return latents



    @staticmethod
    def generate_dense_grid_points(bbox_min: np.ndarray,
                               bbox_max: np.ndarray,
                               octree_depth: int,
                               indexing: str = "ij",
                               octree_resolution: int = None,
                               ):
        length = bbox_max - bbox_min
        num_cells = np.exp2(octree_depth)
        if octree_resolution is not None:
            num_cells = octree_resolution

        x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
        y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
        z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
        [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
        xyz = np.stack((xs, ys, zs), axis=-1)
        xyz = xyz.reshape(-1, 3)
        grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

        return xyz, grid_size, length

    @staticmethod
    def center_vertices(vertices):
        """Translate the vertices so that bounding box is centered at zero."""
        vert_min = vertices.min(dim=0)[0]
        vert_max = vertices.max(dim=0)[0]
        vert_center = 0.5 * (vert_min + vert_max)
        return vertices - vert_center

    

    @torch.no_grad()
    def latents2mesh(
        self,
        latents: torch.FloatTensor,
        bounds: Union[Tuple[float], List[float], float] = 1.1,
        octree_depth: int = 7,
        num_chunks: int = 10000,
        mc_level: float = -1 / 512,
        octree_resolution: int = None,
        mc_algo: str = 'dmc',
    ):
        device = latents.device

        # 1. 生成查询点
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        xyz_samples, grid_size, length = self.generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_depth=octree_depth,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        xyz_samples = torch.FloatTensor(xyz_samples)

        # 2. 将latents转换成体素
        batch_logits = []
        batch_size = latents.shape[0]
        for start in tqdm(range(0, xyz_samples.shape[0], num_chunks),
                          desc=f"MC Level {mc_level} Implicit Function:"):
            queries = xyz_samples[start: start + num_chunks, :].to(device)
            queries = queries.half()
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)

            logits = self.geo_decoder(batch_queries.to(latents.dtype), latents)
            if mc_level == -1:
                mc_level = 0
                logits = torch.sigmoid(logits) * 2 - 1
                print(f'Training with soft labels, inference with sigmoid and marching cubes level 0.')
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, grid_size[0], grid_size[1], grid_size[2])).float()

        # 3. 将体素转换成网格
        outputs = {"vertices":None,"faces":None}
        for i in range(batch_size):
            try:
                if mc_algo == 'mc':
                    from skimage import measure
                    vertices, faces, normals, _ = measure.marching_cubes(
                        grid_logits[i].cpu().numpy(),
                        mc_level,
                        method="lewiner"
                    )
                    vertices = vertices / grid_size * bbox_size + bbox_min
                elif mc_algo == 'dmc':
                    if not hasattr(self, 'dmc'):
                        try:
                            from diso import DiffDMC
                        except:
                            raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
                        self.dmc = DiffDMC(dtype=torch.float32).to(device)
                    octree_resolution = 2 ** octree_depth if octree_resolution is None else octree_resolution
                    sdf = -grid_logits[i] / octree_resolution
                    verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
                    verts = self.center_vertices(verts)
                    vertices = verts.detach().cpu().numpy()
                    faces = faces.detach().cpu().numpy()[:, ::-1]
                else:
                    raise ValueError(f"mc_algo {mc_algo} not supported.")

                outputs["vertices"] = vertices.astype(np.float32)
                outputs["faces"] = np.ascontiguousarray(faces)

            except Exception as e:
                print(e)
                # 返回None
                pass

        return outputs
    
    
if __name__ =="__main__":
    num_latents = 16
    embed_dim = 32
    width = 64
    heads = 4
    num_decoder_layers = 2
    # 初始化模型
    model = ShapeVAE(
        num_latents=num_latents,
        embed_dim=embed_dim,
        width=width,
        heads=heads,
        num_decoder_layers=num_decoder_layers,
    )
    # 推理
    batch_size = 2
    latents = torch.randn(batch_size, num_latents, embed_dim)
    output = model.forward(latents)
    print(f"input.shape:{latents.shape} \noutput.shape: {output.shape}")

    #重建为网格
    output_mesh_dict= model.latents2mesh(latents=output)
    print(output_mesh_dict)
    print(f"v:{output_mesh_dict['vertices'].shape},f: {output_mesh_dict['faces'].shape}")
    # import vedo
    # vedo.Mesh([output_mesh_dict["vertices"],output_mesh_dict["faces"]]).show().close()
    

    
        