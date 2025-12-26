from typing import Optional
import torch
from torch import nn
from torch_cluster import fps

from sindre.ai.attention_utils.cross_attention import ResidualCrossAttentionBlock
from sindre.ai.attention_utils.models.transformer import Transformer
from sindre.ai.embedder import FourierEmbedder


class CrossAttentionDecoder(nn.Module):
    """交叉注意力解码器模块，用于通过潜在变量（Latents）增强查询（Queries）的特征表示。

    该模块将输入查询通过傅里叶嵌入编码后，与潜在变量进行交叉注意力交互，最终生成目标输出（如分类概率）。

    Args:
        num_latents (int): 潜在变量的数量（即每个样本的上下文标记数）。
        out_channels (int): 输出通道数（如分类类别数）。
        fourier_embedder (FourierEmbedder): 傅里叶特征嵌入器，用于编码输入查询。
        width (int): 特征投影后的维度（注意力模块的隐藏层宽度）。
        heads (int): 注意力头的数量。
        qkv_bias (bool): 是否在 Q/K/V 投影中添加偏置项，默认为 True。
        qk_norm (bool): 是否对 Q/K 进行层归一化，默认为 False。

    Attributes:
        query_proj (nn.Linear): 将傅里叶嵌入后的查询投影到指定宽度的线性层。
        cross_attn_decoder (ResidualCrossAttentionBlock): 残差交叉注意力块。
        ln_post (nn.LayerNorm): 输出前的层归一化。
        output_proj (nn.Linear): 最终输出投影层。

    Shape:
        - 输入 queries: (bs, num_queries, query_dim)
        - 输入 latents: (bs, num_latents, latent_dim)
        - 输出 occ: (bs, num_queries, out_channels)
    """

    def __init__(
            self,
            *,
            out_channels: int,
            fourier_embedder: FourierEmbedder,
            width: int,
            heads: int,
            qkv_bias: bool = True,
            norm_method: str = "LayerNorm",
            atten_method: str = "SDPA"
    ):
        super().__init__()
        self.fourier_embedder = fourier_embedder

        # 将傅里叶嵌入后的查询投影到指定维度（width）
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)

        # 残差交叉注意力模块（处理查询与潜在变量的交互）
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
            norm_method=norm_method,
            atten_method=atten_method
        )

        # 后处理层
        self.ln_post = nn.LayerNorm(width)  # 输出归一化
        self.output_proj = nn.Linear(width, out_channels)  # 输出投影



    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor) -> torch.FloatTensor:
        """前向传播流程：傅里叶嵌入 -> 投影 -> 交叉注意力 -> 归一化 -> 输出投影。

        Args:
            queries (torch.FloatTensor): 输入查询张量，形状 (bs, num_queries, query_dim)
            latents (torch.FloatTensor): 潜在变量张量，形状 (bs, num_latents, latent_dim)

        Returns:
            torch.FloatTensor: 输出张量，形状 (bs, num_queries, out_channels)
        """
        # 傅里叶嵌入 + 投影（保持与潜在变量相同的数据类型）
        queries = self.query_proj(self.fourier_embedder(queries).to(latents.dtype))

        # 残差交叉注意力交互
        x = self.cross_attn_decoder(queries, latents)

        # 后处理与输出
        x = self.ln_post(x)
        occ = self.output_proj(x)  # 输出如占据概率、分类logits等

        return occ




class PointCrossAttentionEncoder(nn.Module):

    def __init__(
        self, *,
        downsample_ratio: float=20,
        pc_size: int=81920//2,
        pc_sharpedge_size: int=81920//2,
        num_latents: int,
        fourier_embedder: FourierEmbedder,
        point_feats: int,
        width: int,
        heads: int,
        layers: int,
        normal_pe: bool = False,
        qkv_bias: bool = True,

    ):

        super().__init__()
        self.num_latents = num_latents
        self.downsample_ratio = downsample_ratio
        self.point_feats = point_feats
        self.normal_pe = normal_pe

        if pc_sharpedge_size == 0:
            print(
                f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is zero')
        else:
            print(
                f'PointCrossAttentionEncoder INFO: pc_sharpedge_size is given, using pc_size={pc_size}, pc_sharpedge_size={pc_sharpedge_size}')

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size

        self.fourier_embedder = fourier_embedder

        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)
        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias
        )

 
    
        self.self_attn = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            qkv_bias=qkv_bias,
        )

        self.ln_post = nn.LayerNorm(width)


    def sample_points_and_latents(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        B, N, D = pc.shape
        num_pts = self.num_latents * self.downsample_ratio

        # Compute number of latents
        num_latents = int(num_pts / self.downsample_ratio)

        # Compute the number of random and sharpedge latents
        num_random_query = self.pc_size / (self.pc_size + self.pc_sharpedge_size) * num_latents
        num_sharpedge_query = num_latents - num_random_query

        # Split random and sharpedge surface points
        random_pc, sharpedge_pc = torch.split(pc, [self.pc_size, self.pc_sharpedge_size], dim=1)
        assert random_pc.shape[1] <= self.pc_size, "Random surface points size must be less than or equal to pc_size"
        assert sharpedge_pc.shape[
                   1] <= self.pc_sharpedge_size, "Sharpedge surface points size must be less than or equal to pc_sharpedge_size"

        # Randomly select random surface points and random query points
        input_random_pc_size = int(num_random_query * self.downsample_ratio)
        random_query_ratio = num_random_query / input_random_pc_size
        idx_random_pc = torch.randperm(random_pc.shape[1], device=random_pc.device)[:input_random_pc_size]
        input_random_pc = random_pc[:, idx_random_pc, :]
        flatten_input_random_pc = input_random_pc.view(B * input_random_pc_size, D)
        N_down = int(flatten_input_random_pc.shape[0] / B)
        batch_down = torch.arange(B).to(pc.device)
        batch_down = torch.repeat_interleave(batch_down, N_down)
        idx_query_random = fps(flatten_input_random_pc.to(torch.float32), batch_down, ratio=random_query_ratio)
        query_random_pc = flatten_input_random_pc[idx_query_random].view(B, -1, D)

        # Randomly select sharpedge surface points and sharpedge query points
        input_sharpedge_pc_size = int(num_sharpedge_query * self.downsample_ratio)
        if input_sharpedge_pc_size == 0:
            input_sharpedge_pc = torch.zeros(B, 0, D, dtype=input_random_pc.dtype).to(pc.device)
            query_sharpedge_pc = torch.zeros(B, 0, D, dtype=query_random_pc.dtype).to(pc.device)
        else:
            sharpedge_query_ratio = num_sharpedge_query / input_sharpedge_pc_size
            idx_sharpedge_pc = torch.randperm(sharpedge_pc.shape[1], device=sharpedge_pc.device)[
                               :input_sharpedge_pc_size]
            input_sharpedge_pc = sharpedge_pc[:, idx_sharpedge_pc, :]
            flatten_input_sharpedge_surface_points = input_sharpedge_pc.view(B * input_sharpedge_pc_size, D)
            N_down = int(flatten_input_sharpedge_surface_points.shape[0] / B)
            batch_down = torch.arange(B).to(pc.device)
            batch_down = torch.repeat_interleave(batch_down, N_down)
            idx_query_sharpedge = fps(flatten_input_sharpedge_surface_points.to(torch.float32), batch_down, ratio=sharpedge_query_ratio)
            query_sharpedge_pc = flatten_input_sharpedge_surface_points[idx_query_sharpedge].view(B, -1, D)

        # Concatenate random and sharpedge surface points and query points
        query_pc = torch.cat([query_random_pc, query_sharpedge_pc], dim=1)
        input_pc = torch.cat([input_random_pc, input_sharpedge_pc], dim=1)

        # PE
        query = self.fourier_embedder(query_pc)
        data = self.fourier_embedder(input_pc)

        # Concat normal if given
        if self.point_feats != 0:

            random_surface_feats, sharpedge_surface_feats = torch.split(feats, [self.pc_size, self.pc_sharpedge_size],
                                                                        dim=1)
            input_random_surface_feats = random_surface_feats[:, idx_random_pc, :]
            flatten_input_random_surface_feats = input_random_surface_feats.view(B * input_random_pc_size, -1)
            query_random_feats = flatten_input_random_surface_feats[idx_query_random].view(B, -1,
                                                                                           flatten_input_random_surface_feats.shape[
                                                                                               -1])

            if input_sharpedge_pc_size == 0:
                input_sharpedge_surface_feats = torch.zeros(B, 0, self.point_feats,
                                                            dtype=input_random_surface_feats.dtype).to(pc.device)
                query_sharpedge_feats = torch.zeros(B, 0, self.point_feats, dtype=query_random_feats.dtype).to(
                    pc.device)
            else:
                input_sharpedge_surface_feats = sharpedge_surface_feats[:, idx_sharpedge_pc, :]
                flatten_input_sharpedge_surface_feats = input_sharpedge_surface_feats.view(B * input_sharpedge_pc_size,
                                                                                           -1)
                query_sharpedge_feats = flatten_input_sharpedge_surface_feats[idx_query_sharpedge].view(B, -1,
                                                                                                        flatten_input_sharpedge_surface_feats.shape[
                                                                                                            -1])

            query_feats = torch.cat([query_random_feats, query_sharpedge_feats], dim=1)
            input_feats = torch.cat([input_random_surface_feats, input_sharpedge_surface_feats], dim=1)

            if self.normal_pe:
                query_normal_pe = self.fourier_embedder(query_feats[..., :3])
                input_normal_pe = self.fourier_embedder(input_feats[..., :3])
                query_feats = torch.cat([query_normal_pe, query_feats[..., 3:]], dim=-1)
                input_feats = torch.cat([input_normal_pe, input_feats[..., 3:]], dim=-1)

            query = torch.cat([query, query_feats], dim=-1)
            data = torch.cat([data, input_feats], dim=-1)

        if input_sharpedge_pc_size == 0:
            query_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)
            input_sharpedge_pc = torch.zeros(B, 1, D).to(pc.device)

        # print(f'query_pc: {query_pc.shape}')
        # print(f'input_pc: {input_pc.shape}')
        # print(f'query_random_pc: {query_random_pc.shape}')
        # print(f'input_random_pc: {input_random_pc.shape}')
        # print(f'query_sharpedge_pc: {query_sharpedge_pc.shape}')
        # print(f'input_sharpedge_pc: {input_sharpedge_pc.shape}')

        return query.view(B, -1, query.shape[-1]), data.view(B, -1, data.shape[-1]), [query_pc, input_pc,
                                                                                      query_random_pc, input_random_pc,
                                                                                      query_sharpedge_pc,
                                                                                      input_sharpedge_pc]

    def forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """
        query, data, pc_infos = self.sample_points_and_latents(pc, feats)
        

        query = self.input_proj(query)
        query = query
        data = self.input_proj(data)
        data = data
        
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)
        latents = self.ln_post(latents)

        return latents, pc_infos
    
    
    

class TwoCrossAttentionEncoder(nn.Module):
    # 改造于CrossAttentionDecoder
    def __init__(self,
                 use_downsample:bool,
                 num_latents: int,
                 embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 qkv_bias: bool = True):

        super().__init__()
        self.use_downsample=use_downsample
        self.num_latents = num_latents

        if not self.use_downsample:
            self.query = nn.Parameter(torch.randn((num_latents, width)) * 0.02)


        self.embedder = embedder
        self.input_proj_surface = nn.Linear(self.embedder.out_dim*2, width)
        self.input_proj_sharp = nn.Linear(self.embedder.out_dim*2, width)

        self.cross_attn = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
        )

        self.cross_attn1 = ResidualCrossAttentionBlock(
            width=width,
            heads=heads,
            qkv_bias=qkv_bias,
        )

        self.self_attn = Transformer(
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            qkv_bias=qkv_bias,
        )

        self.ln_post = nn.LayerNorm(width)


    def forward(self, coarse_pc, sharp_pc, coarse_feats , sharp_feats, split):
        bs, N_coarse, D_coarse = coarse_pc.shape
        bs, N_sharp, D_sharp = sharp_pc.shape
        # 点云1编码
        coarse_data = self.embedder(coarse_pc)
        coarse_feats = self.embedder(coarse_feats)
        coarse_data = torch.cat([coarse_data, coarse_feats], dim=-1)
        coarse_data = self.input_proj_surface(coarse_data)
        # 点云2编码
        sharp_data = self.embedder(sharp_pc)
        sharp_feats = self.embedder(sharp_feats)
        sharp_data = torch.cat([sharp_data, sharp_feats], dim=-1)
        sharp_data = self.input_proj_sharp(sharp_data)

        if self.use_downsample:
            ###### fps
            #tokens = np.array([128.0,256.0,384.0,512.0,640.0,1024.0,2048.0])
            # tokens = np.array([128.0,256.0,384.0])
            # coarse_ratios = tokens/ N_coarse
            # sharp_ratios = tokens/ N_sharp
            # if split =='val':
            #     #probabilities = np.array([0,0,0,0,0,1,0])
            #     probabilities = np.array([0,1,0])
            # elif split =='train':
            #     #probabilities = np.array([ 0.1,0.1,0.1,0.1,0.1,0.3,0.2])
            #     probabilities = np.array([ 0.2,0.5,0.3])
            # ratio_coarse = np.random.choice(coarse_ratios, size=1, p=probabilities)[0]
            # index = np.where(coarse_ratios == ratio_coarse)[0]
            # ratio_sharp = sharp_ratios[index].item()
            ratio_coarse=4096.0 / N_coarse  # 固定
            ratio_sharp =4096.0 / N_sharp

            flattened = coarse_pc.view(bs*N_coarse, D_coarse)
            batch = torch.arange(bs).to(coarse_pc.device)
            batch = torch.repeat_interleave(batch, N_coarse)
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_coarse)
            query_coarse = coarse_data.view(bs*N_coarse, -1)[idx].view(bs, -1, coarse_data.shape[-1])

            flattened = sharp_pc.view(bs*N_sharp, D_sharp)
            batch = torch.arange(bs).to(sharp_pc.device)
            batch = torch.repeat_interleave(batch, N_sharp)
            pos = flattened
            idx = fps(pos, batch, ratio=ratio_sharp)
            query_sharp = sharp_data.view(bs*N_sharp, -1)[idx].view(bs, -1, sharp_data.shape[-1])

            query = torch.cat([query_coarse, query_sharp], dim=1)
            #print('query shape',f'{query.shape}')
        else:
            query = self.query
            query = repeat(query, "m c -> b m c", b=bs)


        latents_coarse = self.cross_attn(query, coarse_data)
        latents_sharp=  self.cross_attn1(query, sharp_data)
        latents = latents_coarse + latents_sharp

        latents = self.self_attn(latents)
        latents = self.ln_post(latents)

        return latents

