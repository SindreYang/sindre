from typing import Optional

import torch
from einops import repeat
from torch import nn

from sindre.ai.attention_utils.cross_attention import ResidualCrossAttentionBlock
from sindre.ai.attention_utils.models.transformer import Transformer
from sindre.ai.embedder import FourierEmbedder


class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 qkv_bias: bool = True,
                 use_ln_post: bool = False,
                 ):

        super().__init__()
        self.num_latents = num_latents

        self.query = nn.Parameter(torch.randn(num_latents, width) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width)
        self.cross_attn = ResidualCrossAttentionBlock(
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

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width)
        else:
            self.ln_post = None

    def forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc


