import torch
from torch import nn

from sindre.ai.attention_utils.cross_attention import ResidualCrossAttentionBlock
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

