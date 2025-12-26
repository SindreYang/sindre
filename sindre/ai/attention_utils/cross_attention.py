import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional
from einops import rearrange

from sindre.ai.layers import MLP,attention,QKNorm





class QKVMultiheadCrossAttention(nn.Module):
    """基于查询（Query）、键值对（Key-Value）的多头交叉注意力计算模块。

    通过将输入的 `q` 和 `kv` 分割为多头，应用缩放点积注意力（Scaled Dot-Product Attention），
    并可选对 Q/K 进行归一化处理。

    Args:
        heads (int): 注意力头的数量。
        n_data (int, optional): 键值对数据的数量（上下文长度），默认为 None。
        width (int, optional): 输入特征的维度，默认为 None。


    Attributes:
        heads (int): 继承自 Args 的注意力头数量。


    Shape:
        - 输入 q: (bs, n_ctx, width)
        - 输入 kv: (bs, n_data, width * 2)  # 包含键和值拼接后的张量
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self,
                 *,
                 heads: int,
                 width=None,
                 norm_method: str = "LayerNorm",
                 atten_method: str = "SDPA"):
        super().__init__()
        self.heads = heads
        self.atten_method = atten_method
        # 初始化 Q/K 归一化层
        self.qknorm = QKNorm(width // heads,method=norm_method)

    def forward(self, q, kv):
        # 分割多头并计算注意力
        bs, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2  # 计算每个注意力头的通道数
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)  # 分割键和值

        # 归一化处理
        q,k = self.qknorm(q,k)
        # 重排维度并计算注意力
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out =attention(q, k, v,method=self.atten_method)
        return out



class MultiheadCrossAttention(nn.Module):
    """多头交叉注意力模块，包含线性投影和注意力计算。

    将输入 `x` 和 `data` 分别投影为 Q 和 K/V，并通过 `QKVMultiheadCrossAttention` 计算交叉注意力。

    Args:
        width (int): 输入/输出特征维度。
        heads (int): 注意力头的数量。
        qkv_bias (bool): 是否在 Q/K/V 投影中添加偏置项，默认为 True。
        data_width (int, optional): 输入数据 `data` 的特征维度，默认为 None（同 width）。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输入 data: (bs, n_data, data_width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self, *,
                 width: int,
                 heads: int,
                 qkv_bias: bool = True,
                 data_width: Optional[int] = None,
                 norm_method: str = "LayerNorm",
                 atten_method: str = "SDPA"):
        super().__init__()
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width

        # 初始化 Q/K/V 投影层
        self.c_q = nn.Linear(width, width, bias=qkv_bias)  # 查询投影
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias)  # 键值投影
        self.c_proj = nn.Linear(width, width)  # 输出投影

        # 注意力计算模块
        self.attention = QKVMultiheadCrossAttention(
            heads=heads, width=width, norm_method=norm_method, atten_method=atten_method
        )

    def forward(self, x, data):
        x = self.c_q(x)  # 投影查询向量
        data = self.c_kv(data)  # 投影键值对
        x = self.attention(x, data)  # 计算交叉注意力
        x = self.c_proj(x)  # 投影回原始维度
        return x





class ResidualCrossAttentionBlock(nn.Module):
    """残差交叉注意力块，包含多头交叉注意力和 MLP 子模块。

    结构：LN -> Cross-Attention -> Add -> LN -> MLP -> Add

    Args:
        n_data (int, optional): 键值对数据的数量，默认为 None。
        width (int): 输入特征维度。
        heads (int): 注意力头的数量。
        data_width (int, optional): 输入数据 `data` 的特征维度，默认为 None（同 width）。
        qkv_bias (bool): 是否在 Q/K/V 投影中添加偏置项，默认为 True。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输入 data: (bs, n_data, data_width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self,
                 width: int,
                 heads: int,
                 data_width: Optional[int] = None,
                 qkv_bias: bool = True,
                 norm_method: str = "LayerNorm",
                 atten_method: str = "SDPA"):
        super().__init__()
        if data_width is None:
            data_width = width

        # 初始化子模块
        self.attn = MultiheadCrossAttention(
            width=width, heads=heads, data_width=data_width,
            qkv_bias=qkv_bias, norm_method=norm_method, atten_method=atten_method
        )
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)  # 输入归一化
        self.ln_2 = nn.LayerNorm(data_width, eps=1e-6)  # 数据归一化
        self.ln_3 = nn.LayerNorm(width, eps=1e-6)  # MLP 前归一化
        self.mlp = MLP(width=width)  # 多层感知机

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        # 残差连接：交叉注意力
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        # 残差连接：MLP
        x = x + self.mlp(self.ln_3(x))
        return x




