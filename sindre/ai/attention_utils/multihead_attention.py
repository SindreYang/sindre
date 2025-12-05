
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional
from einops import rearrange
from sindre.ai.layers import MLP,attention,RMSNorm,QKNorm,DropPath

class QKVMultiheadAttention(nn.Module):
    """基于 QKV 拼接的多头自注意力计算模块。

    将输入的拼接 QKV 张量分割为独立的 Q/K/V，并应用缩放点积注意力。

    Args:
        heads (int): 注意力头数量。
        n_ctx (int): 上下文长度（序列长度）。
        width (int, optional): 输入特征维度，默认为 None。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。

    Shape:
        - 输入 qkv: (bs, n_ctx, width * 3)  # Q/K/V 拼接后的张量
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self,
                 heads: int,
                 n_ctx: int,
                 width=None,
                 norm_method: str = "LayerNorm",
                 atten_method: str = "SDPA"):
        super().__init__()
        self.atten_method=atten_method
        self.heads = heads
        self.n_ctx = n_ctx
        self.qknorm = QKNorm(width // heads,method=norm_method)

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3  # 计算每个注意力头的通道数
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)  # 分割 Q/K/V

        # 归一化处理
        q,k = self.qknorm(q,k)

        # 重排维度并计算注意力
        q, k, v = map(lambda t: rearrange(t, 'b n h d -> b h n d', h=self.heads), (q, k, v))
        out =attention(q, k, v,method=self.atten_method)
        return out



class MultiheadAttention(nn.Module):
    """多头自注意力模块，包含 QKV 投影和注意力计算。

    Args:
        n_ctx (int): 上下文长度（序列长度）。
        width (int): 输入/输出特征维度。
        heads (int): 注意力头数量。
        qkv_bias (bool): 是否在 QKV 投影中添加偏置项，默认为 True。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0（不启用）。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self,
                 n_ctx: int,
                 width: int,
                 heads: int,
                 qkv_bias: bool=True,
                 norm_method: str = "LayerNorm",
                 atten_method: str = "SDPA",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads

        # 初始化投影层
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias)  # QKV 拼接投影
        self.c_proj = nn.Linear(width, width)  # 输出投影
        self.attention = QKVMultiheadAttention(  # 注意力计算模块
            heads=heads, n_ctx=n_ctx, width=width, norm_method=norm_method, atten_method=atten_method
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        x = self.c_qkv(x)  # 投影 QKV
        x = self.attention(x)  # 计算自注意力
        x = self.drop_path(self.c_proj(x))  # 输出投影 + DropPath
        return x




class SelfAttention(nn.Module):
    """带QK归一化的多头自注意力

    Args:
        dim: 输入维度
        num_heads: 注意力头数
        qkv_bias: 是否在qkv投影中使用偏置
    """
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            norm_method: str = "RMSNorm",
            atten_method: str = "SAGE",
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.atten_method = atten_method

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim,method=norm_method)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k)
        x = attention(q, k, v,method=self.atten_method)
        x = self.proj(x)
        return x


class ResidualAttentionBlock(nn.Module):
    """残差自注意力块，包含多头自注意力和 MLP 子模块。

    结构：LN -> Self-Attention -> Add -> LN -> MLP -> Add

    Args:
        n_ctx (int): 上下文长度（序列长度）。
        width (int): 输入特征维度。
        heads (int): 注意力头数量。
        qkv_bias (bool): 是否在 QKV 投影中添加偏置项，默认为 True。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self,
                 n_ctx: int,
                 width: int,
                 heads: int,
                 qkv_bias: bool = True,
                 norm_method: str = "LayerNorm",
                 atten_method: str = "SDPA",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.attn = MultiheadAttention(  # 自注意力模块
            n_ctx=n_ctx, width=width, heads=heads, qkv_bias=qkv_bias,
            norm_method=norm_method, atten_method=atten_method, drop_path_rate=drop_path_rate
        )
        self.ln_1 = nn.LayerNorm(width, eps=1e-6)  # 自注意力前归一化
        self.mlp = MLP(width=width, drop_path_rate=drop_path_rate)  # MLP 模块
        self.ln_2 = nn.LayerNorm(width, eps=1e-6)  # MLP 前归一化

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))  # 残差连接：自注意力
        x = x + self.mlp(self.ln_2(x))    # 残差连接：MLP
        return x




