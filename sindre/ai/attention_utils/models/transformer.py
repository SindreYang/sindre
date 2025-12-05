
import torch
import torch.nn as nn
from sindre.ai.attention_utils.multihead_attention import ResidualAttentionBlock

class Transformer(nn.Module):
    """Transformer 模型，由多层 `ResidualAttentionBlock` 堆叠而成。

    Args:
        n_ctx (int): 上下文长度（序列长度）。
        width (int): 输入特征维度。
        layers (int): 残差注意力块的层数。
        heads (int): 注意力头数量。
        qkv_bias (bool): 是否在 QKV 投影中添加偏置项，默认为 True。
        norm_layer (nn.Module): 归一化层类型，默认为 `nn.LayerNorm`。
        qk_norm (bool): 是否对 Q/K 进行归一化，默认为 False。
        drop_path_rate (float): DropPath 的丢弃概率，默认为 0.0。

    Shape:
        - 输入 x: (bs, n_ctx, width)
        - 输出: (bs, n_ctx, width)
    """

    def __init__(self,
                 *,
                 n_ctx: int,
                 width: int,
                 layers: int,
                 heads: int,
                 qkv_bias: bool = True,
                 norm_method: str = "LayerNorm",
                 atten_method: str = "SDPA",
                 drop_path_rate: float = 0.0):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        # 初始化多层残差块
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                n_ctx=n_ctx,
                width=width,
                heads=heads,
                qkv_bias=qkv_bias,
                norm_method=norm_method,
                atten_method=atten_method,
                drop_path_rate=drop_path_rate
            )
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)  # 逐层计算
        return x
