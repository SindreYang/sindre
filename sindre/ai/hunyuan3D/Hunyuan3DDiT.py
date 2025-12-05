"""Hunyuan3D 扩散变换器(DiT)的多模态生成实现

模块包含以下核心实现:
- 图文联合处理的双流Transformer架构
- 基于扩散时间步的条件调制机制
- 3D感知的位置编码方案

典型用法示例:

    # 初始化模型
    model = Hunyuan3DDiT(
        in_channels=64,
        context_in_dim=1536,
        hidden_size=1024,
        num_heads=16,
        depth=16,
        depth_single_blocks=32
    )

    # 前向传播示例
    x = torch.randn(2, 16, 64)   # 潜在表示
    t = torch.rand(2)            # 扩散时间步
    context = torch.randn(2, 77, 1536)  # 文本嵌入

    output = model(x, t, contexts={'main': context})
"""



from typing import Optional, List, Tuple
import torch
from einops import rearrange
from torch import nn
from sindre.ai.attention_utils.multihead_attention import SelfAttention
from sindre.ai.embedder import TimestepEmbedder
from sindre.ai.layers import LastProjectLayer, Modulation, attention, QKNorm




class DoubleStreamBlock(nn.Module):
    """图文交互的双流Transformer块

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度与hidden_size的比例
        qkv_bias: 是否在qkv投影中使用偏置
    """
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_ratio: float,
            qkv_bias: bool = False,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # 图像流组件
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # 文本流组件
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 图像流处理
        img_mod1, img_mod2 = self.img_mod(vec)
        img_modulated = (1 + img_mod1.scale) * self.img_norm1(img) + img_mod1.shift
        img_q, img_k, img_v = rearrange(self.img_attn.qkv(img_modulated),
                                        "B L (K H D) -> K B H L D", K=3, H=self.num_heads)[:3]
        img_q, img_k = self.img_attn.norm(img_q, img_k)
        img_q, img_k = img_q.to(img_v), img_k.to(img_v)

        # 文本流处理
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        txt_modulated = (1 + txt_mod1.scale) * self.txt_norm1(txt) + txt_mod1.shift
        txt_q, txt_k, txt_v = rearrange(self.txt_attn.qkv(txt_modulated),
                                        "B L (K H D) -> K B H L D", K=3, H=self.num_heads)[:3]
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k)
        txt_q, txt_k = txt_q.to(txt_v), txt_k.to(txt_v)

        # 跨模态注意力
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        attn = attention(q, k, v)

        # 分割并处理输出
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        # 带残差连接的特征更新
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        return img, txt


class SingleStreamBlock(nn.Module):
    """带并行注意力与MLP路径的单流Transformer块

    Args:
        hidden_size: 隐藏层维度
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度与hidden_size的比例
        qk_scale: 注意力logits的可选缩放因子
    """
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            qk_scale: Optional[float] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(head_dim,method="RMSNorm")
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift

        # 并行处理注意力与MLP路径
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k)
        q, k =q.to(v), k.to(v)

        attn = attention(q, k, v)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class Hunyuan3DDiT(nn.Module):
    """Hunyuan3D扩散变换器主模型

    Args:
        in_channels: 输入潜在通道数
        context_in_dim: 文本嵌入维度
        hidden_size: 模型隐藏维度
        mlp_ratio: MLP扩展比例
        num_heads: 注意力头数
        depth: 双流块数量
        depth_single_blocks: 单流块数量
        axes_dim: 各轴的位置编码维度
        theta: 位置编码频率基数
        qkv_bias: 是否在QKV投影中使用偏置
        time_factor: 时间步嵌入缩放因子
        guidance_embed: 是否使用引导嵌入
        ckpt_path: 预训练权重的检查点路径

    Example:
        >>> model = Hunyuan3DDiT(
        ...     in_channels=64,
        ...     context_in_dim=1536,
        ...     hidden_size=1024,
        ...     num_heads=16,
        ...     depth=16,
        ...     depth_single_blocks=32
        ... )
        >>> x = torch.randn(2, 16, 64)  # 潜在编码批次
        >>> t = torch.rand(2)           # 随机时间步
        >>> context = torch.randn(2, 77, 1536)  # 文本嵌入
        >>> output = model(x, t, contexts={'main': context})
        >>> print(output.shape)
        torch.Size([2, 16, 64])
    """
    def __init__(
            self,
            in_channels: int = 64,
            context_in_dim: int = 1536,
            hidden_size: int = 1024,
            mlp_ratio: float = 4.0,
            num_heads: int = 16,
            depth: int = 16,
            depth_single_blocks: int = 32,
            theta: int = 10_000,
            qkv_bias: bool = True,
            ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.depth = depth
        self.depth_single_blocks = depth_single_blocks
        self.theta = theta
        self.qkv_bias = qkv_bias
        self.out_channels = self.in_channels
        # 维度约束验证
        if hidden_size % num_heads != 0:
            raise ValueError(f"隐藏维度{hidden_size}必须能被注意力头数{num_heads}整除")

        # 初始化核心组件
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.latent_in = nn.Linear(in_channels, hidden_size)
        self.cond_in = nn.Linear(context_in_dim, hidden_size)

        # 时间步层
        self.TimestepEmbedder=TimestepEmbedder(256, hidden_size)

        # 构建Transformer块
        self.double_blocks = nn.ModuleList([
            DoubleStreamBlock(hidden_size, num_heads, mlp_ratio, qkv_bias)
            for _ in range(depth)
        ])
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth_single_blocks)
        ])

        # 最终输出投影
        self.final_layer = LastProjectLayer(hidden_size, self.out_channels)

        # 加载预训练权重
        if ckpt_path:
            self._load_pretrained_weights(ckpt_path)

    def _load_pretrained_weights(self, ckpt_path: str):
        """从检查点加载预训练权重"""
        print(f'正在从{ckpt_path}加载预训练权重')
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get('state_dict', ckpt)  # 处理deepspeed检查点

        # 适配检查点键名
        final_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('_forward_module.', '').replace('model.', '')
            final_state_dict[new_k] = v

        load_info = self.load_state_dict(final_state_dict, strict=False)
        print(f'Unexpected keys: {load_info.unexpected_keys}')
        print(f'Missing keys: {load_info.missing_keys}')

    def forward(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            contexts: dict,
    ) -> torch.Tensor:
        """扩散变换器的前向传播

        Args:
            x: 输入潜在张量，形状 (B, L, C)
            t: 时间步张量，形状 (B,)
            contexts: 包含'main'键下文本嵌入的字典
        Returns:
            与输入同形状的预测噪声张量
        """
        # 处理输入条件
        cond = self.cond_in(contexts['main'])
        latent = self.latent_in(x)
        dtype = x.dtype

        # 时间与引导嵌入
        vec=self.TimestepEmbedder(t,dtype=dtype)

        # 双流处理
        for block in self.double_blocks:
            latent, cond = block(img=latent, txt=cond, vec=vec)

        # 单流处理
        combined = torch.cat((cond, latent), 1)
        for block in self.single_blocks:
            combined = block(combined, vec=vec)

        # 最终投影
        return self.final_layer(combined[:, cond.shape[1]:], vec)




if __name__ =="__main__":
    # 初始化模型
    model = Hunyuan3DDiT(
        in_channels=64,
        context_in_dim=1536,
        hidden_size=1024,
        num_heads=16,
        depth=16,
        depth_single_blocks=32
    )

    # 前向传播示例
    x = torch.randn(2, 16, 64)   # 潜在表示
    t = torch.rand(2)            # 扩散时间步
    context = torch.randn(2, 77, 1536)  # 文本嵌入

    output = model(x, t, contexts={'main': context})
    print(output.shape)
    assert x.shape ==output.shape , "输出应该跟输入x形状一致"