"""
公共层

"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from typing import Tuple, List, Union, Optional
import torch.nn.functional as F
from einops import rearrange
def attention(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              method="SDPA"):
    """

    Args:
        q: 查询张量，形状为 (B, H, L, D)
            - B: batch size，H: 注意力头数，L: 查询序列长度，D: 每个头的维度
        k: 键张量，形状为 (B, H, S, D)
            - S: 键/值序列长度（自注意力时S=L，交叉注意力时S≠L）
        v: 值张量，形状为 (B, H, S, D)
        method: 注意力实现方法：
            - SDPA: torch内置scaled_dot_product_attention（推荐，高效）
            - SAGE: sageattention实现
            - FLASH: flash_attn变长序列实现（需安装flash-attn）

    Returns:
         注意力输出张量，形状为 (B, L, H*D)
    """
    if method == "SDPA":
        x=F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x
    elif method == "SAGE":
        from sageattention import sageattn
        x= F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x
    elif method == "FLASH":
        from flash_attn import flash_attn_func
        if q.size(-1) > 256:
            raise ValueError("FlashAttention only supports head dimension up to 256.")
        x=flash_attn_func(q,k,v,causal=False)
        x = rearrange(x, "B H L D -> B L (H D)")
        return x
    else:
        RuntimeError("只支持SDPA(torch内置),SAGE(sageattention),FLASH(flash_attn)")


class RMSNorm(torch.nn.Module):
    """均方根层归一化

    Args:
        dim: 归一化维度
    """
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    """注意力机制中查询和键的归一化

    Args:
        dim: 查询/键的向量维度
        method: ”RMSNorm"(均方根层归一化)/"LayerNorm"(层归一化）/"Identity"(单位化(
    """
    def __init__(self, dim: int,method="LayerNorm"):
        super().__init__()
        if method == "RMSNorm":
            self.query_norm = RMSNorm(dim)
            self.key_norm = RMSNorm(dim)

        elif method == "LayerNorm":
            self.query_norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
            self.key_norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        elif method == "Identity":
            self.query_norm =nn.Identity()
            self.key_norm =nn.Identity()
        else:
            raise ValueError(f"不支持归一化方法")


    def forward(self, q: torch.Tensor, k:  torch.Tensor) -> Tuple[ torch.Tensor,  torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q,k




class DropPath(nn.Module):
    """随机深度（Stochastic Depth）模块，用于在残差块的主路径上随机丢弃样本路径。

    该模块通过以概率 `drop_prob` 将输入张量置零（跳过当前残差块），同时根据 `scale_by_keep`
    决定是否缩放输出值以保持期望不变。常用于正则化深层网络（如ResNet、Vision Transformer）。

    Notes:

        它与作者为 EfficientNet 等网络创建的 DropConnect 实现类似，但原来的名称具有误导性，
        因为 “Drop Connect” 在另一篇论文中是一种不同形式的丢弃技术。
        作者选择将层和参数名称更改为 “drop path”，
        而不是将 DropConnect 作为层名并使用 “survival rate（生存概率）” 作为参数。
        [https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956]


    Args:
        drop_prob (float): 路径丢弃概率，取值范围 [0, 1)，默认为 0（不丢弃）。
        scale_by_keep (bool): 若为 True，保留路径时会进行缩放补偿（除以 `1 - drop_prob`），
            以保持输出的期望值不变，默认为 True。

    Attributes:
        drop_prob (float): 继承自 Args 的路径丢弃概率。
        scale_by_keep (bool): 继承自 Args 的缩放开关。

    Example:
        >>> x = torch.randn(2, 3, 16, 16)
        >>> drop_path = DropPath(drop_prob=0.2)
        >>> train_output = drop_path(x)  # 训练时随机丢弃路径
        >>> drop_path.eval()
        >>> eval_output = drop_path(x)   # 推理时直接返回原值
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        """初始化方法，配置丢弃概率和缩放开关"""
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，训练时随机丢弃路径，推理时直接返回输入。

        具体实现逻辑：
        1. 若 `drop_prob=0` 或处于推理模式，直接返回输入。
        2. 生成与输入张量 `x` 的 batch 维度对齐的随机二值掩码。
        3. 根据 `scale_by_keep` 决定是否对保留路径的样本进行缩放补偿。

        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, ...]

        Returns:
            torch.Tensor: 输出张量，训练时可能被部分置零，形状与输入一致。
        """
        # 若无需丢弃或处于推理模式，直接返回原值
        if self.drop_prob == 0. or not self.training:
            return x

        # 计算保留概率并生成随机二值掩码
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适配不同维度（Conv2D/3D, Linear等）
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

        # 缩放补偿：保持输出的期望值 E[output] = x（仅训练时生效）
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)

        return x * random_tensor  # 随机置零部分样本的路径

    def extra_repr(self) -> str:
        """用于打印模块的附加信息"""
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'



class MLP(nn.Module):
    """多层感知机（MLP）模块，包含扩展投影、激活函数、收缩投影和可选的 DropPath 正则化。

    Args:
        width (int): 输入特征维度。
        output_width (int, optional): 输出特征维度，默认为 None（与输入相同）。
        drop_path_rate (float): DropPath 的路径丢弃概率，默认为 0.0（不启用）。

    Shape:
        - 输入 x: (..., width)
        - 输出: (..., output_width or width)
    """

    def __init__(self, *, width: int, output_width: int = None, drop_path_rate: float = 0.0):
        super().__init__()
        self.width = width
        # 扩展层：将维度扩展为 4 倍以增加非线性能力
        self.c_fc = nn.Linear(width, width * 4)
        # 收缩层：将维度投影回目标输出维度（默认与输入相同）
        self.c_proj = nn.Linear(width * 4, output_width if output_width is not None else width)
        self.gelu = nn.GELU()  # GELU 激活函数
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播流程：扩展 -> 激活 -> 收缩 -> DropPath（若启用）"""
        x = self.c_fc(x)      # 扩展维度
        x = self.gelu(x)      # 激活函数
        x = self.c_proj(x)    # 投影回目标维度
        x = self.drop_path(x) # 应用 DropPath（训练时随机丢弃路径）
        return x






class GEGLU(nn.Module):
    """
    GeGLU activation function.

    Taken from 3DShape2VecSet, Zhang et al., SIGGRAPH23.
    https://github.com/1zb/3DShape2VecSet/blob/master/models_ae.py
    """

    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

    def __repr__(self):
        return f"GEGLU()"


@dataclass
class ModulationOut:
    """调制参数输出容器"""
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    """基于条件向量的动态特征调制

    Args:
        dim: 调制参数的维度
        double: 是否生成两组调制参数
    """
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: torch.Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class LastProjectLayer(nn.Module):
    """带自适应调制的最终投影层

    Args:
        hidden_size: 输入维度
        out_channels: 输出通道维度
    """
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        return self.linear(x)




class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) layer.

    将隐向量中编码的风格迁移到输入张量中。
    首先对输入张量进行归一化处理（"白化"），然后使用从隐向量生成的参数
    进行反归一化，从而将风格信息编码到输入张量中。

    原始论文: https://arxiv.org/abs/1703.06868
    基于实现: https://github.com/SiskonEmilia/StyleGAN-PyTorch

    Attributes:
    norm: 归一化层，用于对输入图像进行"白化"处理。
    默认为InstanceNorm2d，也可以是其他归一化模块。
    """

    def __init__(self, n_channels):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(n_channels)

    def forward(self, image, style):
        factor, bias = style.view(style.size(0), style.size(1), 1, 1).chunk(2, dim=1)
        result = self.norm(image) * factor + bias
        return result


class binarize(torch.autograd.Function):
    """
    自定义二值化操作的PyTorch函数实现。
    继承自torch.autograd.Function，支持自动求导。
    功能：将输入张量根据阈值转换为二值张量（0或1）。
    """
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        """
        前向传播：将输入张量根据阈值二值化。

        Args:
            ctx: 上下文对象
            x: 输入张量，可以是任意形状
            threshold: 二值化阈值，默认值为0.5

        Returns:
            binarized: 二值化后的张量，与x形状相同，值为0或1
        """
        with torch.no_grad():
            # 大于阈值的元素设为1，否则设为0
            binarized = (x > threshold).float()
            # 标记二值化结果为不可微分
            ctx.mark_non_differentiable(binarized)

            return binarized

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：计算输入的梯度。

        Args:
            ctx: 上下文对象
            grad_output: 输出的梯度，形状同输出

        Returns:
            grad_inputs: 输入x的梯度，形状同x（直通估计器）
        """
        grad_inputs = None

        # 如果需要计算输入x的梯度
        if ctx.needs_input_grad[0]:
            # 输入的梯度直接等于输出的梯度（直通估计器）
            grad_inputs = grad_output.clone()

        return grad_inputs

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)