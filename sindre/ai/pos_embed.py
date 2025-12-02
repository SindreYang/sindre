
# --------------------------------------------------------
# 位置嵌入工具
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
from typing import Union, Optional, Tuple



class LabelEmbedding(nn.Module):
    """
    将类别标签嵌入为向量表示。同时支持标签dropout功能;

    场景：
        1. 用于无分类器引导（classifier-free guidance）训练；
        2. 在训练时随机丢弃部分样本的类别标签（替换为特殊空标签），让模型同时学习 "有条件生成" 和 "无条件生成" 的能力，避免模型过度依赖标签而导致生成结果单一;


    Args:
        num_classes (`int`): 类别总数。
        hidden_size (`int`): 嵌入向量的维度大小。
        dropout_prob (`float`): 标签被dropout（丢弃）的概率。
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        # 当dropout概率大于0时，需要额外添加一个"空标签"的嵌入（用于表示被丢弃的标签）
        use_cfg_embedding = dropout_prob > 0
        # 嵌入表大小 = 类别数 + （是否需要空标签嵌入）
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes  # 保存类别总数
        self.dropout_prob = dropout_prob  # 保存dropout概率

    def token_drop(self, labels, force_drop_ids=None):
        """
        对标签执行dropout操作，为无分类器引导（CFG）提供支持。
        被dropout的标签会被替换为一个特殊的"空标签ID"（即num_classes对应的索引）。

        参数:
            labels (`torch.LongTensor`): 输入标签张量，形状为 [batch_size] 或 [batch_size, ...]
            force_drop_ids (`list` 或 `None`): 强制指定哪些样本需要dropout（优先级高于随机dropout）。
                若为list，元素为0或1，1表示对应样本的标签需要被dropout；若为None，则按概率随机dropout。

        返回:
            `torch.LongTensor`: 处理后的标签张量（被dropout的位置已替换为特殊ID）
        """
        if force_drop_ids is None:
            # 随机生成dropout掩码：每个样本以dropout_prob的概率被选中dropout
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            # 强制使用指定的dropout掩码：将force_drop_ids转换为布尔张量（1→True，0→False）
            drop_ids = torch.tensor(force_drop_ids == 1, device=labels.device)

        # 对drop_ids扩展维度，确保与labels的形状匹配（支持多维labels输入）
        drop_ids = drop_ids.view(-1, *([1] * (len(labels.shape) - 1)))
        # 替换标签：被dropout的位置→num_classes（特殊空标签），否则保持原标签
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels: torch.LongTensor, force_drop_ids=None):
        """
        前向传播：将标签转换为嵌入向量。

        参数:
            labels (`torch.LongTensor`): 输入标签张量，形状为 [batch_size] 或 [batch_size, seq_len]
            force_drop_ids (`list` 或 `None`): 强制dropout的样本索引列表（可选）

        返回:
            `torch.FloatTensor`: 标签的嵌入向量，形状为 [batch_size, hidden_size] 或 [batch_size, seq_len, hidden_size]
        """
        # 判断是否需要启用dropout（训练模式且dropout概率>0，或强制指定了dropout掩码）
        use_dropout = self.dropout_prob > 0
        if (self.training and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        # 从嵌入表中查询标签对应的嵌入向量
        embeddings = self.embedding_table(labels)
        return embeddings



class TextImageProjection(nn.Module):
    """
    文本-图像投影融合模块：将文本嵌入和图像嵌入转换到同一维度空间，并拼接融合。
    核心作用是对齐文本与图像的特征维度，生成可用于跨模态注意力计算的联合特征。


    参数:
        text_embed_dim (`int`, 可选, 默认=1024): 输入文本嵌入的维度。
        image_embed_dim (`int`, 可选, 默认=768): 输入图像嵌入的维度。
        cross_attention_dim (`int`, 可选, 默认=768): 跨模态注意力机制使用的目标特征维度（统一后的维度）。
        num_image_text_embeds (`int`, 可选, 默认=10): 图像嵌入被拆分/扩展后的特征序列长度（将单张图像的全局嵌入转换为多token序列）。
    """

    def __init__(
            self,
            text_embed_dim: int = 1024,
            image_embed_dim: int = 768,
            cross_attention_dim: int = 768,
            num_image_text_embeds: int = 10,
    ):
        super().__init__()

        # 保存图像特征扩展后的序列长度
        self.num_image_text_embeds = num_image_text_embeds

        # 图像嵌入投影层：将图像嵌入（维度image_embed_dim）映射到 (num_image_text_embeds * cross_attention_dim) 维度
        # 后续会拆分为 num_image_text_embeds 个长度为 cross_attention_dim 的特征token
        self.image_embeds = nn.Linear(image_embed_dim, self.num_image_text_embeds * cross_attention_dim)

        # 文本嵌入投影层：将文本嵌入（维度text_embed_dim）映射到 cross_attention_dim 维度（与图像特征对齐）
        self.text_proj = nn.Linear(text_embed_dim, cross_attention_dim)

    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor):
        """
        前向传播：对齐文本和图像嵌入的维度，并拼接为联合特征序列。

        参数:
            text_embeds (`torch.Tensor`): 文本嵌入张量，形状为 [batch_size, text_seq_len, text_embed_dim]
                （text_seq_len为文本token的序列长度，如句子的单词数）
            image_embeds (`torch.Tensor`): 图像嵌入张量，形状为 [batch_size, image_embed_dim]
                （通常是图像全局特征，如CLIP的image encoder输出的单向量）

        返回:
            `torch.Tensor`: 文本-图像联合特征张量，形状为 [batch_size, (num_image_text_embeds + text_seq_len), cross_attention_dim]
                （第一部分为图像扩展后的特征token，第二部分为文本投影后的特征token）
        """
        batch_size = text_embeds.shape[0]  # 获取批次大小

        # 1. 图像嵌入处理：投影+拆分为多token序列
        # 先通过全连接层将图像全局嵌入映射到目标总维度
        image_text_embeds = self.image_embeds(image_embeds)
        # 拆分维度：从 [batch_size, num_image_text_embeds * cross_attention_dim] 拆分为 [batch_size, num_image_text_embeds, cross_attention_dim]
        # 即将单张图像的全局特征转换为 num_image_text_embeds 个特征token（适配序列式跨模态注意力）
        image_text_embeds = image_text_embeds.reshape(batch_size, self.num_image_text_embeds, -1)

        # 2. 文本嵌入处理：维度对齐投影
        # 将文本嵌入从原始维度 text_embed_dim 投影到 cross_attention_dim，与图像特征维度一致
        text_embeds = self.text_proj(text_embeds)  # 输出形状：[batch_size, text_seq_len, cross_attention_dim]

        # 3. 特征拼接：在序列维度（dim=1）拼接图像特征token和文本特征token
        # 最终得到联合特征序列，可直接输入跨模态注意力层进行交互计算
        return torch.cat([image_text_embeds, text_embeds], dim=1)


def get_3d_sincos_pos_embed(
        embed_dim: int,
        spatial_size: Union[int, Tuple[int, int]],
        temporal_size: int,
        spatial_interpolation_scale: float = 1.0,
        temporal_interpolation_scale: float = 1.0,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""
    生成3D正弦余弦位置嵌入（纯PyTorch实现）。
    适用于视频等时空数据，同时编码时间维度（帧序列）和空间维度（高×宽）的位置信息。

    Args:
        embed_dim (`int`):
            嵌入维度，必须能被4整除。
        spatial_size (`int` 或 `Tuple[int, int]`):
            空间维度（高度、宽度）。若为整数，默认高度=宽度。
        temporal_size (`int`):
            时间维度（帧数量）。
        spatial_interpolation_scale (`float`, 默认=1.0):
            空间网格插值缩放因子，用于适配不同尺寸的输入。
        temporal_interpolation_scale (`float`, 默认=1.0):
            时间网格插值缩放因子，用于适配不同长度的帧序列。
        device (`torch.device`, 可选):
            输出张量的设备（如CPU、CUDA），默认与输入网格一致。

    Returns:
        `torch.Tensor`:
            3D位置嵌入张量，形状为 `[temporal_size, spatial_size[0] * spatial_size[1], embed_dim]`
            （时间步×空间Patch总数×嵌入维度）。
    """
    # 校验嵌入维度合法性
    if embed_dim % 4 != 0:
        raise ValueError(f"嵌入维度 `embed_dim` 必须能被4整除，当前为 {embed_dim}")
    # 处理空间尺寸（统一为元组格式）
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)  # (高度, 宽度)

    # 分配嵌入维度：3/4给空间，1/4给时间
    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. 生成空间位置嵌入（2D）
    # 生成空间网格坐标（高度、宽度），并应用插值缩放
    grid_h = torch.arange(spatial_size[1], device=device, dtype=torch.float32) / spatial_interpolation_scale
    grid_w = torch.arange(spatial_size[0], device=device, dtype=torch.float32) / spatial_interpolation_scale
    # 生成网格：indexing="xy" 表示 (x,y) 对应 (宽度, 高度)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)  # 形状: [2, 宽度, 高度]
    # 维度调整：[2, 宽度, 高度] → [2, 1, 高度, 宽度]（适配2D嵌入生成函数）
    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    # 生成2D空间位置嵌入
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid, device=device)

    # 2. 生成时间位置嵌入（1D）
    # 生成时间网格坐标，并应用插值缩放
    grid_t = torch.arange(temporal_size, device=device, dtype=torch.float32) / temporal_interpolation_scale
    # 生成1D时间位置嵌入
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, device=device)

    # 3. 融合时空嵌入（广播对齐维度）
    # 空间嵌入扩展时间维度：[H*W, D_spatial] → [1, H*W, D_spatial] → [T, H*W, D_spatial]
    pos_embed_spatial = pos_embed_spatial[None, :, :].repeat_interleave(
        temporal_size, dim=0, output_size=temporal_size
    )
    # 时间嵌入扩展空间维度：[T, D_temporal] → [T, 1, D_temporal] → [T, H*W, D_temporal]
    pos_embed_temporal = pos_embed_temporal[:, None, :].repeat_interleave(
        spatial_size[0] * spatial_size[1], dim=1
    )

    # 拼接时空嵌入：[T, H*W, D_temporal + D_spatial] = [T, H*W, embed_dim]
    pos_embed = torch.concat([pos_embed_temporal, pos_embed_spatial], dim=-1)
    return pos_embed


def get_2d_sincos_pos_embed(
        embed_dim: int,
        grid_size: Union[int, Tuple[int, int]],
        cls_token: bool = False,
        extra_tokens: int = 0,
        interpolation_scale: float = 1.0,
        base_size: int = 16,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    生成2D正弦余弦位置嵌入（纯PyTorch实现）。
    适用于图像、特征图等2D数据，编码空间位置信息（高×宽）。

    Args:
        embed_dim (`int`):
            嵌入维度，必须能被2整除。
        grid_size (`int` 或 `Tuple[int, int]`):
            网格尺寸（高度、宽度）。若为整数，默认高度=宽度。
        cls_token (`bool`, 默认=False):
            是否为分类token预留位置（仅当extra_tokens>0时生效）。
        extra_tokens (`int`, 默认=0):
            额外token数量（如分类token、掩码token），会在嵌入前添加零向量。
        interpolation_scale (`float`, 默认=1.0):
            插值缩放因子，用于适配不同尺寸的输入网格。
        base_size (`int`, 默认=16):
            基础网格尺寸，用于调整位置编码的频率分布。
        device (`torch.device`, 可选):
            输出张量的设备，默认与网格一致。

    Returns:
        `torch.Tensor`:
            2D位置嵌入张量，形状为：
            - 无额外token：`[grid_size[0] * grid_size[1], embed_dim]`（Patch总数×嵌入维度）
            - 有额外token：`[extra_tokens + grid_size[0] * grid_size[1], embed_dim]`
    """
    # 处理网格尺寸（统一为元组格式）
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)  # (高度, 宽度)

    # 生成2D网格坐标，并应用基础尺寸和插值缩放
    grid_h = (
            torch.arange(grid_size[0], device=device, dtype=torch.float32)
            / (grid_size[0] / base_size)
            / interpolation_scale
    )
    grid_w = (
            torch.arange(grid_size[1], device=device, dtype=torch.float32)
            / (grid_size[1] / base_size)
            / interpolation_scale
    )
    # 生成网格：indexing="xy" 表示 (x,y) 对应 (宽度, 高度)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)  # 形状: [2, 宽度, 高度]
    # 维度调整：[2, 宽度, 高度] → [2, 1, 高度, 宽度]（适配嵌入生成函数）
    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])

    # 生成2D位置嵌入
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, device=device)

    # 若有额外token（如cls_token），在嵌入前添加零向量
    if extra_tokens > 0:
        pos_embed = torch.concat([torch.zeros([extra_tokens, embed_dim], device=device), pos_embed], dim=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(
        embed_dim: int,
        grid: torch.Tensor,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    r"""
    从2D网格生成2D正弦余弦位置嵌入（纯PyTorch实现）。
    内部调用1D嵌入生成函数，分别对高度和宽度维度编码后拼接。

    Args:
        embed_dim (`int`):
            嵌入维度，必须能被2整除。
        grid (`torch.Tensor`):
            2D网格坐标张量，形状为 `[2, 1, H, W]`（2表示x/y轴，H=高度，W=宽度）。
        device (`torch.device`, 可选):
            输出张量的设备，默认与grid一致。

    Returns:
        `torch.Tensor`:
            2D位置嵌入张量，形状为 `[H*W, embed_dim]`（网格总点数×嵌入维度）。
    """
    # 校验嵌入维度合法性
    if embed_dim % 2 != 0:
        raise ValueError(f"嵌入维度 `embed_dim` 必须能被2整除，当前为 {embed_dim}")

    # 拆分高度和宽度网格，展平为1D序列
    grid_h = grid[0].flatten()  # 宽度维度 → 展平为 [H*W]
    grid_w = grid[1].flatten()  # 高度维度 → 展平为 [H*W]

    # 分别生成高度和宽度的1D位置嵌入（各占embed_dim/2维度）
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_h, device=device)  # [H*W, D/2]
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_w, device=device)  # [H*W, D/2]

    # 拼接高度和宽度嵌入，得到完整2D位置嵌入
    emb = torch.concat([emb_h, emb_w], dim=1)  # [H*W, D]
    return emb


def get_1d_sincos_pos_embed_from_grid(
        embed_dim: int,
        pos: torch.Tensor,
        device: Optional[torch.device] = None,
        flip_sin_to_cos: bool = False,
) -> torch.Tensor:
    """
    从1D位置序列生成1D正弦余弦位置嵌入（纯PyTorch实现）。
    基于Transformer原论文的正弦余弦位置编码公式，无训练参数，泛化性强。

    Args:
        embed_dim (`int`):
            嵌入维度 `D`，必须能被2整除。
        pos (`torch.Tensor`):
            1D位置序列张量，形状为 `[M]`（M为位置数量）。
        device (`torch.device`, 可选):
            输出张量的设备，默认与pos一致。
        flip_sin_to_cos (`bool`, 默认=False):
            是否交换正弦和余弦分量的顺序：
            - False：[sin, cos]（默认，符合Transformer原论文）
            - True：[cos, sin]（适配部分扩散模型需求）

    Returns:
        `torch.Tensor`:
            1D位置嵌入张量，形状为 `[M, embed_dim]`（位置数量×嵌入维度）。
    """
    # 校验嵌入维度合法性
    if embed_dim % 2 != 0:
        raise ValueError(f"嵌入维度 `embed_dim` 必须能被2整除，当前为 {embed_dim}")
    # 确保设备一致性
    if device is not None:
        pos = pos.to(device)

    # 生成频率因子：omega = 1 / 10000^(2i/D)，i为0~D/2-1
    omega = torch.arange(embed_dim // 2, device=device, dtype=torch.float64)
    omega /= embed_dim / 2.0  # 归一化到[0, 1]
    omega = 1.0 / (10000**omega)  # 频率衰减，低维度对应低频（长周期）

    # 位置与频率的外积：[M] × [D/2] → [M, D/2]
    pos = pos.reshape(-1).float()  # 确保pos为float32类型
    out = torch.outer(pos, omega)  # 每个位置对应不同频率的信号

    # 生成正弦和余弦分量
    emb_sin = torch.sin(out)  # [M, D/2]（正弦分量）
    emb_cos = torch.cos(out)  # [M, D/2]（余弦分量）

    # 拼接正弦和余弦分量，形成完整嵌入
    emb = torch.concat([emb_sin, emb_cos], dim=1)  # [M, D]

    # 可选：交换正弦和余弦的顺序
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, embed_dim // 2:], emb[:, :embed_dim // 2]], dim=1)

    return emb