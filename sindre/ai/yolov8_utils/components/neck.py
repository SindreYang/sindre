
import torch
import torch.nn as nn
from sindre.ai.yolov8.components.block import Conv,C2f
class Neck(nn.Module):
    """YOLO 风格 Neck 网络，基于 FPN（特征金字塔）+ PAN（路径聚合网络）结构设计。

    核心功能是接收 Backbone 输出的多尺度特征（P3/P4/P5），通过“上采样+特征拼接+下采样”的双向融合逻辑，
    强化不同尺度特征间的信息交互，弥补单一尺度特征的语义/细节缺失，最终输出融合后的多尺度特征，
    为后续 Head 层提供更全面的特征支撑（提升小/中/大目标的检测精度）。

    Attributes:
        upsample (nn.Upsample): 固定2倍上采样模块（ nearest 插值，避免模糊），用于将高层特征对齐低层尺度。
        conv3_for_upsample1 (C2f): P5→P4 融合后的特征压缩模块，减少通道数并强化特征表达。
        conv3_for_upsample2 (C2f): P4→P3 融合后的特征压缩模块，适配 P3 尺度的通道需求。
        down_sample1 (Conv): P3→P4 下采样模块
        conv3_for_downsample1 (C2f): P3→P4 下采样融合后的特征强化模块，整合双向信息。
        down_sample2 (Conv): P4→P5 下采样模块
        conv3_for_downsample2 (C2f): P4→P5 下采样融合后的特征强化模块，最终输出 P5 尺度特征。
    """

    def __init__(self, hidden_channels:list=[64,128,256,512,1024], init_iter_n: int = 3):
        """Neck 网络的初始化方法，基于 Backbone 输出的 P5 通道数动态构建模块。

        Args:
            input_channels (int): Backbone 输出的 P5 特征通道数
                决定 Neck 各模块的通道基准（P4 通道=input_channels//2，P3 通道=input_channels//4）。
            init_iter_n (int, optional): C2f 模块的基础重复次数，控制特征强化的深度。默认值为 3，
                次数越多特征表达能力越强，但计算量相应增加。
        """
        super(Neck, self).__init__()
        # 2倍上采样模块（对齐低层特征尺度）
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # 1. FPN 上采样融合分支（高层→低层：P5→P4→P3）
        # P5与P4融合后压缩通道：(P5通道 + P4通道) → P4通道（input_channels + input_channels//2 → input_channels//2）
        self.conv3_for_upsample1 = C2f(
            c1=hidden_channels[4] + hidden_channels[3],
            c2=hidden_channels[3],
            n=init_iter_n,
            shortcut=False  # 关闭残差连接，专注融合后特征重塑
        )
        # P4与P3融合后压缩通道：(P4通道 + P3通道) → P3通道（input_channels//2 + input_channels//4 → input_channels//4）
        self.conv3_for_upsample2 = C2f(
            c1=hidden_channels[3] + hidden_channels[2],
            c2= hidden_channels[2],
            n=init_iter_n,
            shortcut=False
        )

        # 2. PAN 下采样融合分支（低层→高层：P3→P4→P5）
        # P3下采样至P4尺度：通道保持P3通道（input_channels//4）
        self.down_sample1 = Conv(
            c1= hidden_channels[2],
            c2= hidden_channels[2],
            k=3,
            s=2  # 步长2实现2倍下采样
        )
        # P3下采样与P4融合后强化：(P3下采样通道 + P4通道) → P4通道（input_channels//4 + input_channels//2 → input_channels//2）
        self.conv3_for_downsample1 = C2f(
            c1= hidden_channels[3] +  hidden_channels[2],
            c2=hidden_channels[3],
            n=init_iter_n,
            shortcut=False
        )
        # P4下采样至P5尺度：通道保持P4通道（input_channels//2）
        self.down_sample2 = Conv(
            c1=hidden_channels[3],
            c2=hidden_channels[3],
            k=3,
            s=2
        )
        # P4下采样与P5融合后强化：(P4下采样通道 + P5通道) → P5通道（input_channels//2 + input_channels → input_channels）
        self.conv3_for_downsample2 = C2f(
            c1=hidden_channels[4] +hidden_channels[3],
            c2=hidden_channels[4],
            n=init_iter_n,
            shortcut=False
        )

    def forward(self, p3_feat: torch.Tensor, p4_feat: torch.Tensor, p5_feat: torch.Tensor):
        """Neck 网络的前向传播方法，实现双向特征融合（FPN+PAN）。
        Args:
            p3_feat (torch.Tensor): Backbone 输出的 P3 特征图，形状为 [N, C3, H3, W3]，其中：
            p4_feat (torch.Tensor): Backbone 输出的 P4 特征图，形状为 [N, C4, H4, W4]，其中：
            p5_feat (torch.Tensor): Backbone 输出的 P5 特征图，形状为 [N, C5, H5, W5]，其中：

        Returns:
            List[torch.Tensor]: 融合后的多尺度特征列表 [P3_fused, P4_fused, P5_fused]，各特征图属性：
                - P3_fused：形状 [N, C3, H3, W3]，融合 P4 语义信息，提升小目标细节识别能力。
                - P4_fused：形状 [N, C4, H4, W4]，双向融合 P3 细节与 P5 语义，平衡中目标检测精度。
                - P5_fused：形状 [N, C5, H5, W5]，融合 P4 细节信息，减少大目标定位误差。
        """
        # ------------------------ FPN 上采样融合（高层→低层）------------------------
        # 1. P5 → P4 融合：P5 上采样至 P4 尺度（H5*2=H4, W5*2=W4），与 P4 拼接
        P5_upsample = self.upsample(p5_feat)  # 形状：[N, C5, H4, W4]
        P4_temp = torch.cat([P5_upsample, p4_feat], dim=1)  # 拼接后通道：C5+C4=input_channels + input_channels//2
        # 压缩通道至 P4 尺度，并强化特征
        P4_fused1 = self.conv3_for_upsample1(P4_temp)  # 形状：[N, C4, H4, W4]

        # 2. P4 → P3 融合：P4 上采样至 P3 尺度（H4*2=H3, W4*2=W3），与 P3 拼接
        P4_upsample = self.upsample(P4_fused1)  # 形状：[N, C4, H3, W3]
        P3_temp = torch.cat([P4_upsample, p3_feat], dim=1)  # 拼接后通道：C4+C3=input_channels//2 + input_channels//4
        # 压缩通道至 P3 尺度，得到融合后的 P3
        P3_fused = self.conv3_for_upsample2(P3_temp)  # 最终 P3 特征：[N, C3, H3, W3]

        # ------------------------ PAN 下采样融合（低层→高层）------------------------
        # 1. P3 → P4 融合：P3 下采样至 P4 尺度（H3//2=H4, W3//2=W4），与 P4_fused1 拼接
        P3_downsample = self.down_sample1(P3_fused)  # 形状：[N, C3, H4, W4]
        P4_temp2 = torch.cat([P3_downsample, P4_fused1], dim=1)  # 拼接后通道：C3+C4=input_channels//4 + input_channels//2
        # 强化特征并保持 P4 通道，得到融合后的 P4
        P4_fused = self.conv3_for_downsample1(P4_temp2)  # 最终 P4 特征：[N, C4, H4, W4]

        # 2. P4 → P5 融合：P4 下采样至 P5 尺度（H4//2=H5, W4//2=W5），与原始 P5 拼接
        P4_downsample = self.down_sample2(P4_fused)  # 形状：[N, C4, H5, W5]
        P5_temp = torch.cat([P4_downsample, p5_feat], dim=1)  # 拼接后通道：C4+C5=input_channels//2 + input_channels
        # 强化特征并保持 P5 通道，得到融合后的 P5
        P5_fused = self.conv3_for_downsample2(P5_temp)  # 最终 P5 特征：[N, C5, H5, W5]

        # 返回融合后的多尺度特征（顺序：P3, P4, P5）
        return [P3_fused, P4_fused, P5_fused]