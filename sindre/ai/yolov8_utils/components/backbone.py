
import torch
import torch.nn as nn
from sindre.ai.yolov8.components.block import Conv,C2f,SPPF
class Backbone(nn.Module):
    """YOLOV8骨干网络，基于改进的 CSPNet 结构设计。

    核心功能是通过 C2f 模块（改进自 C3 模块）逐步提取输入图像的多尺度特征，
    最终输出三个不同层级的特征图（P3/P4/P5），为后续 Neck 层的特征融合和
    Head 层的任务预测（检测/分割/姿态等）提供基础特征支持。

    Attributes:
        stem (Conv): 初始卷积层，负责对输入图像进行第一次下采样和浅层特征提取。
        dark2 (nn.Sequential): 第二阶段特征提取块，包含卷积层和 C2f 模块，实现下采样与特征强化。
        dark3 (nn.Sequential): 第三阶段特征提取块，通道数和 C2f 模块重复次数提升，提取中层特征。
        dark4 (nn.Sequential): 第四阶段特征提取块，进一步提升通道数，提取深层特征。
        dark5 (nn.Sequential): 第五阶段特征提取块，包含 SPPF 空间金字塔池化层，增强全局感受野，
            输出骨干网络最大尺度的特征图。
    """

    def __init__(self, input_channels: int = 3,init_iter_n: int = 3,hidden_channels:list=[64,128,256,512,1024] ):
        """Backbone 类的初始化方法。

        Args:
            input_channels (int, optional): 输入图像的通道数。默认值为 3，对应 RGB 彩色图像。
            hidden_channels:隐藏层
            init_iter_n (int, optional): C2f 模块的基础重复次数。默认值为 3，不同阶段的 C2f 模块会基于此值
                调整（如 dark3/dark4 为 init_iter_n*2），控制模型深度。
        """
        super().__init__()
        # max_

        self.stem = Conv(input_channels, hidden_channels[0], 3, 2)
        self.dark2 = nn.Sequential(
            Conv(hidden_channels[0], hidden_channels[1], 3, 2),
            C2f(hidden_channels[1], hidden_channels[1], init_iter_n, True),
        )
        self.dark3 = nn.Sequential(
            Conv(hidden_channels[1], hidden_channels[2], 3, 2),
            C2f(hidden_channels[2], hidden_channels[2], init_iter_n * 2, True),
        )
        self.dark4 = nn.Sequential(
            Conv(hidden_channels[2],hidden_channels[3], 3, 2),
            C2f(hidden_channels[3], hidden_channels[3], init_iter_n * 2, True),
        )
        self.dark5 = nn.Sequential(
            Conv(hidden_channels[3],hidden_channels[4], 3, 2),
            C2f(hidden_channels[4], hidden_channels[4], init_iter_n, True),
            SPPF(hidden_channels[4], hidden_channels[4], k=5)  # 空间金字塔池化，增强全局特征捕捉能力
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backbone 网络的前向传播方法。

        输入图像张量依次经过 stem、dark2~dark5 模块，逐步下采样并提取特征，
        最终输出三个不同尺度的特征图（P3/P4/P5），分别对应原图 1/8、1/16、1/32 的尺寸。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 [N, C, H, W]，其中：
                N = 批次大小，C = 通道数（与 input_channels 一致），H/W = 输入图像的高度/宽度。

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 多尺度特征图元组，包含三个特征张量：
                - feat1: P3 尺度特征图，形状为 [N, 4*init_channels, H/8, W/8]（默认通道数 256），
                  对应中层特征，适合小目标检测。
                - feat2: P4 尺度特征图，形状为 [N, 8*init_channels, H/16, W/16]（默认通道数 512），
                  对应深层特征，适合中目标检测。
                - feat3: P5 尺度特征图，形状为 [N, max_channels, H/32, W/32]（默认通道数 1024），
                  对应最深层特征，含全局感受野，适合大目标检测。
        """
        x = self.stem(x)  # 输出：[N, init_channels, H/2, W/2]（默认 64 通道）
        x = self.dark2(x)  # 输出：[N, 2*init_channels, H/4, W/4]（默认 128 通道）
        x = self.dark3(x)  # 输出：[N, 4*init_channels, H/8, W/8]（默认 256 通道）
        p3_feat = x  # 保存 P3 尺度特征
        x = self.dark4(x)  # 输出：[N, 8*init_channels, H/16, W/16]（默认 512 通道）
        p4_feat = x  # 保存 P4 尺度特征
        x = self.dark5(x)  # 输出：[N, max_channels, H/32, W/32]（默认 1024 通道）
        p5_feat = x  # 保存 P5 尺度特征
        return p3_feat, p4_feat, p5_feat
