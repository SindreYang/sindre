
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, groups=1):
    """1x1卷积层"""
    return nn.Conv2d(in_channels, out_channels, 1, groups=groups)
def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    """3x3卷积层"""
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=bias, groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    """2x2上采样层"""
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    else:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv1x1(in_channels, out_channels)
        )


# 卷积块定义
class DownConv(nn.Module):
    """
    下采样卷积块（Conv+ReLU+Conv+ReLU+MaxPool）
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.pool = nn.MaxPool2d(2) if pooling else nn.Identity()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pool(x), x

class UpConv(nn.Module):
    """
    上采样卷积块（UpConv+Merge+Conv+ReLU+Conv+ReLU）
    """
    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='transpose'):
        super().__init__()
        self.upconv = upconv2x2(in_channels, out_channels, up_mode)
        self.conv1 = conv3x3(2*out_channels if merge_mode == 'concat' else out_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.merge_mode = merge_mode

    def forward(self, from_down, from_up):
        from_up = self.upconv(from_up)
        x = torch.cat([from_up, from_down], 1) if self.merge_mode == 'concat' else from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """
    二维UNet网络结构，用于平面特征处理

    Args:
        num_classes (int): 输出通道数
        in_channels (int): 输入通道数
        depth (int): 网络深度
        start_filts (int): 初始卷积核数量
        up_mode (str): 上采样方式 ('transpose'或'upsample')
        same_channels (bool): 是否保持通道数不变
        merge_mode (str): 特征融合方式 ('concat'或'add')
    """
    def __init__(self, num_classes, in_channels=3, depth=5,
                 start_filts=64, up_mode='transpose', same_channels=False,
                 merge_mode='concat', **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        # 编码路径
        self.down_convs = []
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i) if not same_channels else self.in_channels
            pooling = True if i < depth-1 else False
            self.down_convs.append(DownConv(ins, outs, pooling))
        self.down_convs = nn.ModuleList(self.down_convs)

        # 解码路径
        self.up_convs = []
        for i in range(depth-1):
            ins = outs
            outs = ins // 2 if not same_channels else ins
            self.up_convs.append(UpConv(ins, outs, up_mode, merge_mode))
        self.up_convs = nn.ModuleList(self.up_convs)

        # 最终卷积层
        self.conv_final = conv1x1(outs, self.num_classes)

    def forward(self, x):
        """
        前向传播

        Args:
            x (Tensor): 输入特征图 (B, C, H, W)

        Returns:
            Tensor: 输出特征图 (B, num_classes, H, W)
        """
        encoder_outs = []
        # 编码过程
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        # 解码过程
        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # 最终卷积
        x = self.conv_final(x)
        return x
