
import torch
import torch.nn as nn


# ResNet块
class ResnetBlockFC(nn.Module):
    """
    全连接ResNet块

    Args:
        size_in (int): 输入维度
        size_out (int): 输出维度
        size_h (int): 隐藏层维度
    """
    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in
        if size_h is None:
            size_h = min(size_in, size_out)

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()
        self.shortcut = nn.Linear(size_in, size_out) if size_in != size_out else None

        # 初始化
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        residual = x
        x = self.actvn(x)
        x = self.fc_0(x)
        x = self.actvn(x)
        x = self.fc_1(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        return x + residual

