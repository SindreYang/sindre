import torch
from torch import nn
from sindre.general.logs import  CustomLogger
log = CustomLogger('init').get_logger()

def init_linear_modules(module: nn.Module, stddev: float = 0.25,debug=False):
    """
    通用初始化函数：自动遍历模块的所有子层，对 nn.Linear 层执行指定初始化
    Args:
        module: 待初始化的模型/子模块（如 model、model.shape_encoder、ResidualCrossAttentionBlock 等）
        stddev: 线性层权重的正态分布标准差
    """
    # 递归遍历所有子模块，自动检测 Linear 层
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=stddev)  # 权重正态初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)     # 偏置置0
            weight_mean = m.weight.mean().item()
            weight_max = m.weight.max().item()
            weight_norm = torch.norm(m.weight).item()
            if debug:
                log.info(f"按照方差{stddev},初始化 {m} 完成 | mean={weight_mean:.6f}, max={weight_max:.6f}, norm={weight_norm:.6f}")