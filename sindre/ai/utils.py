import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Optional, Dict
from sindre.general.logs import CustomLogger
from sindre.deploy.check_tools import check_gpu_info,timeit
log= CustomLogger(logger_name="ai_utils").get_logger()

def set_global_seeds(seed: int = 1024,cudnn_enable: bool = False) -> None:
    """
    设置全局随机种子，确保Python、NumPy、PyTorch等环境的随机数生成器同步，提升实验可复现性。

    Args:
        seed (int): 要使用的基础随机数种子，默认值为1024。
        cudnn_enable (bool): 是否将CuDNN设置为确定性模式，启用后可能会影响性能但提高可复现性，默认值为False。
    """
    # 设置Python内置的随机数生成器
    import random,os
    random.seed(seed)
    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置NumPy的随机数生成器
    np.random.seed(seed)
    # 尝试设置PyTorch的随机数生成器
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if cudnn_enable:
                # 控制CuDNN的确定性和性能之间的平衡
                torch.backends.cudnn.deterministic = True
                # 禁用CuDNN的自动寻找最优算法
                torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    log.info(f"全局随机种子已设置为 {seed} | CuDNN确定性模式: {'启用' if cudnn_enable else '禁用'}")


def save_checkpoint(
        save_path:str,
        network: torch.nn.Module,
        loss: float,
        optimizer: Optional[torch.optim.Optimizer]=None,
        curr_iter: int=0,
        extra_info: Optional[Dict] = None,
        save_best_only: bool = True
) -> None:
    """
    保存模型状态、优化器状态、当前迭代次数和损失值;
    save_best_only开启后，直接比较已保存模型的loss(避免硬件故障引起保存问题)

    Args:
        save_path: 包含模型保存路径等参数的配置对象
        network: 神经网络模型
        optimizer: 优化器
        loss: 当前损失值
        curr_iter: 当前迭代次数
        extra_info: 可选的额外信息字典，用于保存其他需要的信息
        save_best_only: 是否仅在损失更优时保存模型，默认为True
    """
    try:
        # 判断是否需要最优保存
        if save_best_only:
            # 仅保存最佳模型时，才需要检查当前最佳损失
            curr_best_loss = float('inf')
            if os.path.exists(save_path):
                try:
                    checkpoint = torch.load(save_path, map_location='cpu')
                    curr_best_loss = checkpoint.get("loss", float('inf'))
                except Exception as e:
                    log.warning(f"Failed to load existing checkpoint: {str(e)}")
            # 检查当前损失是否更优
            if loss > curr_best_loss:
                return  # 不保存，直接返回

        # 获取模型状态字典torch.nn.parallel.distributed.DistributedDataParalle
        if "DataParalle" in str(type(network)):
            net_dict = network.module.state_dict()
        else:
            net_dict = network.state_dict()

        # 创建保存字典
        save_dict = {
            "state_dict": net_dict,
            "optimizer": optimizer.state_dict(),
            "curr_iter": curr_iter,
            "loss": loss,
        }

        # 添加额外信息
        if extra_info is not None:
            save_dict.update(extra_info)
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # 保存模型
        torch.save(save_dict, save_path)
        log.info(f"Save model path: {save_path},loss: {loss}, iteration: {curr_iter}")

    except Exception as e:
        log.error(f"Failed to save model: {str(e)}", exc_info=True)
        raise

def load_checkpoint(
        path: str,
        net: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
        check_shape: bool = True,
        map_location: Optional[str] = None
) -> Tuple[int, float, Dict]:
    """
    加载模型状态，可以支持部分参数加载

    加载策略:\n
    - strict==True: 只有名称和形状完全一致的参数才会被加载；
    - strict==False且check_shape==True: 仅加载名称存在且形状匹配的参数；
    - strict==False且check_shape==False: 加载所有名称匹配的参数，不检查形状；

    Args:
        path: 模型文件路径
        net: 要加载参数的神经网络模型
        optimizer: 优化器，如果需要加载优化器状态
        strict: 是否严格匹配模型参数
        check_shape: 是否检查参数形状匹配
        map_location: 指定设备映射，例如"cpu"或"cuda:0"

    Returns:
        curr_iter:加载了最后迭代次数;
        loss: 最后损失值;
        extra_info: 额外信息字典;

    """
    try:
        # 检查模型文件是否存在
        if not os.path.exists(path):
            log.warning(f"模型文件不存在: {path}")
            return 0,float("inf"),{}

        # 加载模型数据
        log.info(f"加载模型: {path}")
        checkpoint = torch.load(path, map_location=map_location)
        model_state, checkpoint_state = net.state_dict(), checkpoint["state_dict"]


        #  DDP前缀适配：统一参数名格式
        is_ddp = "DataParalle" in str(type(net))
        has_module_prefix = any(k.startswith("module.") for k in checkpoint_state.keys())
        norm_ckpt = {}

        log.info(f"参数是DDP:{has_module_prefix}, 网络是DDP：{is_ddp}")
        for k, v in checkpoint_state.items():
            if is_ddp and not has_module_prefix:
                norm_k = f"module.{k}"  # DDP缺前缀→补
            elif not is_ddp and has_module_prefix:
                norm_k = k[7:] if k.startswith("module.") else k  # 普通模型多前缀→删
            else:
                norm_k = k
            norm_ckpt[norm_k] = v
        checkpoint_state=norm_ckpt

        # 处理参数匹配
        if check_shape and not strict:
            filtered = {}
            for k in checkpoint_state:
                if k in model_state and checkpoint_state[k].shape == model_state[k].shape:
                    filtered[k] = checkpoint_state[k]
                elif k in model_state:
                    log.warning(f"参数形状不匹配，跳过: {k} "
                                f"({checkpoint_state[k].shape} vs {model_state[k].shape})")
            net.load_state_dict(filtered, strict=False)
        else:
            net.load_state_dict(checkpoint_state, strict=strict)

        # 加载优化器状态
        if optimizer is not None:
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                log.info("优化器状态已加载")
        # 获取额外信息
        curr_iter = checkpoint.get("curr_iter", 0)
        loss = checkpoint.get("loss", float("inf"))
        known_keys = {"state_dict", "optimizer", "curr_iter", "loss"}
        extra_info = {k: v for k, v in checkpoint.items() if k not in known_keys}
        log.info(f"模型加载完成，最后迭代次数: {curr_iter}, 最后损失值: {loss:.6f},额外信息:{extra_info.keys()}")
        return  curr_iter, loss, extra_info
    except Exception as e:
        log.error(f"加载模型失败: {str(e)}", exc_info=True)
        raise





def pca_color_by_feat(feat, brightness=1.25, center=True):
    """
    通过PCA将高维特征转换为RGB颜色，用于可视化。

    该函数使用主成分分析(PCA)对输入特征进行降维，
    组合前6个主成分生成3维颜色向量，并将其归一化到[0, 1]范围，
    适用于作为RGB颜色值进行点云等数据的可视化。

    Args:
        feat (torch.Tensor): 输入的高维特征张量。
            形状应为(num_points, feature_dim)，其中num_points是点的数量，
            feature_dim是每个特征的维度。
        brightness (float, 可选): 颜色亮度的缩放因子。
            值越高，整体颜色越明亮。默认值为1.25。
        center (bool, 可选): 在执行PCA之前是否对特征进行中心化（减去均值）。
            默认值为True。

    Returns:
         torch.Tensor: 归一化到[0, 1]范围的RGB颜色值。
            形状为(num_points, 3)，每行代表(R, G, B)三个通道的颜色值。

    """
    u, s, v = torch.pca_lowrank(feat, center=center, q=6, niter=5)
    projection = feat @ v
    projection = projection[:, :3] * 0.6 + projection[:, 3:6] * 0.4
    min_val = projection.min(dim=-2, keepdim=True)[0]
    max_val = projection.max(dim=-2, keepdim=True)[0]
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (projection - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color



def set_ssl(SteamToolsCertificate_path:str):
    """
    将steam++的证书写入到ssl中,防止requests.exceptions.SSLError报错
    SteamToolsCertificate_path = os.path.join(os.path.dirname(__file__),"data","SteamTools.Certificate.pfx")
    """

    import requests
    import certifi
    print("将steam++的证书写入到ssl中,防止requests.exceptions.SSLError报错 ,原则上只调用一次")
    try:
        print('Checking connection to Huggingface...')
        test = requests.get('https://huggingface.co')
        print('Connection to Huggingface OK.')
    except requests.exceptions.SSLError as err:
        print('SSL Error. Adding custom certs to Certifi store...')
        cafile = certifi.where()
        with open(SteamToolsCertificate_path, 'rb') as infile:
            customca = infile.read()
        with open(cafile, 'ab') as outfile:
            outfile.write(customca)
        print('That might have worked.')


def disable_ssl():
    import requests
    import warnings
    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    # 禁用 SSL 验证
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # 忽略警告
    session = requests.Session()
    session.verify = False  # 禁用验证
    requests.Session = lambda: session  # 全局覆盖 Session










