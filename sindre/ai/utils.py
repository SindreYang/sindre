import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List, Mapping, Tuple, Optional, Dict, Union
from sindre.general.logs import CustomLogger
from sindre.deploy.check_tools import check_gpu_info,timeit
import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Union, List, Tuple

import datetime
import builtins
import torch
import torch.distributed as dist

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
            "optimizer": optimizer.state_dict() if optimizer else None,
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
    - strict==True: 仅加载修正后名称和形状完全一致的参数，不匹配则抛出异常；
    - strict==False,check_shape=True: 仅加载修正后名称存在且形状匹配的参数，其余跳过；(修改网络结构迁移预训练模型参数)
    - strict==False,check_shape=False: 加载所有修正后名称匹配的参数（加载同结构模型，ckpt 包含多余键值）；

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
            if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr = optimizer.param_groups[0]['lr']  # 获取第一个参数组的学习率
                log.info(
                    f"优化器状态已加载,优化器类型：{type(optimizer).__name__}, 学习率：{lr:.6f}"
                )
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


def disable_huggingface_ssl():
    """
    禁用huggingface的ssl验证
    """
    from huggingface_hub import configure_http_backend
    import requests
    # # 2. 配置requests不验证SSL
    def custom_requests_session():
        session = requests.Session()
        session.verify = False  # 禁用SSL验证
        return session
    configure_http_backend(custom_requests_session)
    





class MultiGPUManager:
    """
    通用分布式训练工具类（适配torchrun，补充dist.barrier的device_ids）
    默认单GPU运行（无GPU则回退到CPU），cuda_visible_devices仅支持None/逗号分隔字符串

    NOTE: 核心使用场景与参数/命令说明
    =====================================
    场景1：默认单GPU（非分布式）
    - 参数配置：无需额外参数
    - 代码调用：manager = MultiGPUManager()
    - 启动命令：python your_train_script.py

    场景2：单节点3卡分布式（适配你的原有命令）
    - 参数配置：
      manager = MultiGPUManager(
          distributed=True,
          cuda_visible_devices="1,2,3"  # 指定GPU 1,2,3
      )
    - 启动命令：CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 your_train_script.py

    场景3：单节点4卡分布式（手动传参）
    - 参数配置：
      manager = MultiGPUManager(
          distributed=True,
          rank=0,
          local_rank=0,
          world_size=4,
          master_addr="127.0.0.1",
          master_port="29500",
          cuda_visible_devices="0,1,2,3"
      )
    - 启动命令：torchrun --nproc_per_node=4 your_train_script.py

    种子逻辑说明：
    - 非分布式：固定种子（默认1024）；
    - 分布式：种子=1024+local_rank（确保每个GPU随机数不同）。
    """
    def __init__(
        self,
        distributed: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: str = "29500",
        seed: int = 1024,
        cuda_visible_devices: Optional[str] = None  # 仅支持None/逗号分隔字符串
    ):
        """
        初始化分布式环境（补充dist.barrier的device_ids参数）
        Args:
            distributed: 是否开启分布式（默认False）
            rank: 全局进程编号（优先读环境变量，默认0）
            local_rank: 本地GPU编号（优先读环境变量，默认0）
            world_size: 总进程数（优先读环境变量，默认1）
            master_addr: 主节点IP（默认本地回环）
            master_port: 主节点端口（默认29500）
            seed: 基础随机种子（分布式下+local_rank）
            cuda_visible_devices: 指定可见GPU：
                - None: 分布式默认全部GPU，非分布式默认GPU 0；
                - str: 逗号分隔字符串（如"1,2,3"）。
        """
        # 1. 处理CUDA_VISIBLE_DEVICES（仅None/str）
        self._setup_cuda_visible_devices(cuda_visible_devices, distributed)

        # 2. 优先读环境变量（适配torchrun），未读取则用传入参数
        self.rank = int(os.environ.get("RANK", rank))
        self.local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
        self.world_size = int(os.environ.get("WORLD_SIZE", world_size))
        self.distributed = distributed if distributed else (self.world_size > 1)
        self.master_addr = master_addr
        self.master_port = master_port
        self.seed = seed

        # 3. 设备初始化（适配指定的GPU）
        self._setup_device()

        # 4. 非分布式模式
        if not self.distributed:
            set_global_seeds(self.seed)
            self._setup_distributed_print()
            return

        # 5. 分布式模式：参数校验 + 环境初始化 + 种子设置
        self._validate_dist_params()
        self._init_distributed_env()
        self._setup_distributed_print()
        set_global_seeds(self.seed + self.local_rank)  # 分布式种子+local_rank
        

    def _setup_cuda_visible_devices(self, cuda_visible_devices: Optional[str], distributed: bool):
        """设置CUDA_VISIBLE_DEVICES（仅None/str，校验合法性）"""
        total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if total_gpus == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return

        # 默认逻辑：分布式用全部GPU，非分布式用GPU 0
        if cuda_visible_devices is None:
            visible_str = ",".join(map(str, range(total_gpus))) if distributed else "0"
        # 字符串类型：校验每个GPU ID合法性
        elif isinstance(cuda_visible_devices, str):
            visible_list = [int(g) for g in cuda_visible_devices.split(",")]
            for gpu_id in visible_list:
                if gpu_id < 0 or gpu_id >= total_gpus:
                    raise ValueError(f"GPU {gpu_id} 超出可用范围（0~{total_gpus-1}）")
            visible_str = cuda_visible_devices
        else:
            raise TypeError("cuda_visible_devices仅支持None/逗号分隔字符串")

        os.environ["CUDA_VISIBLE_DEVICES"] = visible_str
        print(f"已设置CUDA_VISIBLE_DEVICES: {visible_str}")

    def _setup_device(self):
        """初始化设备（适配指定的GPU）"""
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        elif self.distributed:
            self.device = torch.device("cuda", self.local_rank)
        else:
            # 非分布式：用指定的第一个GPU（如cuda_visible_devices="2"则用cuda:0，对应物理2）
            self.device = torch.device("cuda:0")

    def _validate_dist_params(self):
        """校验分布式参数合法性"""
        if self.world_size < 2:
            raise ValueError(f"分布式world_size需≥2，当前{self.world_size}")
        if not (0 <= self.rank < self.world_size):
            raise ValueError(f"rank={self.rank} 超出范围[0, {self.world_size-1}]")
        if self.local_rank < 0:
            raise ValueError(f"local_rank不能为负，当前{self.local_rank}")
        if not torch.cuda.is_available():
            raise RuntimeError("分布式模式需要CUDA支持，但未检测到GPU")

    def _init_distributed_env(self):
        """初始化分布式环境（补充device_ids）"""
        # 绑定GPU并初始化进程组
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend="nccl",
            # 优先用env://（适配torchrun），未读取则用tcp://
            init_method=os.environ.get("DIST_INIT_METHOD", f"tcp://{self.master_addr}:{self.master_port}"),
            world_size=self.world_size,
            rank=self.rank
        )
        # 关键：dist.barrier加入device_ids
        dist.barrier(device_ids=[self.local_rank])

    def _setup_distributed_print(self):
        """控制日志仅主进程打印（带时间戳）"""
        builtin_print = builtins.print
        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if self.rank == 0 or force:
                now = datetime.datetime.now().time()
                builtin_print(f'[{now}] ', end='')
                builtin_print(*args, **kwargs)
        builtins.print = print

    def all_reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """多进程张量均值聚合（补充device_ids）"""
        if not self.distributed:
            return tensor.to(self.device)

        tensor = tensor.to(self.device)
        dist.barrier(device_ids=[self.local_rank])  # 加入device_ids
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """封装DDP模型（补充device_ids）"""
        model = model.to(self.device)
        if not self.distributed:
            return model

        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(model, device_ids=[self.local_rank])
        dist.barrier(device_ids=[self.local_rank])  # 加入device_ids
        # 同步主进程参数到所有进程
        for param in ddp_model.parameters():
            dist.broadcast(param.data, src=0, device_ids=[self.local_rank])
        return ddp_model
    

    

    def cleanup(self):
        """清理分布式环境（补充device_ids）"""
        if self.distributed and dist.is_initialized():
            dist.barrier(device_ids=[self.local_rank])  # 加入device_ids
            dist.destroy_process_group()
            
            
            




class TensorBoardSummary(SummaryWriter):

    def __init__(self, *args, **kwargs):
        """无需额外配置"""
        super().__init__(*args, **kwargs)
        
        from sindre.utils3d.algorithm import labels2colors
        self._color_map=torch.from_numpy( labels2colors(np.array([i for i in range(1000)]))[...,:3],dtype=torch.int)# 返回RGB,[0-255]

    def add_mesh_by_vertexlabels(
        self,
        tag: str,
        verts: torch.Tensor,
        faces:torch.Tensor,
        labels:torch.Tensor,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """
        根据顶点labels渲染颜色
        
        Args:
            tag: Mesh在TensorBoard中的标签名（如 "3d_mesh/train_sample"）
            verts: 顶点坐标，shape=(batch_size, num_vertices, 3) 或 (num_vertices, 3)
            faces: 面片索引， (batch_size, num_faces, 3)或 shape=(num_faces, 3) 
            labels: 顶点标签（用于颜色映射），shape与verts前两维一致
            global_step: 训练步骤/epoch
            walltime: 事件时间戳（默认当前时间）
        """
        device = verts.device
        if labels.dim() != 2 :
            raise ValueError(f"labels维度错误！必须是 (B, N) \n")
        colors = self._color_map[labels].to(device=device)
        super().add_mesh(
            tag=tag,
            vertices=verts,
            faces=faces,
            colors=colors,
            global_step=global_step,
            walltime=walltime
        )


    def add_pointcloud_by_vertexlabels(
            self,
            tag: str,
            points: torch.Tensor,
            labels: torch.Tensor,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None
        ):
        device = points.device
        if labels.dim() != 2 :
            raise ValueError(f"labels维度错误！必须是 (B, N) \n")
        colors = self._color_map[labels].to(device=device)
        super().add_mesh(
            tag=tag,
            vertices=points,
            colors=colors,
            global_step=global_step,
            walltime=walltime
        )
        
    def add_pointcloud(
            self,
            tag: str,
            points: torch.Tensor,
            global_step: Optional[int] = None,
            walltime: Optional[float] = None
        ):
        batch_size, num_vertices = points.shape[:2]
        labels = torch.zeros((batch_size, num_vertices), dtype=torch.long)
        device = points.device
        colors = self._color_map[labels].to(device=device)
        super().add_mesh(
            tag=tag,
            vertices=points,
            colors=colors,
            global_step=global_step,
            walltime=walltime
        )
    
    
    def add_3d_mesh(
        self,
        tag: str,
        verts: torch.Tensor,
        faces:torch.Tensor,
        global_step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """
        根据顶点labels渲染颜色
        
        Args:
            tag: Mesh在TensorBoard中的标签名（如 "3d_mesh/train_sample"）
            verts: 顶点坐标，shape=(batch_size, num_vertices, 3) 或 (num_vertices, 3)
            faces: 面片索引， (batch_size, num_faces, 3)或 shape=(num_faces, 3) 
            labels: 顶点标签（用于颜色映射），shape与verts前两维一致
            global_step: 训练步骤/epoch
            walltime: 事件时间戳（默认当前时间）
        """
        batch_size, num_vertices = verts.shape[:2]
        labels = torch.zeros((batch_size, num_vertices), dtype=torch.long)
        device = verts.device
        colors = self._color_map[labels].to(device=device)
        super().add_mesh(
            tag=tag,
            vertices=verts,
            faces=faces,
            colors=colors,
            global_step=global_step,
            walltime=walltime
        )
        
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if isinstance(v, dict):
                super().add_scalars(k, v, step)
            else:
                try:
                    super().add_scalar(k, v, step)
                except Exception as ex:
                    m = f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex
                
    
                
            



