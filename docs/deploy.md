# Deploy 部署模块

> 模型部署和推理加速工具，支持ONNX、TensorRT、共享内存等多种部署方式

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [使用指南](#使用指南)
- [高级功能](#高级功能)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

## ✨ 功能特性

- 🚀 **多格式支持**: ONNX、TensorRT、OpenVINO等推理引擎
- 💾 **共享内存**: 高效的进程间数据传输
- 🔧 **模型优化**: 自动模型量化和优化
- 📊 **性能监控**: 实时推理性能统计
- 🔄 **热更新**: 支持模型动态更新
- 🛡️ **错误处理**: 完善的异常处理机制
- 📈 **可扩展**: 支持自定义推理后端

## 🚀 快速开始

### 基本使用

```python
from sindre.deploy import onnxruntime_deploy, TenserRT_deploy

# ONNX Runtime部署
onnx_infer = onnxruntime_deploy.OnnxInfer("model.onnx")
result = onnx_infer(input_data)

# TensorRT部署
trt_infer = TenserRT_deploy.TRTInfer()
trt_infer.load_model("model.engine")
result = trt_infer(input_data)
```

### 共享内存部署

```python
from sindre.deploy import python_share_memory

# 创建共享内存服务
server = python_share_memory.SharedMemoryServer("model_server")
server.start()

# 客户端连接
client = python_share_memory.SharedMemoryClient("model_server")
result = client.infer(input_data)
```

## 🔧 核心功能

### ONNX Runtime 部署

```python
class OnnxInfer:
    """ONNX模型推理类"""
    
    def __init__(self, onnx_path: str, providers: List[Tuple[str, Dict[str, Any]]] = [('CPUExecutionProvider', {})], enable_log: bool = False):
        """
        初始化ONNX推理
        
        Args:
            onnx_path: ONNX模型文件路径
            providers: 推理提供者列表
            enable_log: 是否启用日志
        """
    
    def __call__(self, inputs: np.ndarray) -> List[np.ndarray]:
        """
        执行模型推理
        
        Args:
            inputs: 输入数据（numpy数组或字典）
            
        Returns:
            List[np.ndarray]: 推理结果
        """
    
    def optimizer(self, save_onnx: str):
        """
        优化并简化ONNX模型
        
        Args:
            save_onnx: 保存路径
        """
    
    def convert_opset_version(self, save_path: str, target_version: int):
        """
        转换ONNX模型的Opset版本
        
        Args:
            save_path: 保存路径
            target_version: 目标Opset版本
        """
    
    def fix_input_shape(self, save_path: str, input_shapes: list):
        """
        固定ONNX模型的输入尺寸
        
        Args:
            save_path: 保存路径
            input_shapes: 输入形状列表
        """
    
    def dynamic_input_shape(self, save_path: str, dynamic_dims: list):
        """
        设置ONNX模型的输入为动态尺寸
        
        Args:
            save_path: 保存路径
            dynamic_dims: 动态维度列表
        """
    
    def test_performance(self, loop: int = 10, warmup: int = 3):
        """
        测试推理性能
        
        Args:
            loop: 正式测试循环次数
            warmup: 预热次数
        """
```

### TensorRT 部署

```python
class TRTInfer:
    """TensorRT推理类"""
    
    def __init__(self):
        """初始化TensorRT推理"""
    
    def load_model(self, engine_path: str):
        """
        加载TensorRT引擎
        
        Args:
            engine_path: 引擎文件路径
        """
    
    def build_engine(self, onnx_path: str, engine_path: str, max_workspace_size=4<<30, 
                    fp16=False, dynamic_shape_profile=None, hardware_compatibility="", 
                    optimization_level=3, version_compatible=False):
        """
        从ONNX构建TensorRT引擎
        
        Args:
            onnx_path: ONNX模型路径
            engine_path: 输出引擎路径
            max_workspace_size: 最大工作空间大小
            fp16: 是否使用FP16
            dynamic_shape_profile: 动态形状配置
            hardware_compatibility: 硬件兼容性
            optimization_level: 优化级别
            version_compatible: 版本兼容性
        """
    
    def __call__(self, data):
        """
        执行推理
        
        Args:
            data: 输入数据
            
        Returns:
            List[np.ndarray]: 推理结果
        """
    
    def test_performance(self, loop: int = 10, warmup: int = 3) -> float:
        """
        测试推理性能
        
        Args:
            loop: 正式测试循环次数
            warmup: 预热次数
            
        Returns:
            float: 平均推理时间
        """
```

### 共享内存部署

```python
class SharedMemoryServer:
    """共享内存服务器"""
    
    def __init__(self, name: str, model_path: str):
        """
        初始化共享内存服务器
        
        Args:
            name: 服务器名称
            model_path: 模型路径
        """
    
    def start(self):
        """启动服务器"""
    
    def stop(self):
        """停止服务器"""

class SharedMemoryClient:
    """共享内存客户端"""
    
    def __init__(self, server_name: str):
        """
        初始化共享内存客户端
        
        Args:
            server_name: 服务器名称
        """
    
    def infer(self, input_data: dict):
        """
        通过共享内存执行推理
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            dict: 推理结果
        """
```

## 📖 使用指南

### 1. ONNX Runtime 部署

#### 基本推理

```python
from sindre.deploy import onnxruntime_deploy
import numpy as np

# 创建推理实例
infer = onnxruntime_deploy.OnnxInfer("model.onnx")

# 准备输入数据
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 执行推理
result = infer(input_data)
print(f"推理结果: {result}")

# 获取推理时间
infer.test_performance(loop=10, warmup=3)
```

#### 多输入推理

```python
# 多输入模型
input_data = {
    "input1": np.random.rand(1, 3, 224, 224).astype(np.float32),
    "input2": np.random.rand(1, 10).astype(np.float32)
}

result = infer(input_data)
print(f"多输入推理结果: {result}")
```

#### 模型优化

```python
# 优化模型
infer.optimizer("optimized_model.onnx")

# 转换Opset版本
infer.convert_opset_version("model_v16.onnx", 16)

# 固定输入形状
infer.fix_input_shape("fixed_model.onnx", [[1, 3, 224, 224]])

# 设置动态输入
infer.dynamic_input_shape("dynamic_model.onnx", [[None, 3, None, None]])
```

### 2. TensorRT 部署

#### 基本推理

```python
from sindre.deploy import TenserRT_deploy
import numpy as np

# 创建推理实例
trt_infer = TenserRT_deploy.TRTInfer()

# 加载引擎
trt_infer.load_model("model.engine")

# 准备输入数据
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 执行推理
result = trt_infer(input_data)
print(f"TensorRT推理结果: {result}")

# 性能测试
avg_time = trt_infer.test_performance(loop=100, warmup=10)
print(f"平均推理时间: {avg_time:.3f}ms")
```

#### 构建引擎

```python
# 从ONNX构建引擎
trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="model.engine",
    max_workspace_size=4<<30,  # 4GB
    fp16=True,  # 使用FP16
    optimization_level=3
)

# 动态形状引擎
dynamic_profile = {
    "input": [(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)]
}

trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="dynamic_model.engine",
    dynamic_shape_profile=dynamic_profile
)
```

### 3. 共享内存部署

#### 服务器端

```python
from sindre.deploy import python_share_memory

# 创建服务器
server = python_share_memory.SharedMemoryServer("model_server", "model.onnx")

# 启动服务器
server.start()

# 保持运行
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()
```

#### 客户端

```python
from sindre.deploy import python_share_memory

# 创建客户端
client = python_share_memory.SharedMemoryClient("model_server")

# 准备数据
input_data = {
    "input": np.random.rand(1, 3, 224, 224).astype(np.float32)
}

# 执行推理
result = client.infer(input_data)
print(f"共享内存推理结果: {result}")
```

### 4. 系统检测工具

```python
from sindre.deploy import check_tools

# 检测GPU和系统信息
check_tools.check_gpu_info()

# 性能测量工具 - CPU模式
with check_tools.timeit("CPU计算"):
    result = [i**2 for i in range(10**6)]

# 性能测量工具 - GPU模式
import torch
if torch.cuda.is_available():
    with check_tools.timeit("GPU计算", use_torch=True):
        tensor = torch.randn(10000, 10000).cuda()
        result = tensor @ tensor.T
```

**check_gpu_info() 功能**:
- 检测操作系统信息
- 显示CPU核心数、频率、使用率
- 显示内存总量和使用情况
- 检测GPU设备数量和详细信息
- 显示CUDA和cuDNN版本
- 检查硬件支持的数据类型（FP16、BF16、INT8等）

**timeit 上下文管理器**:
- 测量函数执行时间
- 监控内存使用变化
- 支持CPU和GPU模式
- 显示显存使用情况（GPU模式）

## 🚀 高级功能

### 1. 模型优化

#### ONNX优化

```python
# 自动优化
infer.optimizer("optimized.onnx")

# 手动优化选项
optimization_passes = [
    'eliminate_deadend',
    'eliminate_identity',
    'fuse_bn_into_conv',
    'fuse_consecutive_concats'
]
```

#### TensorRT优化

```python
# FP16优化
trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="fp16_model.engine",
    fp16=True
)

# INT8量化
trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="int8_model.engine",
    int8=True
)
```

### 2. 动态形状支持

```python
# ONNX动态形状
infer.dynamic_input_shape("dynamic.onnx", [[None, 3, None, None]])

# TensorRT动态形状
dynamic_profile = {
    "input": [(1, 3, 224, 224), (4, 3, 224, 224), (8, 3, 224, 224)]
}
trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="dynamic.engine",
    dynamic_shape_profile=dynamic_profile
)
```

### 3. 性能监控

```python
# ONNX性能测试
infer.test_performance(loop=100, warmup=10)

# TensorRT性能测试
avg_time = trt_infer.test_performance(loop=100, warmup=10)
print(f"平均推理时间: {avg_time:.3f}ms")

# 内存使用监控
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"内存使用: {memory_usage:.2f} MB")
```

### 4. 错误处理

```python
try:
    result = infer(input_data)
except RuntimeError as e:
    print(f"推理错误: {e}")
    # 检查模型文件
    if not os.path.exists("model.onnx"):
        print("模型文件不存在")
    # 检查输入数据
    if input_data.shape != expected_shape:
        print(f"输入形状不匹配: {input_data.shape} vs {expected_shape}")
except Exception as e:
    print(f"未知错误: {e}")
```

## ⚡ 性能优化

### 1. 内存优化

```python
# 使用内存池
import numpy as np
from contextlib import contextmanager

@contextmanager
def memory_pool():
    """内存池上下文管理器"""
    pool = {}
    try:
        yield pool
    finally:
        pool.clear()

# 使用内存池
with memory_pool() as pool:
    if "input_buffer" not in pool:
        pool["input_buffer"] = np.zeros((1, 3, 224, 224), dtype=np.float32)
    input_data = pool["input_buffer"]
    result = infer(input_data)
```

### 2. 批处理优化

```python
# 批量推理
def batch_inference(infer, data_list, batch_size=4):
    """批量推理"""
    results = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_result = infer(batch)
        results.extend(batch_result)
    return results

# 使用批量推理
data_list = [np.random.rand(1, 3, 224, 224) for _ in range(100)]
results = batch_inference(infer, data_list, batch_size=8)
```

### 3. 多线程优化

```python
import threading
from queue import Queue

class ThreadedInfer:
    """多线程推理"""
    def __init__(self, model_path, num_threads=4):
        self.infer = onnxruntime_deploy.OnnxInfer(model_path)
        self.queue = Queue()
        self.threads = []
        
        for _ in range(num_threads):
            thread = threading.Thread(target=self._worker)
            thread.start()
            self.threads.append(thread)
    
    def _worker(self):
        while True:
            try:
                data, callback = self.queue.get(timeout=1)
                result = self.infer(data)
                callback(result)
            except:
                break
    
    def infer_async(self, data, callback):
        """异步推理"""
        self.queue.put((data, callback))

# 使用多线程推理
threaded_infer = ThreadedInfer("model.onnx", num_threads=4)

def on_result(result):
    print(f"异步推理结果: {result}")

threaded_infer.infer_async(input_data, on_result)
```

## ❓ 常见问题

### Q1: ONNX模型加载失败？

**A**: 检查以下几点：
```python
# 1. 检查模型文件
if not os.path.exists("model.onnx"):
    print("模型文件不存在")

# 2. 检查模型格式
import onnx
try:
    model = onnx.load("model.onnx")
    onnx.checker.check_model(model)
except Exception as e:
    print(f"模型格式错误: {e}")

# 3. 检查推理提供者
available_providers = onnxruntime.get_available_providers()
print(f"可用提供者: {available_providers}")
```

### Q2: TensorRT引擎构建失败？

**A**: 常见解决方案：
```python
# 1. 检查TensorRT版本
import tensorrt as trt
print(f"TensorRT版本: {trt.__version__}")

# 2. 检查CUDA版本
import torch
print(f"CUDA版本: {torch.version.cuda}")

# 3. 减少工作空间大小
trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="model.engine",
    max_workspace_size=1<<30  # 1GB
)
```

### Q3: 推理性能不理想？

**A**: 性能优化建议：
```python
# 1. 使用GPU推理
infer = onnxruntime_deploy.OnnxInfer(
    "model.onnx",
    providers=[('CUDAExecutionProvider', {})]
)

# 2. 启用模型优化
infer.optimizer("optimized.onnx")

# 3. 使用FP16
trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="fp16.engine",
    fp16=True
)
```

### Q4: 内存不足？

**A**: 内存优化方法：
```python
# 1. 减少批处理大小
batch_size = 1  # 从4减少到1

# 2. 使用动态形状
infer.dynamic_input_shape("dynamic.onnx", [[None, 3, None, None]])

# 3. 及时释放内存
import gc
gc.collect()
```

### Q5: 共享内存连接失败？

**A**: 检查连接设置：
```python
# 1. 检查服务器状态
if not server.is_running():
    print("服务器未运行")

# 2. 检查端口占用
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
result = sock.connect_ex(('localhost', 8080))
if result == 0:
    print("端口被占用")

# 3. 检查权限
import os
if not os.access("/dev/shm", os.W_OK):
    print("共享内存权限不足")
```

### Q6: 模型精度下降？

**A**: 精度优化方法：
```python
# 1. 使用FP32而不是FP16
trt_infer.build_engine(
    onnx_path="model.onnx",
    engine_path="fp32.engine",
    fp16=False
)

# 2. 检查量化设置
# 避免过度量化

# 3. 验证推理结果
expected_result = reference_inference(input_data)
actual_result = optimized_inference(input_data)
diff = np.abs(expected_result - actual_result).max()
print(f"最大误差: {diff}")
```

## 📊 性能基准

| 推理引擎 | 模型大小 | 推理时间 | 内存使用 | 适用场景 |
|----------|----------|----------|----------|----------|
| ONNX Runtime (CPU) | 50MB | ~50ms | ~200MB | 开发测试 |
| ONNX Runtime (GPU) | 50MB | ~10ms | ~500MB | 生产环境 |
| TensorRT | 50MB | ~5ms | ~800MB | 高性能需求 |
| 共享内存 | 50MB | ~2ms | ~100MB | 低延迟需求 |

## 🔗 相关链接

- [ONNX Runtime文档](https://onnxruntime.ai/)
- [TensorRT文档](https://docs.nvidia.com/deeplearning/tensorrt/)
- [OpenVINO文档](https://docs.openvino.ai/)
- [共享内存文档](https://docs.python.org/3/library/multiprocessing.html#shared-memory)

---



如有问题，请查看 [常见问题](#常见问题) 或提交 [Issue](https://github.com/SindreYang/sindre/issues)
