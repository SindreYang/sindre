def check_gpu_info():
    """检测系统信息"""
    import torch
    import psutil
    import platform
    # 获取CPU信息
    try:
        cpu_count = psutil.cpu_count(logical=False)
        logical_cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else "N/A"
        cpu_percent = psutil.cpu_percent(interval=1)
        os_name = platform.system()
        os_version = platform.version()
        print(f"\n=== 系统信息 ===")
        print(f'操作系统名称: {os_name}')
        print(f'操作系统版本: {os_version}')
        print(f"\n=== CPU 信息 ===")
        print(f"CPU型号: {platform.processor()}")
        print(f"物理CPU核心数: {cpu_count}")
        print(f"逻辑CPU核心数: {logical_cpu_count}")
        print(f"CPU当前频率: {cpu_freq:.2f} MHz" if cpu_freq != "N/A" else "CPU频率: N/A")
        print(f"CPU使用率: {cpu_percent}%")
        
        # 获取内存信息
        mem = psutil.virtual_memory()
        print("\n内存信息:")
        print(f"总内存: {mem.total / (1024**3):.2f} GB")
        print(f"已使用: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
        print(f"可用内存: {mem.available / (1024**3):.2f} GB")
    except Exception as e:
        print(f"获取CPU/内存信息时出错: {e}")

    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        print(f"\n发现 {torch.cuda.device_count()} 个可用的GPU设备")
        
        # 遍历所有GPU设备并显示信息
        for i in range(torch.cuda.device_count()):
            print(f"\n=== GPU {i} 信息 ===")
            
            try:
                print(f"设备名称: {torch.cuda.get_device_name(i)}")
                major, minor = torch.cuda.get_device_capability(i)
                print(f"设备能力: {major}.{minor}")
                print(f"显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024:.2f} MB")
                
                # 获取当前GPU的内存使用情况
                print("\n显存使用情况:")
                print(f"已分配: {torch.cuda.memory_allocated(i) / 1024 / 1024:.2f} MB")
                print(f"已缓存: {torch.cuda.memory_reserved(i) / 1024 / 1024:.2f} MB")
                
                
                # 检查CUDA版本
                print(f"\nCUDA版本: {torch.version.cuda}")
                
                # 检查cuDNN版本
                print(f"cuDNN版本: {torch.backends.cudnn.version()}")

                # 检查ROCm支持（AMD GPU）
                if hasattr(torch.version, 'hip'):
                    print(f"\nROCm版本: {torch.version.hip}")
                
                # 显示GPU支持的数据类型
                print("\n=== 硬件支持的数据类型 ===")
                print("浮点类型:")
                print(f"- FP16 (torch.float16): {major >= 5}")
                print(f"- BF16 (torch.bfloat16): {torch.cuda.is_bf16_supported()}")
                
                print("\n整数类型:")
                print(f"- INT8 (torch.int8): {major >= 6}")
                print(f"- UINT8 (torch.uint8): {major >= 6}")
                
                print("\n特殊计算支持:")
                print(f"- Tensor Cores: {major >= 7}")
                if {major >= 7}:
                    print(f"\t- 支持 FP16 Tensor Cores")
                    if torch.cuda.is_bf16_supported():
                        print(f"\t- 支持 BF16 Tensor Cores")
            except Exception as e:
                print(f"获取GPU {i} 信息时出错: {e}")
        
        print("\n检测完成")
    else:
        print("\n未发现可用的GPU设备")
    
    
    



class timeit_tool:
    """测量函数执行时间的上下文管理器类，附加内存监控功能"""
    def __init__(self, use_torch: bool = False) -> None:
        import psutil
        self.use_torch = use_torch
        self.elapsed_time = None
        self.process = psutil.Process()
        
    def __enter__(self):
        # 记录初始内存状态
        self.initial_mem = self.process.memory_info().rss
        
        # CUDA相关初始化
        if self.use_torch:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available but use_torch=True")
            torch.cuda.reset_peak_memory_stats()  # 重置显存统计
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            import time
            self.start_time = time.time()
            
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 计算时间消耗
        if self.use_torch:
            import torch
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start_event.elapsed_time(self.end_event)
        else:
            import time
            self.elapsed_time = (time.time() - self.start_time) * 1000  # 转为毫秒

        # 计算内存变化
        current_mem = self.process.memory_info().rss
        mem_usage = (current_mem - self.initial_mem) / (1024 ** 2)  # 转为MB

        # 构建输出信息
        output = [
            f"Cost time: {self.elapsed_time:.3f} ms",
            f"Memory Δ: {mem_usage:+.2f} MB"
        ]

        # 显存监控（仅限PyTorch CUDA模式）
        if self.use_torch:
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转为MB
            output.append(f"GPU Memory: {gpu_mem:.2f} MB")

        print(" | ".join(output))

def timeit(use_torch: bool = False):
    """测量函数执行时间的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer = timeit_tool(use_torch=use_torch)
            with timer:
                result = func(*args, **kwargs)
            print(f"Function '{func.__name__}' executed in {timer.elapsed_time:.3f} ms")
            return result
        return wrapper
    return decorator
            
            

if __name__ == "__main__":
    check_gpu_info()   
    # CPU模式示例
    with timeit_tool():
        _ = [i**2 for i in range(10**6)]

    # GPU模式示例
    import torch
    if torch.cuda.is_available():
        with timeit_tool(use_torch=True):
            tensor = torch.randn(10000, 10000).cuda()
            _ = tensor @ tensor.T 