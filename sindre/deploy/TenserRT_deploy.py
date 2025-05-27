

import tensorrt as trt
import numpy as np
import os
try:
    #from cuda import cuda, cudart 
    import cuda.bindings.runtime as cudart
    import cuda.bindings.driver  as cuda
except ImportError:
    raise RuntimeError("需安装 pip install cuda-python")
if int(trt.__version__.split('.')[0])< 10:
    raise RuntimeError(f"需要TensorRT 10或更高版本，当前版本是{trt.__version__}")





class TRTInfer:
    def __init__(self):
        self.trt_logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.trt_logger)
        self.engine = None
        self.context = None
        self.inputs = None
        self.outputs = None
        self.allocated_buffers = False
        self.stream = None

    def __del__(self):
        if not getattr(self, '_is_shutdown', True):
            self._free_buffers()

    def cpu_gpu_map(self,tensor_name):
        import ctypes
        info={}
        # 计算内存大小与数据类型
        shape = self.engine.get_tensor_shape(tensor_name)
        trt_type =self.engine.get_tensor_dtype(tensor_name)
        size = trt.volume(shape)
        info["name"]=tensor_name
        info["shape"]=shape
        info["trt_type"]=trt_type
        # 分配主机和设备内存
        try:
            #  numpy 支持的类型（如FP32、FP16、INT8），转换为 numpy 数据类型
            dtype = np.dtype(trt.nptype(trt_type))
            nbytes = size * dtype.itemsize
            host_mem = self.check_cuda(cudart.cudaMallocHost(nbytes))
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            info["host"] = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
            info["device"] = self.check_cuda(cudart.cudaMalloc(nbytes))
            info["nbytes"] = nbytes
            
        except TypeError: 
            #  处理numpy不支持的类型（如BF16、FP8、INT4)，直接分配字节数组
            size = int(size * trt_type.itemsize)
            dtype = np.dtype(np.uint8)
            nbytes = size * dtype.itemsize
            host_mem = self.check_cuda(cudart.cudaMallocHost(nbytes))
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            info["host"] = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
            info["device"] = self.check_cuda(cudart.cudaMalloc(nbytes))
            info["nbytes"] = nbytes
        return info
        
    
    
    def check_cuda(self,call):
        err, res = call[0], call[1:]
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError("Cuda Error: {}".format(err))
        if isinstance(err, cudart.cudaError_t):
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("Cuda Runtime Error: {}".format(err))
        else:
            raise RuntimeError("Unknown error type: {}".format(err))
        if len(res) == 1:
            res = res[0]
        return res

        
    def _allocate_buffers(self):
        """分配GPU和主机内存缓冲区"""
        if self.allocated_buffers:
            self._free_buffers()
            
        try:
            inputs = []
            outputs = []
            bindings = []
            self.stream = self.check_cuda(cudart.cudaStreamCreate())
            tensor_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            for binding in tensor_names:
                # 获取主机和设备内存对
                bindingMemory = self.cpu_gpu_map(binding)
                # 构建绑定列表与输入输出分组
                bindings.append(int(bindingMemory["device"]))  # 添加设备内存地址到绑定列表
                if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    inputs.append(bindingMemory)
                    print(f"输入: {bindingMemory["name"]}, 形状: {bindingMemory["shape"]}, 类型: {bindingMemory["trt_type"]}")
                else:
                    outputs.append(bindingMemory)
                    print(f"输出: {bindingMemory["name"]}, 形状: {bindingMemory["shape"]}, 类型: {bindingMemory["trt_type"]}")

            self.inputs = inputs
            self.outputs =  outputs
            self.bindings = bindings
            self.allocated_buffers = True
                
         
        except Exception as e:
            print(f"内存分配失败: {e}")
            self._free_buffers()
            raise

    def _free_buffers(self):
        """释放已分配的内存缓冲区"""
        try:
            for mem in (self.inputs or []) + (self.outputs or []):
                self.check_cuda(cudart.cudaFree(mem["device"]))
                self.check_cuda(cudart.cudaFreeHost(mem["host"].ctypes.data))
            if self.stream:
                self.check_cuda(cudart.cudaStreamDestroy(self.stream))
                
            self.inputs = None
            self.outputs = None
            self.bindings = None
            self.stream = None
            self.allocated_buffers = False
        except Exception as e:
            print(f"释放资源时出错: {e}")


    def __call__(self, data):
        """执行推理"""
        if self.engine is None or self.context is None:
            raise RuntimeError("请先加载模型")
            
        try:
            # 设置输入数据
            if isinstance(data, np.ndarray):
                data = [data]
                
            if len(data) != len(self.inputs):
                raise ValueError(f"输入数量不匹配。期望 {len(self.inputs)}，但得到 {len(data)}")
                
            for i, (input_array, input_info) in enumerate(zip(data, self.inputs)):
                # 确保输入数据形状匹配
                expected_shape = input_info["shape"]
                if input_array.shape != tuple(expected_shape):
                    raise ValueError(f"输入 {i} 形状不匹配。期望 {expected_shape}，但得到 {input_array.shape}")
                # 确保数据类型匹配
                expected_dtype = trt.nptype(input_info["trt_type"])
                if input_array.dtype != expected_dtype:
                    input_array = input_array.astype(expected_dtype)
                # 复制数据到主机内存
                input_info["host"] = input_array.flatten()
            
            # 设置上下文张量地址
            num_io = self.engine.num_io_tensors
            for i in range(num_io):
                self.context.set_tensor_address(self.engine.get_tensor_name(i),  self.bindings[i])

            # 数据拷贝到GPU
            kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
            [self.check_cuda(cudart.cudaMemcpyAsync(inp["device"], inp["host"], inp["nbytes"], kind, self.stream)) for inp in self.inputs]
            
            # 执行推理
            self.context.execute_async_v3(stream_handle=self.stream)
            
            # 结果转到cpu
            kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
            [self.check_cuda(cudart.cudaMemcpyAsync(out["host"], out["device"], out["nbytes"], kind, self.stream)) for out in self.outputs]
            
            # 同步流
            self.check_cuda(cudart.cudaStreamSynchronize(self.stream))
            
            return [out["host"].reshape(out["shape"]) for out in self.outputs]
            
        except Exception as e:
            print(f"推理过程中出错: {e}")
            raise
    def test_performance(self, loop: int = 10, warmup: int = 3) -> float:
        """
        测试推理性能（自动生成随机输入）
        
        参数:
            loop: 正式测试循环次数
            warmup: 预热次数
        """
        import time
        # 生成随机测试输入
        test_inputs = []
        for inp_info in self.inputs:
            shape = inp_info["shape"]
            trt_type = inp_info["trt_type"]
            dtype = trt.nptype(trt_type)
            
            # 生成随机数据（支持常见类型）
            if dtype == np.float32:
                data = np.random.randn(*shape).astype(dtype)
            elif dtype == np.float16:
                data = (np.random.randn(*shape) * 100).astype(np.float32).view(np.float16)
            elif dtype == np.int32:
                data = np.random.randint(0, 100, size=shape, dtype=dtype)
            elif dtype == np.uint8:
                data = np.random.randint(0, 256, size=shape, dtype=dtype)
            else:
                raise NotImplementedError(f"不支持的测试数据类型: {dtype}")
            
            test_inputs.append(data)
        
        # 预热运行
        print(f"开始预热 ({warmup} 次)...")
        start = time.time()
        for _ in range(warmup):
            _ = self(test_inputs)
        print(f"预热3次耗时:  {time.time()-start}秒")
            
        # 性能测试
        print(f"开始性能测试 ({loop} 次)...")
        start_time = time.time()
        for i in range(loop):
            start = time.time()
            _ = self(test_inputs)
            elapsed = time.time() - start
            print(f"第 {i+1}/{loop} 次推理，耗时: {elapsed:.4f} 秒")
            
        avg_time = (time.time() - start_time) / loop
        print(f"平均推理耗时: {avg_time:.4f} 秒 ({1/avg_time:.2f} FPS)")

        print("测试性能，将占用内存空间，如涉及有关变量，请deepcopy()")
        

    def load_model(self, engine_path):
        """从TensorRT引擎文件加载模型"""
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"引擎文件不存在: {engine_path}")
            
        try:
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()
                
            self.engine = self.runtime.deserialize_cuda_engine(serialized_engine)
            if self.engine is None:
                raise RuntimeError("引擎反序列化失败")
                
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("无法创建执行上下文")


            # 分配内存缓冲区
            self._allocate_buffers()
            print(f"模型加载成功: {engine_path}")
         
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            self.engine = None
            self.context = None
            raise

    def build_engine(self, onnx_path, engine_path, max_workspace_size=4<<30, 
                    fp16=False,dynamic_shape_profile=None, hardware_compatibility="",optimization_level=3):
        """
        从ONNX模型构建TensorRT引擎
        
        参数:
            onnx_path (str): ONNX模型路径
            engine_path (str, optional): 引擎保存路径
            max_workspace_size (int, optional): 最大工作空间大小，默认为4GB
            fp16_mode (bool, optional): 是否启用FP16精度
            dynamic_shape_profile (dict, optional): 动态形状配置，格式为:
                {
                    "input_name": {
                        "min": (1, 3, 224, 224),  # 最小形状
                        "opt": (4, 3, 224, 224),  # 优化形状
                        "max": (8, 3, 224, 224)   # 最大形状
                    }
                }
            hardware_compatibility (str, optional): 硬件兼容性级别，可选值:
                - "None": 默认(最快)
                - "same_sm": 相同计算能力(其次)
                - "ampere_plus": Ampere及更高架构(最慢)
                Pascal(10系)、Volta(V100)、Turing(20系)、Ampere(30系)、
                Ada(40系)、Hopper(H100)、Blackwell(50系)

            optimization_level (int): 优化级别,默认最优级别3;
                ・等级 0：通过禁用动态内核生成并选择执行成功的第一个策略，实现最快的编译。这也不会考虑计时缓存。
                ・等级 1：可用策略按启发式方法排序，但仅测试排名靠前的策略以选择最佳策略。如果生成动态内核，其编译优化程度较低。
                ・等级 2：可用策略按启发式方法排序，但仅测试最快的策略以选择最佳策略。
                ・等级 3：应用启发式方法，判断静态预编译内核是否适用，或者是否必须动态编译新内核。
                ・等级 4：始终编译动态内核。
                ・等级 5：始终编译动态内核，并将其与静态内核进行比较。
                
        """
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX文件不存在: {onnx_path}")
            
        try:
            # 创建构建器和网络
            builder = trt.Builder(self.trt_logger)
            network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            network = builder.create_network(network_flags)
            
            # 创建解析器并解析ONNX模型
            parser = trt.OnnxParser(network, self.trt_logger)
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    error_msgs = []
                    for error in range(parser.num_errors):
                        error_msgs.append(f"解析错误 {error}: {parser.get_error(error)}")
                    raise RuntimeError("\n".join(error_msgs))
            
            # 配置构建器
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
            config.builder_optimization_level = optimization_level # 优化级别
           
            
            
            
            # 设置硬件兼容性
            if hardware_compatibility == "ampere_plus":
                config.hardware_compatibility_level=trt.HardwareCompatibilityLevel.AMPERE_PLUS
                print("启用Ampere+硬件兼容性模式")
            elif hardware_compatibility == "same_sm":
                config.hardware_compatibility_level=trt.HardwareCompatibilityLevel.SAME_COMPUTE_CAPABILITY
                print("启用相同计算能力兼容性模式:计算能力:https://developer.nvidia.cn/cuda-gpus")


            
            # 设置精度模式
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                print("启用FP16模式")

            # 构建缓存处理
            timing_cache = None
            cache_path = engine_path+".cache"
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = f.read()
                        timing_cache = config.create_timing_cache(cache_data)
                        config.set_timing_cache(timing_cache, ignore_mismatch=False)
                        print(f"已加载构建缓存: {cache_path}")
                except Exception as e:
                    print(f"加载缓存失败: {e}，将创建新缓存")
                    timing_cache = config.create_timing_cache(b"")
            else:
                timing_cache = config.create_timing_cache(b"")
                print("创建新的构建缓存")
            config.set_timing_cache(timing_cache, ignore_mismatch=False)

  
            
            # 配置动态形状
            if dynamic_shape_profile:
                if not isinstance(dynamic_shape_profile, dict):
                    raise ValueError("dynamic_shape_profile必须是字典类型")
                    
                profile = builder.create_optimization_profile()
                for input_name, shapes in dynamic_shape_profile.items():
                    min_shape = shapes.get("min")
                    opt_shape = shapes.get("opt")
                    max_shape = shapes.get("max")
                    
                    if not all([min_shape, opt_shape, max_shape]):
                        raise ValueError(f"动态形状配置缺少必要参数: {input_name}")
                        
                    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
                    print(f"为输入 '{input_name}' 设置动态形状范围: {min_shape} - {opt_shape} - {max_shape}")
                    
                config.add_optimization_profile(profile)
            

            # 构建引擎
            print("正在构建TensorRT引擎...")
            engine = builder.build_serialized_network(network, config)
            if engine is None:
                raise RuntimeError("引擎构建失败")
            else:
                # 保存缓存
                updated_cache = config.get_timing_cache()
                cache_data = updated_cache.serialize()
                with open(cache_path, 'wb') as f:
                    f.write(cache_data)
                print(f"已保存构建缓存: {cache_path}")
                # 保存引擎到文件
                with open(engine_path, "wb") as f:
                    f.write(engine)
                print(f"引擎已保存到: {engine_path}")
                return self.load_model(engine_path)
            
        except Exception as e:
            print(f"构建引擎时出错: {e}")
            self.engine = None
            self.context = None
            raise 

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    onnx_path = r"../test/data/SegformerMITB0.onnx"
    engine_path = r"../test/data/segformer.engine"
    img_path = r"../test/data/tooth_test.bmp"
    save_path =r"../test/data/masked.bmp"
    import cv2
    import copy
    trt_infer = TRTInfer()
    # # 构建或加载引擎
    trt_infer.build_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        max_workspace_size=2 << 30,  
        fp16=True,
        #hardware_compatibility="same_sm"
    )
    # 生成测试输入
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 192))
    mean = np.array([
        128.5962491,
        152.23387713,
        193.64875669,
        ], dtype=np.float32)
    std = np.array([
        37.50885872,
        30.88513081,
        27.15953715
        ], dtype=np.float32)
    
    img=(img-mean)/std
    input_data = np.array(img, dtype=np.float32).transpose((2, 0, 1))[None]
    # 执行推理
    outputs =copy.deepcopy(trt_infer(input_data.astype(np.float16)))
    trt_infer.test_performance()
    # 处理输出结果
    for data in outputs:
        print(f"输出 '{np.unique(data)}' 形状: {data.shape}, 类型: {data.dtype}")
        cv2.imwrite(save_path, (data[0][0]*50))
    
    
