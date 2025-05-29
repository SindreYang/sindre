import onnx
import onnxruntime
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class OnnxInfer:
    def __init__(self, onnx_path: str, providers: List[Tuple[str, Dict[str, Any]]] = [('CPUExecutionProvider', {})],enable_log:bool=False) -> None:
        """
        ONNX模型推理类，支持多种推理后端（CPU/GPU等）
        
        参数:
            onnx_path: ONNX模型文件路径
            providers: 推理提供者列表，格式为[(provider_name, options_dict), ...]
        """
        available_providers = onnxruntime.get_available_providers()
        # 检查所有指定的providers是否可用
        for provider, _ in providers:
            if provider not in available_providers:
                raise RuntimeError(f"不支持的推理提供者: {provider}，可用提供者: {available_providers} \n \
                                    dml:pip install onnxruntime-DirectML \t cuda:pip install onnxruntime-gpu \
                                        pip install onnxruntime-gpu[cuda,cudnn]")
        if not os.path.exists(onnx_path):
            raise RuntimeError(f"{onnx_path} 不存在")
        self.onnx_path = onnx_path
        self.providers = providers
        self.session = None
        self.inputs = None
        self.outputs = None
        self.inputs_name = None
        self.outputs_name =None
        self.get_session(onnx_path, providers,enable_log)
        
    def get_session(self, onnx_path: str, providers: List[Tuple[str, Dict[str, Any]]],enable_log:bool=False):
        """加载并验证ONNX模型，创建推理会话"""
        # 验证模型
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print(f"ONNX模型验证失败: {e}")
        
        # 配置会话选项
        sess_opt = onnxruntime.SessionOptions()
        sess_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.enable_cpu_mem_arena = True
        sess_opt.enable_mem_pattern = True
        
        # 设置日志和性能分析
        if enable_log:
            model_dir = os.path.dirname(onnx_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            sess_opt.profile_file_prefix = os.path.join(model_dir, "onnx_profile")
            sess_opt.enable_profiling = True
        
        # 创建推理会话
        self.session = onnxruntime.InferenceSession(onnx_path, sess_options=sess_opt, providers=providers)

        # 获取IO
        inputs=[]
        outputs=[]
        inputs_name = []
        outputs_name =[]
        for inp in self.session.get_inputs():
            data = {}
            data["name"]=inp.name
            data["shape"]=inp.shape
            data["type"]=inp.type
            inputs.append(data)
            inputs_name.append(data["name"])
            print(f"模型内部输入 '{data["name"]}' 形状: {data["shape"]}, 类型: {data["type"]}")
        for  out in self.session.get_outputs():
            data = {}
            data["name"]=out.name
            data["shape"]=out.shape
            data["type"]=out.type
            outputs.append(data)
            outputs_name.append(data["name"])
            print(f"模型内部输出 '{data["name"]}' 形状: {data["shape"]}, 类型: {data["type"]}")
            
            
        self.inputs = inputs
        self.outputs= outputs
        self.inputs_name = inputs_name
        self.outputs_name =outputs_name

    


    
    def __call__(self, inputs: np.ndarray) -> List[np.ndarray]:
        """执行模型推理"""
        if isinstance(inputs, np.ndarray):
            # 如果只提供了一个输入，将其映射到第一个输入节点
            input_feed = {self.inputs_name[0]: inputs}
        elif isinstance(inputs, dict):
            # 如果提供了字典，则使用名称映射
            input_feed = inputs
        else:
            raise ValueError(f"输入必须是numpy数组或字典:{self.inputs}")
            
        # 执行推理
        outputs = self.session.run(self.outputs_name, input_feed)
        return outputs

    
    def optimizer(self,save_onnx):
        """优化并简化ONNX模型"""
        import onnxoptimizer
        from onnxsim import simplify
        
        # 加载模型
        model = onnx.load(self.onnx_path)
        
        # 优化模型
        passes = [
            'eliminate_deadend',
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_pad',
            'eliminate_unused_initializer',
            'fuse_bn_into_conv',
            'fuse_consecutive_concats',
            'fuse_consecutive_log_softmax',
            'fuse_consecutive_reduce_unsqueeze',
            'fuse_consecutive_squeezes',
            'fuse_consecutive_transposes',
            'fuse_matmul_add_bias_into_gemm',
            'fuse_pad_into_conv',
            'fuse_transpose_into_gemm'
        ]
        optimized_model = onnxoptimizer.optimize(model, passes)
        
        # 简化模型
        model_simp, check = simplify(optimized_model)
        if not check:
            print("模型简化失败，返回优化后的模型")
            onnx.save(optimized_model, save_onnx)
        else:
            onnx.save(model_simp, save_onnx)
        print(f"优化后的模型已保存至: {save_onnx}")
        
    def convert_opset_version(self, save_path: str, target_version: int) -> None:
        """
        转换ONNX模型的Opset版本
        :param save_path: 保存路径
        :param target_version: 目标Opset版本（如16）
        """
        model = onnx.load(self.onnx_path)
        # 版本转换（自动处理兼容检查）
        model = onnx.version_converter.convert_version(model, target_version)
        onnx.save(model, save_path)
        print(f"Opset {target_version} 模型已保存至: {save_path}")

    def fix_input_shape(self, save_path: str, input_shapes: list) -> None:
        """
        固定ONNX模型的输入尺寸（支持多输入）
        :param save_path: 保存路径
        :param input_shapes: 输入形状列表，如 [[1,3,416,480], [1,3,416,480]] or [[1,3,416,480]]
        """
        model = onnx.load(self.onnx_path)
        inputs = model.graph.input
        # 校验输入数量匹配
        if len(inputs) != len(input_shapes):
            raise ValueError(
                f"输入节点数({len(inputs)})与形状列表数({len(input_shapes)})不匹配"
            )
        for idx, (input_node, shape) in enumerate(zip(inputs, input_shapes)):
            tensor_type = input_node.type.tensor_type
            # 清除原有动态维度
            del tensor_type.shape.dim[:]
            # 添加固定维度（支持静态形状，如[1,3,416,480]）
            for dim_val in shape:
                tensor_type.shape.dim.append(onnx.helper.make_dimension(None, dim_val))
        onnx.save(model, save_path)
        print(f"固定输入形状模型已保存至: {save_path}")

    def dynamic_input_shape(self, save_path: str, dynamic_dims: list) -> None:
        """
        设置ONNX模型的输入为动态尺寸（支持多输入，None表示动态维度）
        :param dynamic_dims: 动态维度列表，如 [[None, 3, None, 480], [None, 3, None, 480]]
                            每个子列表对应一个输入的维度，None表示该维度动态可变
        """
        model = onnx.load(self.onnx_path)
        inputs = model.graph.input
        if len(inputs) != len(dynamic_dims):
            raise ValueError(f"输入节点数与动态维度列表不匹配: {len(inputs)} vs {len(dynamic_dims)}")
        for in_node, dims in zip(inputs, dynamic_dims):
            tensor_type = in_node.type.tensor_type
            del tensor_type.shape.dim[:]
            for dim in dims:
                # 动态维度（None）或固定维度（数值）
                tensor_type.shape.dim.append(
                    onnx.helper.make_dimension(None, dim) if dim is not None else onnx.helper.make_dimension(None, None)
                )
        onnx.save(model, save_path)
        print(f"动态输入形状模型已保存至: {save_path}")

        
        
        
        
    def test_performance(self, loop: int = 10, warmup: int = 3):
        """测试模型推理速度"""
        import time 
        # 生成随机测试输入
         # 为所有输入生成随机测试数据
        input_feed = {}
        for input_info in self.inputs:
            type_map = {
                'tensor(float)': np.float32,
                'tensor(double)': np.float64,
                'tensor(int32)': np.int32,
                'tensor(int64)': np.int64,
                'tensor(uint8)': np.uint8,
                'tensor(int8)': np.int8,
                'tensor(bool)': np.bool_,
            }
            name = input_info["name"]
            shape = []
            for dim in input_info["shape"]:
                if isinstance(dim, str):
                    shape.append(1)
                elif isinstance(dim, int):
                    shape.append(dim)  # 固定维度直接使用
                else:
                    shape.append(1)  # 未知类型维度默认设为1
            
            dtype = type_map.get( input_info["type"], np.float32)
            
            # 生成符合形状和类型的随机数据
            if dtype == np.float32 or dtype == np.float64:
                input_data = np.random.randn(*shape).astype(dtype)
            elif dtype == np.int32 or dtype == np.int64:
                input_data = np.random.randint(0, 100, size=shape, dtype=dtype)
            else:
                print(f"警告: 不支持的输入类型 {input_info['type']}，使用float32代替")
                input_data = np.random.randn(*shape).astype(np.float32)
                
            input_feed[name] = input_data
            print(f"测试为输入 '{name}' 生成测试数据，形状: {shape}, 类型: {dtype}")
        
        
        # 预热运行
        print(f"开始预热 ({warmup} 次)...")
        start = time.time()
        for _ in range(warmup):
            self.session.run(None, input_feed)
        print(f"预热3次耗时:  {time.time()-start}秒")
            
        # 性能测试
        print(f"开始性能测试 ({loop} 次)...")
        start_sum = time.time()
        for i in range(loop):
            start = time.time()
            outputs = self.session.run(None, input_feed)
            elapsed = time.time() - start
            
            # 打印第一个输出的信息
            print(f"第 {i+1}/{loop} 次推理，输出形状: {[out.shape for out in outputs]}，耗时: {elapsed:.4f}秒")
            
        avg_time = (time.time() - start_sum) / loop
        print(f"平均推理耗时: {avg_time:.4f}秒 ({1/avg_time:.2f} FPS)")
    
        
 
  
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    onnx_path = r"../test/data/SegformerMITB0.onnx"
    engine_path = r"../test/data/segformer.engine"
    img_path = r"../test/data/tooth_test.bmp"
    save_path =r"../test/data/masked.bmp"
    import cv2
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
    Infer = OnnxInfer(onnx_path)
    #Infer.optimizer("../test/data/SegformerMITB0_opt.onnx")
    outputs = Infer(input_data)
    Infer.test_performance()
    # 处理输出结果
    for data in outputs:
        print(f"输出 '{np.unique(data)}' 形状: {data.shape}, 类型: {data.dtype}")
        cv2.imwrite(save_path, (data[0][0]*50).astype(np.uint8))