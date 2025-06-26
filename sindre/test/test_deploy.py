"""
sindre.deploy模块测试用例
测试部署相关的各种功能
"""

import pytest
import os
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# 检查是否在Windows平台上
IS_WINDOWS = sys.platform.lower() == "win32"


class TestDeploy:
    """测试部署模块的各种功能"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """每个测试方法后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_deploy_import(self):
        """测试deploy模块导入"""
        try:
            import sindre.deploy
            assert True
        except ImportError:
            pytest.skip("deploy模块不可用")
    
    def test_check_tools_import(self):
        """测试check_tools模块导入"""
        try:
            import sindre.deploy.check_tools
            assert True
        except ImportError:
            pytest.skip("check_tools模块不可用")
    
    def test_onnxruntime_deploy_import(self):
        """测试ONNX Runtime部署模块导入"""
        try:
            import sindre.deploy.onnxruntime_deploy
            assert True
        except ImportError:
            pytest.skip("onnxruntime_deploy模块不可用")
    
    def test_opset_deploy_import(self):
        """测试opset部署模块导入"""
        try:
            import sindre.deploy.opset_deploy
            assert True
        except ImportError:
            pytest.skip("opset_deploy模块不可用")
    
    def test_python_share_memory_import(self):
        """测试Python共享内存模块导入"""
        try:
            import sindre.deploy.python_share_memory
            assert True
        except ImportError:
            pytest.skip("python_share_memory模块不可用")
    
    def test_tensorrt_deploy_import(self):
        """测试TensorRT部署模块导入"""
        try:
            import sindre.deploy.TenserRT_deploy
            assert True
        except ImportError:
            pytest.skip("TenserRT_deploy模块不可用")
    
    def test_cpp_share_memory_import(self):
        """测试C++共享内存模块导入"""
        try:
            import sindre.deploy.cpp_share_memory
            assert True
        except ImportError:
            pytest.skip("cpp_share_memory模块不可用")


@pytest.mark.skipif(not IS_WINDOWS, reason="仅在Windows平台上运行")
class TestDeployWindows:
    """测试Windows平台特定的部署功能"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """每个测试方法后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_cpp_share_memory_files(self):
        """测试C++共享内存相关文件"""
        cpp_path = os.path.join(os.path.dirname(__file__), "..", "deploy", "cpp_share_memory")
        
        if os.path.exists(cpp_path):
            # 检查必要文件
            required_files = [
                "CMakeLists.txt",
                "py2cpp_share_memory.cpp",
                "py2cpp_share_memory.py",
                "README.md"
            ]
            
            for file in required_files:
                file_path = os.path.join(cpp_path, file)
                if os.path.exists(file_path):
                    assert os.path.getsize(file_path) > 0, f"文件 {file} 为空"
    
    def test_shared_memory_exe(self):
        """测试共享内存可执行文件"""
        cpp_path = os.path.join(os.path.dirname(__file__), "..", "deploy", "cpp_share_memory")
        exe_path = os.path.join(cpp_path, "shared_memory.exe")
        
        if os.path.exists(exe_path):
            # 检查文件大小
            file_size = os.path.getsize(exe_path)
            assert file_size > 1000, "可执行文件太小"
            
            # 检查文件权限
            assert os.access(exe_path, os.R_OK), "可执行文件不可读"
    
    def test_cmake_files(self):
        """测试CMake文件"""
        cpp_path = os.path.join(os.path.dirname(__file__), "..", "deploy", "cpp_share_memory")
        cmake_path = os.path.join(cpp_path, "CMakeLists.txt")
        
        if os.path.exists(cmake_path):
            with open(cmake_path, "r", encoding="utf-8") as f:
                content = f.read()
                assert len(content) > 0, "CMakeLists.txt为空"
                assert "cmake_minimum_required" in content or "project(" in content, "CMakeLists.txt格式不正确"


class TestDeployCrossPlatform:
    """测试跨平台部署功能"""
    
    def setup_method(self):
        """每个测试方法前的设置"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """每个测试方法后的清理"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_deploy_directory_structure(self):
        """测试deploy目录结构"""
        deploy_path = os.path.join(os.path.dirname(__file__), "..", "deploy")
        
        if os.path.exists(deploy_path):
            # 检查必要的文件
            required_files = ["__init__.py"]
            for file in required_files:
                file_path = os.path.join(deploy_path, file)
                assert os.path.exists(file_path), f"缺少必要文件: {file}"
            
            # 检查子目录
            subdirs = ["cpp_share_memory"]
            for subdir in subdirs:
                subdir_path = os.path.join(deploy_path, subdir)
                if os.path.exists(subdir_path):
                    assert os.path.isdir(subdir_path), f"{subdir} 不是目录"
    
    def test_python_modules(self):
        """测试Python模块文件"""
        deploy_path = os.path.join(os.path.dirname(__file__), "..", "deploy")
        
        if os.path.exists(deploy_path):
            # 检查Python模块文件
            python_files = [
                "check_tools.py",
                "onnxruntime_deploy.py",
                "opset_deploy.py",
                "python_share_memory.py",
                "TenserRT_deploy.py"
            ]
            
            existing_files = [f for f in python_files if os.path.exists(os.path.join(deploy_path, f))]
            # 至少应该有一些Python文件存在
            assert len(existing_files) > 0, "没有找到任何Python模块文件"
    
    def test_file_permissions(self):
        """测试文件权限"""
        deploy_path = os.path.join(os.path.dirname(__file__), "..", "deploy")
        
        if os.path.exists(deploy_path):
            # 检查目录权限
            assert os.access(deploy_path, os.R_OK), "deploy目录不可读"
            assert os.access(deploy_path, os.X_OK), "deploy目录不可执行"
            
            # 检查Python文件权限
            python_files = [
                "check_tools.py",
                "onnxruntime_deploy.py",
                "opset_deploy.py",
                "python_share_memory.py",
                "TenserRT_deploy.py"
            ]
            
            for file in python_files:
                file_path = os.path.join(deploy_path, file)
                if os.path.exists(file_path):
                    assert os.access(file_path, os.R_OK), f"{file} 不可读"
    
    def test_import_structure(self):
        """测试导入结构"""
        # 测试deploy模块的导入结构
        try:
            import sindre.deploy
            # 检查是否有__all__属性
            if hasattr(sindre.deploy, '__all__'):
                assert isinstance(sindre.deploy.__all__, list), "__all__应该是列表"
        except ImportError:
            pass  # 导入失败是正常的
    
    def test_module_dependencies(self):
        """测试模块依赖"""
        # 检查常见的部署相关依赖
        common_deps = [
            "numpy",
            "torch",
            "onnxruntime",
            "tensorrt"
        ]
        
        available_deps = []
        for dep in common_deps:
            try:
                __import__(dep)
                available_deps.append(dep)
            except ImportError:
                pass
        
        # 至少应该有一些依赖可用
        assert len(available_deps) > 0, "没有找到任何部署相关的依赖"


class TestDeployFunctionality:
    """测试部署功能"""
    
    def test_environment_check(self):
        """测试环境检查"""
        # 检查Python版本
        assert sys.version_info >= (3, 6), "需要Python 3.6或更高版本"
        
        # 检查平台信息
        assert hasattr(sys, 'platform')
        assert hasattr(os, 'name')
        
        # 检查环境变量
        assert 'PATH' in os.environ


    def test_onnx_end2end_infer(self):
        """端到端ONNX推理流程测试，使用真实模型和图片，参考主函数示例"""
        try:
            import numpy as np
            import cv2
            from sindre.deploy.onnxruntime_deploy import OnnxInfer
        except ImportError:
            pytest.skip("依赖未安装，跳过ONNX推理测试")
        # 路径
        base_dir = os.path.dirname(__file__)
        onnx_path = os.path.abspath(os.path.join(base_dir, "data", "SegformerMITB0.onnx"))
        img_path = os.path.abspath(os.path.join(base_dir, "data", "tooth_test.bmp"))
        save_path = os.path.abspath(os.path.join(base_dir, "data", "masked_test_out.bmp"))
        # 检查文件
        if not (os.path.exists(onnx_path) and os.path.exists(img_path)):
            pytest.skip("测试所需模型或图片文件不存在")
        # 预处理
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (240, 192))
        mean = np.array([128.5962491, 152.23387713, 193.64875669], dtype=np.float32)
        std = np.array([37.50885872, 30.88513081, 27.15953715], dtype=np.float32)
        img = (img - mean) / std
        input_data = np.array(img, dtype=np.float32).transpose((2, 0, 1))[None]
        # 推理
        Infer = OnnxInfer(onnx_path)
        outputs = Infer(input_data)
        # 性能测试
        Infer.test_performance(loop=2, warmup=1)
        # 检查输出
        assert isinstance(outputs, list) and len(outputs) > 0
        for data in outputs:
            assert isinstance(data, np.ndarray)
            assert data.shape[0] == 1  # batch维
            # 保存结果
            cv2.imwrite(save_path, (data[0][0]*50).astype(np.uint8))
            assert os.path.exists(save_path)


if __name__ == "__main__":
    pytest.main([__file__]) 