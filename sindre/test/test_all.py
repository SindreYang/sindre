"""
sindre库完整测试套件
运行所有模块的测试用例
"""

import pytest
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def test_import_all_modules():
    """测试所有主要模块的导入"""
    modules_to_test = [
        "sindre",
        "sindre.general",
        "sindre.general.logs",
        "sindre.lmdb",
        "sindre.lmdb.pylmdb",
        "sindre.lmdb.tools",
        "sindre.report",
        "sindre.report.report",
        "sindre.utils3d",
        "sindre.win_tools",
        "sindre.deploy"
    ]
    
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ 成功导入模块: {module_name}")
        except ImportError as e:
            print(f"✗ 导入失败模块: {module_name} - {e}")
            # 对于某些可选模块，导入失败是正常的
            if "win_tools" in module_name and sys.platform.lower() != "win32":
                print(f"  跳过Windows特定模块: {module_name}")
            elif "utils3d" in module_name:
                print(f"  跳过依赖密集型模块: {module_name}")
            else:
                pytest.fail(f"关键模块导入失败: {module_name}")


def test_basic_functionality():
    """测试基本功能"""
    # 测试sindre库的基本导入
    import sindre
    
    # 验证主要属性存在
    assert hasattr(sindre, 'lmdb')
    assert hasattr(sindre, 'report')
    assert hasattr(sindre, 'win_tools')
    assert hasattr(sindre, 'utils3d')
    
    print("✓ sindre库基本功能正常")


def test_platform_compatibility():
    """测试平台兼容性"""
    import sys
    
    print(f"当前平台: {sys.platform}")
    print(f"Python版本: {sys.version}")
    
    # 检查平台特定的模块
    if sys.platform.lower() == "win32":
        print("✓ 在Windows平台上运行")
        try:
            import sindre.win_tools
            print("✓ Windows工具模块可用")
        except ImportError:
            print("✗ Windows工具模块不可用")
    else:
        print("✓ 在非Windows平台上运行")
        print("  跳过Windows特定功能测试")


def test_dependencies():
    """测试依赖项"""
    required_deps = [
        "numpy",
        "lmdb",
        "msgpack",
        "tqdm"
    ]
    
    optional_deps = [
        "torch",
        "vedo",
        "scikit-learn",
        "PIL",
        "loguru"
    ]
    
    print("检查必需依赖:")
    for dep in required_deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - 必需依赖缺失")
            pytest.fail(f"必需依赖缺失: {dep}")
    
    print("检查可选依赖:")
    for dep in optional_deps:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"- {dep} - 可选依赖缺失")


def test_file_structure():
    """测试文件结构"""
    import os
    
    # 检查主要目录结构
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    required_dirs = [
        "general",
        "lmdb", 
        "report",
        "utils3d",
        "win_tools",
        "deploy",
        "test"
    ]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(base_path, dir_name)
        if os.path.exists(dir_path):
            print(f"✓ 目录存在: {dir_name}")
        else:
            print(f"✗ 目录缺失: {dir_name}")
            # 某些目录可能不存在，这不是致命错误
            if dir_name in ["win_tools", "deploy"]:
                print(f"  跳过可选目录: {dir_name}")
    
    # 检查__init__.py文件
    init_files = [
        os.path.join(base_path, "__init__.py"),
        os.path.join(base_path, "general", "__init__.py"),
        os.path.join(base_path, "lmdb", "__init__.py"),
        os.path.join(base_path, "report", "__init__.py"),
        os.path.join(base_path, "utils3d", "__init__.py")
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"✓ __init__.py存在: {os.path.basename(os.path.dirname(init_file))}")
        else:
            print(f"✗ __init__.py缺失: {os.path.basename(os.path.dirname(init_file))}")


def test_test_files():
    """测试测试文件"""
    import os
    
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查测试文件
    test_files = [
        "test_general.py",
        "test_lmdb.py", 
        "test_report.py",
        "test_utils3d.py",
        "test_win_tools.py",
        "test_deploy.py",
        "test_all.py"
    ]
    
    for test_file in test_files:
        test_path = os.path.join(test_dir, test_file)
        if os.path.exists(test_path):
            print(f"✓ 测试文件存在: {test_file}")
        else:
            print(f"✗ 测试文件缺失: {test_file}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("开始运行sindre库完整测试套件")
    print("=" * 60)
    
    # 运行各个测试函数
    test_import_all_modules()
    print()
    
    test_basic_functionality()
    print()
    
    test_platform_compatibility()
    print()
    
    test_dependencies()
    print()
    
    test_file_structure()
    print()
    
    test_test_files()
    print()
    
    print("=" * 60)
    print("测试套件运行完成")
    print("=" * 60)


if __name__ == "__main__":
    # 如果直接运行此文件，执行完整测试套件
    run_all_tests()
    
    # 然后运行pytest
    pytest.main([__file__, "-v"]) 