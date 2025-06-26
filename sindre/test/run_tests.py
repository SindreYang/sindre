#!/usr/bin/env python3
"""
Sindre库测试运行脚本
提供便捷的测试运行方式
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    if description:
        print(f"运行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失败: {e}")
        return False


def check_dependencies():
    """检查测试依赖"""
    print("检查测试依赖...")
    
    required_packages = ["pytest"]
    optional_packages = ["pytest-cov", "pytest-xdist", "pytest-timeout"]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_optional.append(package)
    
    if missing_required:
        print(f"缺少必需依赖: {', '.join(missing_required)}")
        print("请运行: pip install pytest")
        return False
    
    if missing_optional:
        print(f"缺少可选依赖: {', '.join(missing_optional)}")
        print("可选安装: pip install pytest-cov pytest-xdist pytest-timeout")
    
    return True


def run_basic_tests():
    """运行基本测试"""
    return run_command(
        ["python", "-m", "pytest", "-v"],
        "基本测试"
    )


def run_coverage_tests():
    """运行覆盖率测试"""
    return run_command(
        ["python", "-m", "pytest", "--cov=sindre", "--cov-report=term-missing", "-v"],
        "覆盖率测试"
    )


def run_specific_module(module):
    """运行特定模块测试"""
    test_file = f"test_{module}.py"
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return False
    
    return run_command(
        ["python", "-m", "pytest", test_file, "-v"],
        f"{module}模块测试"
    )


def run_windows_tests():
    """运行Windows特定测试"""
    if sys.platform.lower() != "win32":
        print("当前不是Windows平台，跳过Windows特定测试")
        return True
    
    return run_command(
        ["python", "-m", "pytest", "-m", "windows", "-v"],
        "Windows特定测试"
    )


def run_fast_tests():
    """运行快速测试（跳过慢速测试）"""
    return run_command(
        ["python", "-m", "pytest", "-m", "not slow", "-v"],
        "快速测试"
    )


def run_parallel_tests():
    """运行并行测试"""
    return run_command(
        ["python", "-m", "pytest", "-n", "auto", "-v"],
        "并行测试"
    )


def run_full_suite():
    """运行完整测试套件"""
    return run_command(
        ["python", "test_all.py"],
        "完整测试套件"
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Sindre库测试运行器")
    parser.add_argument(
        "--basic", 
        action="store_true", 
        help="运行基本测试"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="运行覆盖率测试"
    )
    parser.add_argument(
        "--module", 
        type=str, 
        help="运行特定模块测试 (例如: general, lmdb, report)"
    )
    parser.add_argument(
        "--windows", 
        action="store_true", 
        help="运行Windows特定测试"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="运行快速测试（跳过慢速测试）"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="运行并行测试"
    )
    parser.add_argument(
        "--full", 
        action="store_true", 
        help="运行完整测试套件"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="运行所有测试"
    )
    
    args = parser.parse_args()
    
    # 检查是否在正确的目录中
    if not os.path.exists("pytest.ini"):
        print("错误: 请在sindre/test目录中运行此脚本")
        sys.exit(1)
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    success = True
    
    # 根据参数运行相应测试
    if args.all or not any(vars(args).values()):
        # 运行所有测试
        print("\n运行所有测试...")
        success &= run_basic_tests()
        success &= run_coverage_tests()
        if sys.platform.lower() == "win32":
            success &= run_windows_tests()
    else:
        if args.basic:
            success &= run_basic_tests()
        if args.coverage:
            success &= run_coverage_tests()
        if args.module:
            success &= run_specific_module(args.module)
        if args.windows:
            success &= run_windows_tests()
        if args.fast:
            success &= run_fast_tests()
        if args.parallel:
            success &= run_parallel_tests()
        if args.full:
            success &= run_full_suite()
    
    # 显示结果
    print(f"\n{'='*60}")
    if success:
        print("✓ 所有测试完成")
    else:
        print("✗ 部分测试失败")
        sys.exit(1)
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 