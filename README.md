<div align="center">

# Sindre - 多功能Python工具库

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-LGPL-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-mkdocs-blue.svg)](https://sindreyang.github.io/sindre/)
[![PyPI](https://img.shields.io/badge/PyPI-sindre-red.svg)](https://pypi.org/project/sindre/)

**一个集成了LMDB数据库、3D处理、报告生成、部署工具等多种功能的Python工具库,(2025年8月后版本只维护python3.12)**

<p align="center">
    <br />
    <a href="https://sindreyang.github.io/sindre/"><strong>📖 完整API文档</strong></a>
    <br />
    <br />
    <a href="https://github.com/SindreYang/sindre/releases">📦 下载Releases</a>
    ·
    <a href="https://github.com/SindreYang/sindre/issues">🐛 报告Bug</a>
    ·
    <a href="https://github.com/SindreYang/sindre/issues">💡 提出新特性</a>
  </p>
</div>

## 📋 目录

- [功能特性](#-功能特性)
- [快速开始](#-快速开始)
- [安装指南](#-安装指南)
- [核心模块](#-核心模块)
- [使用示例](#-使用示例)
- [API文档](#-api文档)
- [开发指南](#-开发指南)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

## ✨ 功能特性

- **🔍 LMDB数据库管理**: 高性能的键值存储，支持多进程操作
- **🎯 3D数据处理**: 网格处理、点云操作、3D可视化
- **📊 报告生成**: 自动生成HTML测试报告
- **🚀 部署工具**: ONNX、TensorRT、共享内存等部署方案
- **🛠️ Windows工具**: 系统级工具函数和安装包制作
- **📝 日志管理**: 多进程日志记录和异常捕获
- **🧪 测试框架**: 完整的pytest测试套件

## 🚀 快速开始

### 基本安装

```bash
# 从PyPI安装
pip install sindre

# 或者从源码安装
git clone https://github.com/SindreYang/sindre.git
cd sindre
pip install -e .
```

### 快速示例

```python
# LMDB数据库操作
import sindre.lmdb as lmdb
import numpy as np

# 写入数据
data = {'input': np.random.rand(10, 10), 'target': [0, 2, 10]}
with lmdb.Writer('./test_lmdb') as db:
    db.put_samples(data)

# 读取数据
with lmdb.Reader('./test_lmdb') as reader:
    sample = reader[0]
    print(sample.keys())

# 3D网格处理
from sindre.utils3d.mesh import SindreMesh
import numpy as np

vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2]])
mesh = SindreMesh(vertices=vertices, faces=faces)
mesh.show()  # 3D可视化
```

## 📦 安装指南

### 系统要求

- **Python**: >= 3.8
- **操作系统**: Windows, Linux, macOS
- **内存**: 建议 >= 4GB

### 详细安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/SindreYang/sindre.git
cd sindre
```

#### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 可选：安装开发依赖
pip install pytest mkdocs mkdocs-material
```

#### 3. 安装模式选择

```bash
# 开发模式（推荐用于开发）
pip install -e .

# 生产模式
python setup.py install

# 构建wheel包
python setup.py bdist_wheel
```

### 依赖说明

主要依赖包括：
- `numpy`: 数值计算
- `torch`: 深度学习框架
- `vedo`: 3D可视化
- `lmdb`: 数据库存储
- `onnxruntime`: 模型推理
- `pytest`: 测试框架

## 🧩 核心模块

### 📊 LMDB模块 (`sindre.lmdb`)

高性能键值存储数据库，支持多进程操作。

**主要功能**:
- 高效的数据读写
- 多进程安全
- 内存映射优化
- 元数据管理

### 🎯 3D处理模块 (`sindre.utils3d`)

完整的3D数据处理解决方案。

**主要功能**:
- 网格处理与转换
- 点云操作
- 3D可视化
- 深度学习网络

### 📋 报告模块 (`sindre.report`)

自动生成测试报告和数据分析报告。

**主要功能**:
- HTML报告生成
- 测试结果统计
- 自定义模板
- 多格式导出

### 🚀 部署模块 (`sindre.deploy`)

模型部署和优化工具。

**主要功能**:
- ONNX模型转换
- TensorRT优化
- 共享内存通信
- 多平台部署

### 🛠️ Windows工具 (`sindre.win_tools`)

Windows系统专用工具。

**主要功能**:
- Python文件编译
- 安装包制作
- 系统工具集成

### 📝 通用工具 (`sindre.general`)

通用工具函数和日志管理。

**主要功能**:
- 多进程日志
- 异常捕获
- 工具函数集合

## 💡 使用示例

### LMDB数据库操作

```python
import sindre.lmdb as lmdb
import numpy as np

# 创建数据库并写入数据
data = {
    'input': np.random.rand(100, 64),
    'target': np.random.randint(0, 10, 100),
    'metadata': {'source': 'synthetic', 'version': '1.0'}
}

# 写入数据
with lmdb.Writer('./dataset.lmdb', map_size_limit=1024) as db:
    db.set_meta_str("description", "测试数据集")
    db.put_samples(data)

# 读取数据
with lmdb.Reader('./dataset.lmdb') as reader:
    print(f"数据库大小: {len(reader)}")
    print(f"元数据: {reader.get_meta_key_info()}")
    
    # 获取第一个样本
    sample = reader[0]
    print(f"样本键: {sample.keys()}")
    print(f"输入形状: {sample['input'].shape}")
```

### 3D网格处理

```python
from sindre.utils3d.mesh import SindreMesh
import numpy as np

# 创建简单网格
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
])
faces = np.array([
    [0, 1, 2], [1, 3, 2],  # 底面
    [4, 6, 5], [5, 6, 7],  # 顶面
    [0, 4, 1], [1, 4, 5],  # 侧面
    [2, 3, 6], [3, 7, 6],  # 侧面
    [0, 2, 4], [2, 6, 4],  # 侧面
    [1, 5, 3], [3, 5, 7]   # 侧面
])

# 创建网格对象
mesh = SindreMesh(vertices=vertices, faces=faces)

# 显示网格信息
print(mesh)
print(f"顶点数: {len(mesh.vertices)}")
print(f"面数: {len(mesh.faces)}")

# 3D可视化
mesh.show()

# 采样点云
points = mesh.sample(density=0.1)
print(f"采样点数: {len(points)}")

# 格式转换
trimesh_mesh = mesh.to_trimesh()
open3d_mesh = mesh.to_open3d()
```

### 报告生成

```python
from sindre.report.report import Report

# 创建报告对象
report = Report()

# 添加测试结果
test_results = [
    {
        "className": "TestModel",
        "methodName": "test_accuracy",
        "description": "测试模型准确率",
        "spendTime": 2.5,
        "status": "成功",
        "log": ["准确率: 95.2%", "损失: 0.048"]
    },
    {
        "className": "TestModel", 
        "methodName": "test_inference",
        "description": "测试推理速度",
        "spendTime": 1.8,
        "status": "成功",
        "log": ["平均推理时间: 15ms"]
    }
]

for result in test_results:
    report.append_row(result)

# 生成报告
report.write(output_path="./test_report.html")
```

### 部署工具

```python
from sindre.deploy import onnxruntime_deploy, TenserRT_deploy

# ONNX模型部署
onnx_deployer = onnxruntime_deploy.OnnxDeployer()
onnx_deployer.load_model("model.onnx")
result = onnx_deployer.inference(input_data)

# TensorRT部署
trt_deployer = TenserRT_deploy.TRTDeployer()
trt_deployer.build_engine("model.onnx", "model.engine")
result = trt_deployer.inference(input_data)
```

### Windows工具

```python
from sindre.win_tools import tools

# 编译Python文件为pyd
tools.py2pyd(
    source_dir=r"C:\project\src",
    clear_py=False,  # 保留原py文件
    exclude_patterns=["test_*.py"]  # 排除测试文件
)

# 创建安装包
tools.create_installer(
    source_dir=r"C:\project\dist",
    output_path=r"C:\project\installer.exe",
    app_name="MyApp",
    app_version="1.0.0"
)
```

## 📚 API文档

完整的API文档请访问：[https://sindreyang.github.io/sindre/](https://sindreyang.github.io/sindre/)

### 文档生成

```bash
# 安装依赖
pip install -r requirements.txt

# 启动文档服务器
mkdocs serve

# 构建文档
mkdocs build
```

## 🛠️ 开发指南

### 项目结构

```
sindre/
├── .github/              # GitHub Actions工作流
├── docs/                 # 文档目录
│   ├── index.md         # 主页
│   ├── lmdb.md          # LMDB模块文档
│   ├── report.md        # 报告模块文档
│   ├── 3d.md            # 3D处理模块文档
│   ├── deploy.md        # 部署模块文档
│   ├── win_tools.md     # Windows工具文档
│   └── general.md       # 通用工具文档
├── sindre/              # 核心代码
│   ├── lmdb/            # LMDB数据库模块
│   ├── utils3d/         # 3D处理模块
│   ├── report/          # 报告生成模块
│   ├── deploy/          # 部署工具模块
│   ├── win_tools/       # Windows工具模块
│   └── general/         # 通用工具模块
├── test/                # 测试代码
├── requirements.txt     # 依赖列表
├── setup.py            # 安装配置
└── README.md           # 项目说明
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest test/test_lmdb.py
pytest test/test_utils3d.py

# 生成测试报告
pytest --html=report.html --self-contained-html
```

### 代码规范

- 使用Google风格的docstring
- 遵循PEP 8代码规范
- 所有函数和类都需要类型注解
- 测试覆盖率要求 > 80%

## 🤝 贡献指南

我们欢迎所有形式的贡献！

### 贡献方式

1. **报告Bug**: 在GitHub Issues中报告问题
2. **提出新特性**: 在Issues中提出建议
3. **提交代码**: Fork项目并提交Pull Request
4. **改进文档**: 帮助完善文档和示例

### 开发流程

1. Fork项目到你的GitHub账户
2. 克隆你的fork到本地
3. 创建功能分支: `git checkout -b feature/amazing-feature`
4. 提交更改: `git commit -m 'Add amazing feature'`
5. 推送分支: `git push origin feature/amazing-feature`
6. 创建Pull Request

### 代码审查

所有提交的代码都会经过审查，确保：
- 代码质量符合标准
- 测试覆盖充分
- 文档更新完整
- 向后兼容性

## 📄 许可证

本项目采用LGPL许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户！

## 📞 联系我们

- **GitHub Issues**: [https://github.com/SindreYang/sindre/issues](https://github.com/SindreYang/sindre/issues)
- **文档**: [https://sindreyang.github.io/sindre/](https://sindreyang.github.io/sindre/)
- **邮箱**: 通过GitHub Issues联系

---

<div align="center">
如果这个项目对你有帮助，请给它一个 ⭐️
</div>
