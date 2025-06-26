# Sindre 库文档

> 一个功能丰富的Python工具库，提供LMDB数据库操作、3D处理、报告生成、Windows工具等功能

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/Docs-MkDocs-blue.svg)](https://sindreyang.github.io/sindre/)

## 📋 目录

- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [安装指南](#安装指南)
- [使用示例](#使用示例)
- [API文档](#api文档)
- [测试指南](#测试指南)
- [贡献指南](#贡献指南)

## 🚀 快速开始

### 安装

```bash
# 从PyPI安装
pip install sindre

# 从源码安装
git clone https://github.com/SindreYang/sindre.git
cd sindre
pip install -e .
```

### 最小示例

```python
import sindre

# LMDB数据库操作
from sindre.lmdb import Reader, Writer
import numpy as np

# 写入数据
writer = Writer('./data', map_size_limit=1024*100)  # 100GB
writer.put_samples({0: {'points': np.random.rand(100, 3)}})
writer.close()

# 读取数据
reader = Reader('./data')
data = reader[0]
print(f"读取到 {len(data['points'])} 个点")
reader.close()
```

## 🎯 核心功能

| 模块 | 功能描述 | 适用场景 |
|------|----------|----------|
| **LMDB** | 高性能数据库操作 | 大规模数据存储、机器学习数据集 |
| **Utils3D** | 3D数据处理工具 | 点云处理、网格操作、3D可视化 |
| **Report** | HTML报告生成 | 测试报告、数据分析报告 |
| **WinTools** | Windows系统工具 | Windows应用开发、系统集成 |
| **Deploy** | 模型部署工具 | 模型优化、推理加速 |
| **General** | 通用工具 | 日志记录、通用功能 |

## 📦 安装指南

### 系统要求

- **Python**: 3.8 或更高版本
- **操作系统**: Windows, Linux, macOS
- **内存**: 建议 4GB 以上

### 依赖安装

```bash
# 基础依赖
pip install numpy lmdb msgpack tqdm

# 可选依赖（根据使用场景）
pip install torch vedo scikit-learn pillow loguru
```

### 开发环境

```bash
# 克隆项目
git clone https://github.com/SindreYang/sindre.git
cd sindre

# 安装开发依赖
pip install -r requirements.txt

# 开发模式安装
pip install -e .

# 运行测试
cd sindre/test
python run_tests.py --all
```

## 💡 使用示例

### 1. LMDB 数据库操作

```python
import sindre.lmdb as lmdb
import numpy as np

# 创建数据库
writer = lmdb.Writer('./dataset', map_size_limit=1024*100)  # 100GB

# 写入数据
for i in range(1000):
    data = {
        'points': np.random.rand(100, 3),
        'labels': np.random.randint(0, 10, 100),
        'metadata': {'id': i, 'source': 'synthetic'}
    }
    writer.put_samples({i: data})

writer.close()

# 读取数据
reader = lmdb.Reader('./dataset')
print(f"数据库包含 {len(reader)} 个样本")

# 批量读取
batch = reader.get_samples(0, 10)
print(f"批量读取 {len(batch)} 个样本")

reader.close()
```

### 2. 3D 数据处理

```python
from sindre.utils3d.mesh import SindreMesh
import numpy as np

# 创建网格
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]
])
faces = np.array([[0, 1, 2], [1, 3, 2]])

mesh = SindreMesh()
mesh.vertices = vertices
mesh.faces = faces

# 网格操作
print(f"网格包含 {len(mesh.vertices)} 个顶点，{len(mesh.faces)} 个面")

# 采样点云
points = mesh.sample(density=1.0)
print(f"采样得到 {len(points)} 个点")

# 可视化
mesh.show()
```

### 3. 报告生成

```python
from sindre.report import Report
from PIL import Image

# 创建报告
report = Report()

# 添加测试结果
test_result = {
    "className": "ModelTest",
    "methodName": "test_accuracy",
    "description": "测试模型准确率",
    "spendTime": "2.5 s",
    "status": "成功",
    "log": ["准确率: 95.2%", "损失: 0.048"]
}

report.append_row(test_result)

# 添加图片
image = Image.open('result.png')
b64_image = Report.PIL_To_B64(image)
test_result["log"].append(b64_image)

# 生成报告
report.write('./reports/')
```

### 4. Windows 工具

```python
from sindre.win_tools import tools, taskbar

# 编译Python文件为PYD
tools.py2pyd(r"C:\project\src", clear_py=False)

# 设置窗口透明度
taskbar.set_windows_alpha(128, "Shell_TrayWnd")

# 制作安装包
tools.exe2nsis(
    work_dir=r"C:\project",
    files_to_compress=[r"C:\project\app.exe", r"C:\project\config.ini"],
    exe_name="MyApp.exe"
)
```

### 5. 模型部署

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

### 6. 日志记录

```python
from sindre.general.logs import CustomLogger

# 创建日志记录器
logger = CustomLogger("my_app").get_logger()

# 记录日志
logger.info("应用启动")
logger.warning("发现警告信息")
logger.error("发生错误")

# 重定向print到日志
logger.redirect_print()
print("这条消息会被记录到日志文件")
```

## 📚 API文档

### 模块概览

- **[LMDB模块](lmdb.md)** - 高性能数据库操作
- **[Utils3D模块](3d.md)** - 3D数据处理工具
- **[Report模块](report.md)** - HTML报告生成
- **[WinTools模块](win_tools.md)** - Windows系统工具
- **[Deploy模块](deploy.md)** - 模型部署工具
- **[General模块](general.md)** - 通用工具

### 核心类

#### LMDB模块
- `Writer` - 数据库写入器
- `Reader` - 数据库读取器
- `ReaderList` - 多数据库读取器
- `ReaderSSD` - SSD优化读取器

#### Utils3D模块
- `SindreMesh` - 3D网格处理类
- `pointcloud_augment` - 点云数据增强

#### Report模块
- `Report` - HTML报告生成器

#### WinTools模块
- `tools` - Windows工具函数集合
- `taskbar` - 任务栏管理函数

#### Deploy模块
- `OnnxInfer` - ONNX推理类
- `TRTInfer` - TensorRT推理类

#### General模块
- `CustomLogger` - 自定义日志记录器

## 🧪 测试指南

### 运行测试

```bash
# 运行所有测试
cd sindre/test
python run_tests.py --all

# 运行特定模块测试
python run_tests.py --module lmdb
python run_tests.py --module utils3d
python run_tests.py --module report
python run_tests.py --module win_tools
python run_tests.py --module deploy
python run_tests.py --module general
```

### 测试覆盖率

```bash
# 安装测试依赖
pip install pytest pytest-cov

# 运行测试并生成覆盖率报告
pytest --cov=sindre --cov-report=html
```

### 性能测试

```bash
# 运行性能基准测试
python benchmark_tests.py
```

## 🤝 贡献指南

### 开发环境设置

```bash
# 1. Fork项目
git clone https://github.com/your-username/sindre.git
cd sindre

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 安装开发工具
pip install black flake8 mypy pre-commit
```

### 代码规范

```bash
# 代码格式化
black sindre/

# 代码检查
flake8 sindre/

# 类型检查
mypy sindre/

# 运行预提交钩子
pre-commit run --all-files
```

### 提交规范

```bash
# 提交信息格式
git commit -m "feat: 添加新功能"
git commit -m "fix: 修复bug"
git commit -m "docs: 更新文档"
git commit -m "test: 添加测试"
git commit -m "refactor: 重构代码"
```

### 拉取请求

1. 创建功能分支
2. 实现功能并添加测试
3. 确保所有测试通过
4. 更新文档
5. 提交拉取请求

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢所有为这个项目做出贡献的开发者和用户。

## 📞 联系方式

- **作者**: Sindre Yang
- **邮箱**: yx@mviai.com
- **GitHub**: [https://github.com/SindreYang](https://github.com/SindreYang)

---

**注意**: 这是一个活跃开发中的项目，API可能会发生变化。请查看最新文档获取最新信息。
