# Report 报告生成模块

> 快速生成美观的HTML测试报告，支持图片、链接、数据可视化等功能

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [核心功能](#核心功能)
- [使用指南](#使用指南)
- [高级功能](#高级功能)
- [模板定制](#模板定制)
- [常见问题](#常见问题)

## ✨ 功能特性

- 📊 **HTML报告**: 生成美观的HTML格式测试报告
- 🖼️ **图片支持**: 支持PIL图片和Base64编码图片
- 🔗 **链接嵌入**: 支持下载链接和外部链接
- 📈 **数据可视化**: 支持图表和统计信息展示
- 🎨 **模板定制**: 可自定义报告模板和样式
- 📱 **响应式设计**: 支持移动端和桌面端显示
- ⚡ **快速生成**: 高效的报告生成和处理

## 🚀 快速开始

### 基本使用

```python
from sindre.report import Report

# 创建报告对象
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

# 生成报告
report.write("./reports/")
```

### 带图片的报告

```python
from sindre.report import Report
from PIL import Image

# 创建报告
report = Report()

# 加载图片
image = Image.open('./result.png')

# 转换为Base64
b64_image = Report.PIL_To_B64(image)

# 添加带图片的测试结果
test_result = {
    "className": "ImageTest",
    "methodName": "test_image_processing",
    "description": "测试图像处理功能",
    "spendTime": "1.2 s",
    "status": "成功",
    "log": [
        "处理完成",
        b64_image,  # 嵌入图片
        "这是文本加图片的混合内容"
    ]
}

report.append_row(test_result)
report.write("./reports/")
```

## 🔧 核心功能

### Report 类

```python
class Report:
    """HTML报告生成器"""
    
    def __init__(self):
        """初始化报告对象"""
        self.data = {
            "testPass": 0,
            "testResult": [],
            "testName": "测试报告",
            "testAll": 0,
            "testFail": 0,
            "beginTime": "2024-01-01 00:00:00",
            "totalTime": "",
            "testSkip": 0,
        }
        self.file_path = os.path.dirname(__file__)
    
    def append_row(self, row_data: dict):
        """
        添加测试结果行
        
        Args:
            row_data: 包含测试信息的字典
                - className: 测试类名
                - methodName: 测试方法名
                - description: 测试描述
                - spendTime: 耗时
                - status: 状态 (成功/失败/跳过)
                - log: 日志列表
        """
    
    @staticmethod
    def PIL_To_B64(image: PIL.Image.Image) -> str:
        """
        将PIL图片转换为Base64字符串
        
        Args:
            image: PIL图片对象
            
        Returns:
            str: Base64编码的图片字符串
        """
    
    def write(self, path: str = "./"):
        """
        生成并保存HTML报告
        
        Args:
            path: 保存路径，默认为当前目录
        """
```

## 📖 使用指南

### 1. 基础报告生成

#### 简单测试报告

```python
from sindre.report import Report

# 创建报告
report = Report()

# 添加成功测试
success_test = {
    "className": "BasicTest",
    "methodName": "test_addition",
    "description": "测试加法运算",
    "spendTime": "0.1 s",
    "status": "成功",
    "log": ["1 + 1 = 2", "测试通过"]
}

# 添加失败测试
failed_test = {
    "className": "BasicTest",
    "methodName": "test_division",
    "description": "测试除法运算",
    "spendTime": "0.05 s",
    "status": "失败",
    "log": ["除零错误", "需要修复"]
}

# 添加跳过测试
skipped_test = {
    "className": "BasicTest",
    "methodName": "test_advanced",
    "description": "高级功能测试",
    "spendTime": "0 s",
    "status": "跳过",
    "log": ["功能未实现"]
}

# 添加到报告
report.append_row(success_test)
report.append_row(failed_test)
report.append_row(skipped_test)

# 生成报告
report.write("./test_reports/")
```

#### 批量测试报告

```python
from sindre.report import Report
import time

# 创建报告
report = Report()

# 模拟批量测试
test_cases = [
    ("test_function_1", "功能1测试", "成功"),
    ("test_function_2", "功能2测试", "成功"),
    ("test_function_3", "功能3测试", "失败"),
    ("test_function_4", "功能4测试", "跳过"),
]

for i, (method_name, description, status) in enumerate(test_cases):
    test_result = {
        "className": f"BatchTest{i//2}",
        "methodName": method_name,
        "description": description,
        "spendTime": f"{0.1 + i*0.05:.2f} s",
        "status": status,
        "log": [f"测试结果: {status}"]
    }
    report.append_row(test_result)

# 生成报告
report.write("./batch_reports/")
```

### 2. 图片和可视化报告

#### 图表报告

```python
from sindre.report import Report
import matplotlib.pyplot as plt
import numpy as np

# 创建报告
report = Report()

# 生成测试图表
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)
ax.set_title('测试图表')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')

# 保存图表
plt.savefig('test_chart.png')
plt.close()

# 转换为Base64
from PIL import Image
chart_image = Image.open('test_chart.png')
b64_chart = Report.PIL_To_B64(chart_image)

# 添加带图表的测试结果
chart_test = {
    "className": "VisualizationTest",
    "methodName": "test_chart_generation",
    "description": "测试图表生成功能",
    "spendTime": "1.5 s",
    "status": "成功",
    "log": [
        "图表生成成功",
        b64_chart,
        "图表数据: 100个点"
    ]
}

report.append_row(chart_test)
report.write("./visual_reports/")
```

#### 多图片报告

```python
from sindre.report import Report
from PIL import Image, ImageDraw

# 创建报告
report = Report()

# 生成多个测试图片
for i in range(3):
    # 创建测试图片
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((50, 100), f'Test Image {i+1}', fill='black')
    
    # 转换为Base64
    b64_img = Report.PIL_To_B64(img)
    
    # 添加测试结果
    test_result = {
        "className": "ImageTest",
        "methodName": f"test_image_{i+1}",
        "description": f"测试图片生成 {i+1}",
        "spendTime": "0.2 s",
        "status": "成功",
        "log": [
            f"图片 {i+1} 生成成功",
            b64_img
        ]
    }
    report.append_row(test_result)

report.write("./image_reports/")
```

### 3. 复杂日志报告

#### 混合内容报告

```python
from sindre.report import Report

# 创建报告
report = Report()

# 复杂测试结果
complex_test = {
    "className": "ComplexTest",
    "methodName": "test_complex_function",
    "description": "测试复杂功能",
    "spendTime": "5.2 s",
    "status": "成功",
    "log": [
        "<br></br>",  # HTML换行
        "开始测试...",
        "步骤1: 初始化完成",
        "步骤2: 数据处理完成",
        "步骤3: 结果验证完成",
        "测试完成！",
        "<a href='https://example.com'>查看详细文档</a>",  # HTML链接
        ""
    ]
}

report.append_row(complex_test)
report.write("./complex_reports/")
```

## 🚀 高级功能

### 1. 自定义报告标题

```python
from sindre.report import Report

# 创建报告并设置标题
report = Report()
report.data["testName"] = "深度学习模型评估报告"

# 添加测试结果...
report.write("./custom_reports/")
```

### 2. 统计信息自动计算

```python
from sindre.report import Report

# 创建报告
report = Report()

# 添加各种测试结果
test_results = [
    {"status": "成功", "spendTime": "1.0 s"},
    {"status": "成功", "spendTime": "2.0 s"},
    {"status": "失败", "spendTime": "0.5 s"},
    {"status": "跳过", "spendTime": "0.0 s"},
]

for i, result in enumerate(test_results):
    test_result = {
        "className": "AutoTest",
        "methodName": f"test_{i+1}",
        "description": f"自动测试 {i+1}",
        "spendTime": result["spendTime"],
        "status": result["status"],
        "log": [f"测试 {result['status']}"]
    }
    report.append_row(test_result)

# 生成报告（会自动计算统计信息）
report.write("./auto_reports/")
# 报告会自动包含：
# - testAll: 总测试数
# - testPass: 成功数
# - testFail: 失败数
# - testSkip: 跳过数
# - totalTime: 总耗时
```

### 3. 异常处理报告

```python
from sindre.report import Report

# 创建报告
report = Report()

# 模拟异常测试
try:
    # 模拟可能出错的代码
    result = 1 / 0
except Exception as e:
    error_test = {
        "className": "ExceptionTest",
        "methodName": "test_division_by_zero",
        "description": "测试除零异常处理",
        "spendTime": "0.01 s",
        "status": "失败",
        "log": [
            f"捕获异常: {type(e).__name__}",
            f"异常信息: {str(e)}",
            "异常处理完成"
        ]
    }
    report.append_row(error_test)

report.write("./exception_reports/")
```

## 🎨 模板定制

### 1. 报告模板结构

Report模块使用内置的HTML模板，包含以下部分：

- **头部信息**: 测试名称、开始时间、总耗时
- **统计信息**: 成功/失败/跳过数量统计
- **测试结果表格**: 详细的测试结果列表
- **样式设计**: 响应式CSS样式

### 2. 自定义样式

虽然Report类使用内置模板，但可以通过修改生成的HTML文件来自定义样式：

```html
<!-- 在生成的HTML文件中添加自定义CSS -->
<style>
.custom-style {
    background-color: #f0f0f0;
    border-radius: 5px;
    padding: 10px;
}
</style>
```

## ❓ 常见问题

### Q1: 如何设置报告标题？

**A**: 通过修改data字典中的testName字段：
```python
report = Report()
report.data["testName"] = "我的自定义测试报告"
```

### Q2: 支持哪些图片格式？

**A**: 支持PIL库支持的所有格式，常用格式包括：
- PNG
- JPEG/JPG
- GIF
- BMP
- TIFF

### Q3: 如何添加HTML内容？

**A**: 在log列表中直接添加HTML标签：
```python
test_result = {
    # ... 其他字段
    "log": [
        "普通文本",
        "<strong>粗体文本</strong>",
        "<a href='https://example.com'>链接</a>",
        "<br></br>",  # 换行
        "<img src='data:image/png;base64,...'>"  # 图片
    ]
}
```

### Q4: 报告文件保存在哪里？

**A**: 默认保存在指定路径下的"测试报告.html"文件中：
```python
report.write("./reports/")  # 保存为 ./reports/测试报告.html
```

### Q5: 如何获取测试统计信息？

**A**: 在调用write()方法后，统计信息会自动计算并包含在报告中：
```python
report.write("./reports/")
# 统计信息在report.data中：
# - testAll: 总测试数
# - testPass: 成功数
# - testFail: 失败数
# - testSkip: 跳过数
# - totalTime: 总耗时
```

### Q6: 支持并发测试吗？

**A**: Report类本身不是线程安全的，如果需要并发使用，建议：
```python
# 每个线程使用独立的Report实例
import threading

def worker(thread_id):
    report = Report()
    report.data["testName"] = f"线程{thread_id}测试报告"
    # 添加测试结果...
    report.write(f"./thread_{thread_id}_reports/")

# 创建多个线程
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```

## 📊 报告示例

生成的HTML报告包含以下特性：

- **响应式设计**: 适配不同屏幕尺寸
- **状态标识**: 不同颜色区分成功/失败/跳过
- **时间统计**: 显示执行时间信息
- **多媒体支持**: 支持图片、链接、代码块
- **交互功能**: 可折叠详情、状态筛选
- **美观样式**: 现代化的UI设计

## 🔗 相关链接

- [HTML模板语法](https://developer.mozilla.org/en-US/docs/Web/HTML)
- [CSS样式指南](https://developer.mozilla.org/en-US/docs/Web/CSS)
- [PIL图像处理](https://pillow.readthedocs.io/)

---

<div align="center">

**如有问题，请查看 [常见问题](#常见问题) 或提交 [Issue](https://github.com/SindreYang/sindre/issues)**

</div>