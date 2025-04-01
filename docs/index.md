

# Sindre-自用库
</p>
  <a href="https://github.com/SindreYang/sindre/releases">下载Releases</a>
  ·
  <a href="https://github.com/SindreYang/sindre/issues">报告Bug</a>
  ·
  <a href="https://github.com/SindreYang/sindre/issues">提出新特性</a>
</p>

## 快速安装

```bash   

pip install  sindre

 ```   


 

## 推荐配置


1. python >= 3.8


## 安装步骤

### 安装依赖

```bash

# 克隆项目   
git clone https://github.com/SindreYang/sindre.git
# 安装依赖   
pip install -r requirements.txt


```   

### 构建包


```bash
python setup.py bdist_wheel
```

### 开发模式


```bash
pip install -e .
```



## 生成文档


  
```bash
#设置文档范式 pycharm--->setting--->Tools--->python integrated Tools-->Google

mkdocs serve  -o  -c 

```

## 构建文档

```bash

mkdocs  build

```


## 目录结构

```

├─.github                   # github的actions工作流脚本
├── docs                    # mkdocs文档目录
├── mkdocs.yml              # mkdocs 配置
├── README.md               # 项目文档
├── requirements.txt        # 项目依赖
├── setup.py                # 发布脚本
└── sindre                  # sindre库项目代码

```


## 各模块使用实例<a href="https://sindreyang.github.io/sindre/"><strong><< API文档链接 >></strong></a>

### LMDB 模块
LMDB 模块提供了读写 LMDB 数据库的功能，支持多进程操作。

```python

import sindre.lmdb as lmdb
import numpy as np

# 写入数据
data = {'input': np.random.rand(10, 10), 'target': [0, 2, 10]}
db = lmdb.Writer(dirpath='./test_lmdb', map_size_limit=100)
db.set_meta_str("描述信息", "测试数据")
gb_required = db.check_sample_size(data)
if gb_required < 1.0:  # 假设内存限制为 1GB
    db.put_samples(data)
db.close()
# 修复windows文件大小问题
lmdb.repair_windows_size(dirpath='./test_lmdb')

# 读取数据
reader = lmdb.Reader(dirpath='./test_lmdb')
print(reader)
print(reader.get_meta_key_info())
print(reader.get_data_key_info())
print(reader[0].keys())
reader.close()

```


### 三维网格处理模块 (utils3d)

三维网格处理模块提供了网格的转换、采样、质量检测等功能。

``` python

import numpy as np
from sindre.utils3d.mesh import SindreMesh

# 创建网格
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2]])
mesh = SindreMesh(vertices=vertices, faces=faces)  
mesh = SindreMesh(load_all("test.ply")) # 支持任意类型导入
# 转换为 任意格式，如meshlab，vedo，pytorch3d,dict,open3d等等
trimesh_mesh = mesh.to_trimesh()

# 显示网格信息
print(mesh)

# 渲染网格
mesh.show()

# 带标签渲染
mesh.show(labels=labels)

# 联合vedo一起渲染
mesh.show([vedo.Points(vertices)])

# 采样点云
points = mesh.sample(density=1)
print(points.shape)

```

### 报告生成模块 (report)

报告生成模块可以将测试结果转换为 HTML 报告。

```python

from sindre.report.report import Report

# 创建报告对象
report = Report()

# 添加测试结果
row = {
    "className": "TestClass",
    "methodName": "test_method",
    "description": "测试方法描述",
    "spendTime": 1.5,
    "status": "成功",
    "log": []
}
report.append_row(row)

# 写入报告
report.write()

```

### Windows 工具模块 (win_tools)

Windows 工具模块提供了一些 Windows 系统相关的工具函数。

```python
from sindre.win_tools import tools

# 将目录下所有 py 文件编译为 pyd
tools.py2pyd(r"C:\Users\sindre\Downloads\55555", clear_py=False)

```



## 引文

```bash
{
  @title={sindre},
  author={sindreyang},
  year={2024}
}

```   