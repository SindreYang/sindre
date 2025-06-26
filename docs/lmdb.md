# LMDB 数据库模块

> 高性能的LMDB数据库操作模块，支持大规模数据存储和高效读取

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [核心类](#核心类)
- [使用指南](#使用指南)
- [高级功能](#高级功能)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

## ✨ 功能特性

- 🚀 **高性能**: 基于LMDB的高性能数据库操作
- 📦 **数据序列化**: 自动处理numpy数组和复杂数据结构
- 🔄 **多进程支持**: 支持多进程并发读写
- 💾 **内存优化**: 智能内存管理和大小控制
- 🔧 **工具丰富**: 提供数据库合并、分割、修复等工具
- 🛡️ **数据安全**: 事务性操作，确保数据一致性

## 🚀 快速开始

### 基本使用

```python
import sindre.lmdb as lmdb
import numpy as np

# 创建数据库,支持目录，也支持文件
writer = lmdb.Writer('./data.db', map_size_limit=1024*100)  # map_size_limit单位为MB 
#writer = lmdb.Writer('./data', map_size_limit=1024*100)  # 会创建data目录

# 写入数据
data = {
    'points': np.random.rand(100, 3),
    'labels': np.random.randint(0, 10, 100),
    'version': '1.0'，
}
writer.put_samples(data)
writer.close()

# 读取数据
reader = lmdb.Reader('./data.db')
sample = reader[0]
print(f"读取到 {len(sample['points'])} 个点")
reader.close()
```

### PyTorch 数据集集成

```python
import torch
from sindre.lmdb import Reader

class LMDBDataset(torch.utils.data.Dataset):
    def __init__(self, db_path):
        self.db = Reader(db_path, multiprocessing=False)
    
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, idx):
        data = self.db[idx]
        # 转换为torch张量
        return {k: torch.from_numpy(v) for k, v in data.items()}

# 使用
dataset = LMDBDataset('./data')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

## 🔧 核心类

### Writer - 数据库写入器

```python
class Writer:
    """LMDB数据库写入器"""
    
    def __init__(self, dirpath: str, map_size_limit: int, multiprocessing: bool = False):
        """
        初始化写入器
        
        Args:
            dirpath: 数据库目录路径
            map_size_limit: 数据库大小限制（MB）
            multiprocessing: 是否启用多进程支持
        """
    
    def put_samples(self, samples: dict):
        """批量写入样本数据"""
    
    def change_value(self, num_id: int, samples: dict):
        """修改指定ID的数据"""
    
    def change_db_value(self, key: int, value: dict, safe_model: bool = True):
        """安全修改数据库值，带确认提示"""
    
    def check_sample_size(self, samples: dict):
        """检查样本大小（GB）"""
    
    def close(self):
        """关闭数据库连接"""
```

### Reader - 数据库读取器

```python
class Reader:
    """LMDB数据库读取器"""
    
    def __init__(self, dirpath: str, multiprocessing: bool = False):
        """
        初始化读取器
        
        Args:
            dirpath: 数据库目录路径
            multiprocessing: 是否启用多进程支持
        """
    
    def __getitem__(self, idx: int):
        """通过索引获取数据"""
    
    def get_sample(self, idx: int):
        """获取单个样本"""
    
    def get_samples(self, start_idx: int, size: int):
        """批量获取样本"""
    
    def get_data_keys(self, i: int = 0):
        """获取第i个样本的所有键"""
    
    def get_data_value(self, i: int, key: str):
        """获取第i个样本的指定键值"""
    
    def get_data_specification(self, i: int):
        """获取第i个样本的数据规范"""
    
    def get_meta_str(self, key):
        """获取元数据字符串"""
    
    def __len__(self):
        """获取数据库大小"""
```

### ReaderList - 多数据库读取器

```python
class ReaderList:
    """多个LMDB数据库的统一读取器"""
    
    def __init__(self, db_path_list: list, multiprocessing: bool = True):
        """
        初始化多数据库读取器
        
        Args:
            db_path_list: 数据库路径列表
            multiprocessing: 是否启用多进程支持
        """
```

### ReaderSSD - SSD优化读取器

```python
class ReaderSSD:
    """针对SSD优化的读取器"""
    
    def __init__(self, db_path: str, multiprocessing: bool = False):
        """
        初始化SSD读取器
        
        Args:
            db_path: 数据库路径
            multiprocessing: 是否启用多进程支持
        """
    
    def get_batch(self, indices: list):
        """批量获取数据"""
```

## 📖 使用指南

### 1. 数据写入

#### 基本写入

```python
import sindre.lmdb as lmdb
import numpy as np

# 创建写入器
writer = lmdb.Writer('./dataset', map_size_limit=1024*100)  # 100GB

# 写入单个样本
data = {
    'points': np.random.rand(1000, 3),
    'labels': np.random.randint(0, 10, 1000),
    'features': np.random.rand(1000, 128)
}
writer.put_samples({0: data})

# 批量写入
for i in range(1000):
    data = {
        'points': np.random.rand(100, 3),
        'labels': np.random.randint(0, 10, 100),
        'id': i
    }
    writer.put_samples({i: data})

writer.close()
```

#### 设置元数据

```python
# 设置数据库元数据
writer.set_meta_str("description", "点云数据集")
writer.set_meta_str("version", "1.0")
writer.set_meta_str("created_by", "sindre")
```

#### 数据修改

```python
# 修改现有数据
new_data = {
    'points': np.random.rand(200, 3),
    'labels': np.random.randint(0, 10, 200),
    'updated': True
}
writer.change_value(0, new_data)

# 安全修改（带确认提示）
writer.change_db_value(0, new_data, safe_model=True)
```

#### 内存大小检查

```python
# 检查数据大小
data = {
    'points': np.random.rand(10000, 3),
    'labels': np.random.randint(0, 10, 10000)
}
gb_required = writer.check_sample_size(data)
print(f"数据大小: {gb_required:.2f} GB")
```

### 2. 数据读取

#### 基本读取

```python
# 创建读取器
reader = lmdb.Reader('./dataset')

# 获取数据库大小
print(f"数据库包含 {len(reader)} 个样本")

# 读取单个样本
sample = reader[0]
print(f"样本键: {list(sample.keys())}")

# 读取指定样本
sample = reader.get_sample(0)
print(f"点云数量: {len(sample['points'])}")
```

#### 批量读取

```python
# 批量读取
samples = reader.get_samples(0, 10)
print(f"读取了 {len(samples)} 个样本")

# 使用ReaderList读取多个数据库
reader_list = lmdb.ReaderList(['./db1', './db2', './db3'])
print(f"总样本数: {len(reader_list)}")
```

#### 元数据查询

```python
# 获取元数据
description = reader.get_meta_str("description")
version = reader.get_meta_str("version")
print(f"描述: {description}, 版本: {version}")

# 获取数据键信息
data_keys = reader.get_data_keys(0)
print(f"数据键: {data_keys}")

# 获取数据规范
spec = reader.get_data_specification(0)
for key, info in spec.items():
    print(f"{key}: shape={info['shape']}, dtype={info['dtype']}")
```

### 3. 多进程支持

```python
# 启用多进程写入
writer = lmdb.Writer('./dataset', map_size_limit=1024*100, multiprocessing=True)

# 启用多进程读取
reader = lmdb.Reader('./dataset', multiprocessing=True)
```

## 🔧 高级功能

### 数据库工具函数

```python
import sindre.lmdb as lmdb

# 合并数据库
lmdb.MergeLmdb(
    target_dir='./merged_db',
    source_dirs=['./db1', './db2', './db3'],
    map_size_limit=1024*100,
    multiprocessing=True
)

# 分割数据库
lmdb.SplitLmdb(
    source_dir='./large_db',
    target_dirs=['./part1', './part2', './part3'],
    map_size_limit=1024*50,
    multiprocessing=True
)

# 修复Windows大小问题
lmdb.fix_lmdb_windows_size('./database')

# 并行写入
def process_function(file_path):
    # 处理单个文件的函数
    return {'processed_data': np.random.rand(100, 3)}

lmdb.parallel_write(
    output_dir='./processed_db',
    file_list=['file1.txt', 'file2.txt', 'file3.txt'],
    process=process_function,
    map_size_limit=1024*100,
    num_processes=4,
    multiprocessing=True
)
```

### SSD优化读取

```python
# 使用SSD优化读取器
reader_ssd = lmdb.ReaderSSD('./dataset', multiprocessing=False)

# 批量读取
indices = [0, 1, 2, 3, 4]
batch_data = reader_ssd.get_batch(indices)
print(f"批量读取了 {len(batch_data)} 个样本")

# 多数据库SSD读取
reader_ssd_list = lmdb.ReaderSSDList(['./db1', './db2'], multiprocessing=False)
```

## ⚡ 性能优化

### 1. 内存管理

```python
# 合理设置map_size_limit
# 建议设置为预期数据大小的1.5-2倍
expected_size_gb = 50
map_size_limit_mb = int(expected_size_gb * 1.5 * 1024)
writer = lmdb.Writer('./dataset', map_size_limit=map_size_limit_mb)
```

### 2. 多进程优化

```python
# 写入时使用多进程
writer = lmdb.Writer('./dataset', map_size_limit=1024*100, multiprocessing=True)

# 读取时根据数据大小决定是否使用多进程
if len(reader) > 10000:
    reader = lmdb.Reader('./dataset', multiprocessing=True)
else:
    reader = lmdb.Reader('./dataset', multiprocessing=False)
```

### 3. 批量操作

```python
# 批量写入而不是逐个写入
batch_data = {}
for i in range(1000):
    batch_data[i] = {
        'points': np.random.rand(100, 3),
        'labels': np.random.randint(0, 10, 100)
    }
writer.put_samples(batch_data)
```

## ❓ 常见问题

### Q1: map_size_limit 设置多大合适？

**A**: 建议设置为预期数据大小的1.5-2倍。例如，如果数据大约50GB，可以设置为：
```python
map_size_limit = int(50 * 1.5 * 1024)  # 75GB in MB
```

### Q2: 多进程模式什么时候使用？

**A**: 
- **写入时**: 数据量大（>1GB）时建议使用
- **读取时**: 数据库样本数多（>10000）时建议使用

### Q3: 如何处理数据库损坏？

**A**: 使用修复工具：
```python
lmdb.fix_lmdb_windows_size('./database')
```

### Q4: 如何检查数据库状态？

**A**: 
```python
writer.check_db_stats()  # 检查数据库统计信息
```

### Q5: 支持哪些数据类型？

**A**: 主要支持numpy数组，其他类型会自动转换：
```python
# 支持的数据
data = {
    'points': np.random.rand(100, 3),      # numpy数组
    'labels': np.random.randint(0, 10, 100), # numpy数组
    'metadata': 'test'                      # 字符串（会被序列化）
}
```

### Q6: 如何高效地修改现有数据？

**A**: 使用安全修改模式：
```python
# 带确认提示的安全修改
writer.change_db_value(0, new_data, safe_model=True)

# 直接修改（无确认）
writer.change_value(0, new_data)
```

## 📊 性能基准

| 操作 | 数据大小 | 时间 | 内存使用 |
|------|----------|------|----------|
| 写入 | 1GB | ~30s | ~2GB |
| 读取 | 1GB | ~5s | ~1GB |
| 批量读取 | 1GB | ~2s | ~1.5GB |
| 随机访问 | 1GB | ~10s | ~1GB |

## 🔗 相关链接

- [LMDB官方文档](https://lmdb.readthedocs.io/)
- [PyTorch数据集教程](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [NumPy文档](https://numpy.org/doc/)

---

<div align="center">

**如有问题，请查看 [常见问题](#常见问题) 或提交 [Issue](https://github.com/SindreYang/sindre/issues)**

</div>

   
