# lmdb集成库

## 1. 简介

- 将lmdb与msgpack结合，方便将numpy类型字典加入lmdb存储；
- 进行二次封装，提供更加高级接口，方便使用。
- 每个数据库，由1个meta数据库（str类型）+1个data数据库（numpy类型）组成



## 2. 使用方法

###### 1. 用于pytorch加载器
```python

# pip install sindre
from sindre.lmdb import  Reader
import torch


class TorchDataset(torch.utils.data.Dataset):
    """Object for interfacing with `torch.utils.data.Dataset`.
    Parameter
    ---------
    dirpath : string
        Path to the directory containing the LMDB.
    """
    def __init__(self, dirpath):
        self.dirpath = dirpath
        self.db = Reader(self.dirpath, lock=False)

    def __len__(self):
        return len(self.db)

    def __getitem__(self, key):
        data = self.db[key]
        for k in data.keys():
            data[k] = torch.from_numpy(data[k])

        return data

    def __repr__(self):
        return str(self.db)

    

```


###### 2. 创建数据

```python
import sindre.lmdb as  lmdb
import numpy as np
X = np.random.random((8, 2, 2, 2, 2))
y = np.arange(2, dtype=np.uint8)

# 创建
db =  lmdb.Writer(dirpath=r'data', map_size_limit=1)
print(db)
db.put_samples({'input1': X, 'target1': y})
db.put_samples({"jaw":np.array("upper"),"name":np.array("数据5aaaaaaaaaaa")})
db.set_meta_str("第一个描述信息", "这是创建")
db.close()

```

###### 3. 追加并扩容
```python
import sindre.lmdb as  lmdb
import numpy as np
db = lmdb.Writer(dirpath=r'data', map_size_limit=50)
print(db)
for i in range(100):
    db.put_samples({'rangeX': X, 'rangeY':X})
db.set_meta_str("第二个描述信息", "追加")
db.close()
```

###### 4.  修改

```python
import sindre.lmdb as  lmdb
import numpy as np
#将索引为2的修改为新的内容
db = lmdb.Writer(dirpath=r'data', map_size_limit=10)
db.change_db_value(2,{'y':y, 'x':y})
db.close()

```

###### 5. 修复windows无法实时变化大小
```python
import sindre.lmdb as  lmdb
import numpy as np
# 将会把预先分配的数据大小恢复实际大小
lmdb.repair_windows_size(dirpath=r'data')
```

###### 6. 读取
```python
import sindre.lmdb as  lmdb
import numpy as np
db = lmdb.Reader(dirpath=r'data')
print(db)
print(db.get_meta_key_info())
print(db.get_data_key_info())
print(db.get_meta_str("第一个描述信息"))
print(db.get_meta_str("第二个描述信息"))
print(db[2].keys())
print(db[1].keys())
print(db[0].keys())
db.close()

```


###### 7. 合并数据库
```python
# 合并数据库
from sindre  import   lmdb
import numpy as np
db_A =  lmdb.Writer(dirpath=r'A', map_size_limit=1)
db_A.put_samples({'inputA': X, 'targetA': y})
db_A.set_meta_str("第一个描述信息", "这是A")
db_A.close()

db_B=  lmdb.Writer(dirpath=r'B', map_size_limit=1)
db_B.put_samples({'inputB': X, 'targetB': y})
db_B.set_meta_str("第二个描述信息", "这是B")
db_B.close()
# 开始合并
lmdb.merge_db(merge_dirpath=r'C', A_dirpath="A", B_dirpath="B",map_size_limit=2)

# 对合并数据库进行读取
with lmdb.Reader(dirpath=r'C') as l:
    print(l.get_meta_key_info())
```



## 3. API文档

::: lmdb.pylmdb

   
