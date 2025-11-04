# -*- coding: UTF-8 -*-
import shutil
import os
import numpy as np
import time
import traceback
import sys
from tqdm import tqdm
import pickle
#from sindre.lmdb.tools import *
import multiprocessing as mp
try:
    import lmdb
    import msgpack
except ImportError:
    raise ImportError(
        "Could not import the LMDB library `lmdb` or  `msgpack`. Please refer "
        "to https://github.com/dw/py-lmdb/  or https://github.com/msgpack/msgpack-python or https://github.com/python-lz4/python-lz4 for installation "
        "instructions."
    )

__all__ = ["get_data_value","get_data_size" , "Reader","ReaderList","ReaderSSDList","ReaderSSD", "Writer", "split_lmdb", "merge_lmdb","get_data_size", "fix_lmdb_windows_size","parallel_write"]

class Base:
    """
    公共工具类
    """
    # 数据库标识
    NB_DBS = 2
    DATA_DB = b"data_db"
    META_DB = b"meta_db"
    # 内置常量
    INTERNAL_KEYS=[b"__physical_keys__",
                   b"__read_keys__",
                   b"__deleted_keys__",
                   b"__db_size__"
                   b"nb_samples"]
    # 支持的序列化类型
    TYPES = {
        "none": b"none",
        "dict": b"dict",
        "ndarray": b"ndarray",
        "object": b"object",
        "unknown": b"unknown",

    }

    @staticmethod
    def decode_str(data):
        return data.decode(encoding="utf-8", errors="strict")
    @staticmethod
    def encode_str(string):
        return str(string).encode(encoding="utf-8", errors="strict")


    @staticmethod
    def encode_data(data):
        if data is None:
            return {b"type": Base.TYPES["none"],
                    b"data": None}
        elif isinstance(data, dict):
            return {b"type": Base.TYPES["dict"],
                    b"data": {k: Base.encode_data(v) for k, v in data.items()}}
        # 其他数据,先转换成numpy类型
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            if data.dtype == object:
                data = data.item()
                return {b"type": Base.TYPES["object"],
                        b"data": pickle.dumps(data)
                        }
            else:
                if np.issubdtype(data.dtype, np.float64):
                    data = data.astype(np.float32)
                elif np.issubdtype(data.dtype, np.int64):
                    data = data.astype(np.int32)
                return {
                    b"type": Base.TYPES["ndarray"],
                    b"dtype": data.dtype.str,
                    b"shape": data.shape,
                    b"data": data.tobytes()
                }
        print(f"不支持类型{type(data)}")
        return {
            b"type": Base.TYPES["unknown"],
            b"data": pickle.dumps(data)
        }


    @staticmethod
    def decode_data(encoded_data):
        try:
            data_type = encoded_data[b"type"]
            if data_type == Base.TYPES["none"]:
                return None
            elif data_type == Base.TYPES["ndarray"]:
                dtype = np.dtype(encoded_data[b"dtype"])
                shape = encoded_data[b"shape"]
                data_bytes = encoded_data[b"data"]
                return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
            elif data_type == Base.TYPES["object"]:
                pickled_data = encoded_data[b"data"]
                return pickle.loads(pickled_data)
            elif data_type == Base.TYPES["dict"]:
                encoded_dict = encoded_data[b"data"]
                return {k: Base.decode_data(v) for k, v in encoded_dict.items()}

            # 兼容老数据库
            elif data_type == 2:
                dtype = np.dtype(encoded_data[b"dtype"])
                shape = encoded_data[b"shape"]
                data_bytes = encoded_data[b"data"]
                return np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
            # 兼容老数据库
            elif data_type == 1:
                return encoded_data[b"data"]

            else:
                return encoded_data
        except (KeyError, ValueError, TypeError, pickle.UnpicklingError) as e:
            print(f"数据解码失败: {e}")
            return encoded_data  # 解码失败时返回原始数据，避免崩溃





class Reader(Base):
    """
    用于读取包含张量(`numpy.ndarray`)数据集的对象。
    这些张量是通过使用MessagePack从Lightning Memory-Mapped Database (LMDB)中读取的。

    支持功能：
    - 读取使用新键管理系统存储的数据
    - 兼容旧版本数据库格式
    - 支持多进程读取
    - 支持任意类型元数据读取
    - 支持读取已删除的样本

    Note:
        with Reader(dirpath='dataset.lmdb') as reader:
            # 基本读取操作
            sample = reader[5]                    # 读取第5个样本
            sample = reader.get_sample(5)         # 读取第5个样本
            samples = reader.get_samples([1,3,5]) # 读取多个样本

            # 获取信息
            mapping = reader.get_mapping()        # 获取键映射关系
            data_keys = reader.get_data_keys(0)   # 获取数据键名
            meta_keys = reader.get_meta_keys()    # 获取元数据键名

            # 特殊功能
            deleted_sample = reader.get_delete_sample(10)  # 读取已删除的样本
            meta_value = reader.get_meta('key')   # 读取元数据
    """

    def __init__(self, dirpath: str,multiprocessing:bool=False):
        """
        初始化

        Args:
            dirpath : 包含LMDB的目录路径。
            multiprocessing : 是否开启多进程读取。

        """

        self.dirpath = dirpath
        self.multiprocessing=multiprocessing


        # 键管理系统
        self.physical_keys = []      # 所有物理存在的键
        self.read_keys = []          # 当前有效的读取键
        self.deleted_keys = set()    # 已删除的键
        self.nb_samples = 0          # 数据库大小

        # 以只读模式打开LMDB环境
        subdir_bool =False if  bool(os.path.splitext(dirpath)[1])  else True
        if multiprocessing:
            self._lmdb_env = lmdb.open(dirpath,
                    readonly=True, 
                    meminit=False,
                    max_dbs=Base.NB_DBS,
                    max_spare_txns=32,
                    subdir=subdir_bool, 
                    lock=False)
        else:
            self._lmdb_env = lmdb.open(dirpath,
                                       readonly=True,
                                       max_dbs=Base.NB_DBS,
                                       subdir=subdir_bool, 
                                       lock=True)

        # 打开与环境关联的默认数据库
        self.data_db = self._lmdb_env.open_db(Base.DATA_DB)
        self.meta_db = self._lmdb_env.open_db(Base.META_DB)

        # 加载键管理系统
        self._load_keys()

    def _load_keys(self):
        """加载键管理信息，兼容旧版本数据库"""
        with self._lmdb_env.begin(db=self.meta_db) as txn:
            # 尝试加载新版本的键管理信息
            physical_data = txn.get(b"__physical_keys__")
            read_data = txn.get(b"__read_keys__")
            deleted_data = txn.get(b"__deleted_keys__")
            if physical_data and read_data :
                # 新版本数据库：使用键管理系统
                self.physical_keys = msgpack.unpackb(physical_data)
                self.read_keys = msgpack.unpackb(read_data)
                if deleted_data:
                    self.deleted_keys = set(msgpack.unpackb(deleted_data))
                else:
                    self.deleted_keys = set()

                # nb_samples 与 read_keys 保持一致
                self.nb_samples = len(self.read_keys)
            else:
                # 旧版本数据库：从现有数据重建键管理系统
                print("\033[93m检测到旧版本数据库，使用兼容模式...\033[0m")
                # 从meta_db获取样本数
                nb_samples_data = txn.get(b"nb_samples")
                if nb_samples_data:
                    try:
                        self.nb_samples = int(nb_samples_data.decode(encoding="utf-8"))
                    except:
                        # 如果不是字符串，尝试用msgpack解码
                        self.nb_samples = msgpack.unpackb(nb_samples_data)
                else:
                    # 如果没有样本数信息，从数据中统计
                    with self._lmdb_env.begin(db=self.data_db) as data_txn:
                        cursor = data_txn.cursor()
                        self.nb_samples = sum(1 for _ in cursor)

                # 重建键列表
                self.physical_keys = list(range(self.nb_samples))
                self.read_keys = list(range(self.nb_samples))
                self.deleted_keys = set()
                self.compress_state=False

    def get_meta(self, key) :
        """
       从元数据库读取任意类型的数据

       Args:
           key: 键名

       Returns:
           存储的数据，如果不存在则返回None
       """
        if isinstance(key, str):
            _key = Base.encode_str(key)
        else:
            _key = key

        with self._lmdb_env.begin(db=self.meta_db) as txn:
            data = txn.get(_key)
            if data is None:
                return None
            # 特殊处理键管理信息
            if _key in Base.INTERNAL_KEYS:
                return msgpack.unpackb(data)
            try:
                return msgpack.unpackb(data)
            except:
                return data
    def get_meta_keys(self) -> set:
        """

        Returns:
            获取元数据库所有键

        """
        key_set = set()
        # 创建一个读事务和游标
        with self._lmdb_env.begin(db=self.meta_db) as txn:
            cursor = txn.cursor()
            # 遍历游标并获取键值对
            for key, value in cursor:
                # 特殊处理键管理信息
                if key in  Base.INTERNAL_KEYS:
                    continue
                key_set.add(Base.decode_str(key))
        return key_set

    def get_dict_keys(self,nested_dict, parent_key="", sep="."):
        """
        提取嵌套字典中所有层级键，用分隔符连接后返回列表

        :param nested_dict: 输入的嵌套字典
        :param parent_key: 父级键（递归时使用，外部调用无需传参）
        :param sep: 键的分隔符，默认 "."
        :return: 扁平键列表（如 ['mesh.v', 'mesh.f']）
        """
        keys = []
        for key, value in nested_dict.items():
            # 拼接当前键与父级键（若有）
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            # 如果值是字典，继续递归提取子键；否则当前键为最终键
            if isinstance(value, dict):
                # 递归获取子键并合并到列表
                keys.extend(self.get_dict_keys(value, new_key, sep=sep))
            else:
                keys.append(new_key)
        return keys


    def get_data_size(self,i: int) -> float:
        """
        计算LMDB中单个样本的存储大小（MB）
        :param i: 索引
        :return: 存储大小（MB）
        """
        # 获取对应的物理键
        physical_key = self.read_keys[i]
        # 将物理键转换为带有尾随零的字符串
        key = Base.encode_str("{:010}".format(physical_key))
        with self._lmdb_env.begin(db=self.data_db) as txn:
            value = txn.get(key)  # 读取序列化后的value（bytes）
            if value is None:
                raise KeyError(f"键 {key} 不存在")
            return len(value) / (1024 ** 2)  # 字节转MB

    def get_data_keys(self, i) -> list:
        """
        返回第i个样本在`data_db`中的所有键的列表
        Args:
            i: 索引，默认检查第一个样本

        Returns:
            list: 数据键名列表
        """

        #return list(self[i].keys())
        return self.get_dict_keys(self[i])


    def get_data_value(self, i, key):
        """
        返回第i个样本对应于输入键的值。
        该值从`data_db`中检索。
        因为每个样本都存储在一个msgpack中,所以在返回值之前,我们需要先读取整个msgpack。
        Args:
            i: 索引
            key: 该索引的键（支持多层路径，如"mesh.v"）
        Returns:
            对应的值
        Raises:
            KeyError: 键不存在或路径无效时抛出
        """
        try:
            if isinstance(i, int):
                # 获取第i个样本的数据
                data = self[i]
            else:
                data= i
            # 拆分键路径（如"mesh.v" → ["mesh", "v"]）
            keys = key.split(".")
            # 初始化当前层级为data
            current = data
            # 逐层访问嵌套结构
            for sub_key in keys:
                # 检查当前层级是否为字典，且子键存在
                if not isinstance(current, dict) or sub_key not in current:
                    raise KeyError(f"路径无效：'{key}'（子键 '{sub_key}' 不存在或中间值非字典）")
                # 进入下一层级
                current = current[sub_key]
            # 返回最终值
            return current
        except KeyError as e:
            # 保留原始错误信息并抛出
            raise KeyError(f"键或路径不存在: {key}（详情：{str(e)}）")



    def get_data_specification(self, i: int) -> dict:
        """
        返回第i个样本的所有数据对象的规范。
        规范包括形状和数据类型。这假设每个数据对象都是`numpy.ndarray`。
        Args:
            i: 索引
        Returns:
            dict: 数据规范字典
        """
        spec = {}
        sample = self[i]
        for key in sample.keys():
            spec[key] = {}
            data = sample[key]
            if isinstance(data, np.ndarray):
                spec[key]["dtype"] = data.dtype
                spec[key]["shape"] = data.shape
            elif isinstance(data, dict):
                spec[key]["dtype"] = type(data).__name__
                spec[key]["keys"] = list(data.keys())
                spec[key]["shape"] = len(sample[key])
            else:
                spec[key]["dtype"] = type(data).__name__
        return spec

    def get_mapping(self, phy2log: bool = True):
        """
        获取逻辑索引与物理键的映射关系

        Args:
            phy2log: True=物理键到逻辑索引的映射关系，False=逻辑索引到物理键的映射关系

        Returns:
            dict: 映射关系 {物理键: 逻辑索引} or {逻辑索引: 物理键}
        """
        if phy2log:
            return {physical_key: logical_idx for logical_idx, physical_key in enumerate(self.read_keys)}
        else:
            return {logical_idx: physical_key for logical_idx, physical_key in enumerate(self.read_keys)}
    def get_sample(self, i: int) -> dict:
        """
        从`data_db`返回第i个样本（逻辑索引）
        Args:
            i: 逻辑索引
        Returns:
            dict: 样本数据字典
        Raises:
            IndexError: 如果索引超出范围
        """

        if 0 > i or self.nb_samples <= i:
            raise IndexError("所选样本编号超出范围: %d" % i)

        # 获取对应的物理键
        physical_key = self.read_keys[i]
        # 将物理键转换为带有尾随零的字符串
        key = Base.encode_str("{:010}".format(physical_key))
        obj = {}
        with self._lmdb_env.begin(db=self.data_db) as txn:
            _obj = msgpack.unpackb(txn.get(key), raw=False, use_list=True)
            for k in _obj:
                # 如果键存储为字节对象,则必须对其进行解码
                if isinstance(k, bytes):
                    _k = Base.decode_str(k)
                else:
                    _k = str(k)
                obj[_k] = Base.decode_data(msgpack.unpackb(
                    _obj[_k], raw=False, use_list=False
                ))

        return obj

    def get_samples(self, keys: list) -> list:
        """
        list所有连续样本
        Args:
            keys: 需要返回的索引对应的数据

        Returns:
            list: 所有样本组成的列表

        Raises:
            IndexError: 如果索引范围超出边界
        """
        samples_sum = []
        with self._lmdb_env.begin(db=self.data_db) as txn:
            for _i in keys:
                samples = {}
                # 获取对应的物理键
                try:
                    physical_key = self.read_keys[_i]
                except KeyError:
                    print(f"检测到数据库不存在键{_i},跳过...")
                    continue
                # 将样本编号转换为带有尾随零的字符串
                key =  Base.encode_str("{:010}".format(physical_key))
                # 从LMDB读取msgpack,解码其中的每个值,并将其添加到检索到的样本集合中
                obj = msgpack.unpackb(txn.get(key), raw=False, use_list=True)
                for k in obj:
                    print(k)
                    # 如果键存储为字节对象,则必须对其进行解码
                    if isinstance(k, bytes):
                        _k = Base.decode_str(k)
                    else:
                        _k = str(k)
                    samples[_k] = msgpack.unpackb(
                        obj[_k], raw=False, use_list=False, object_hook=Base.decode_data
                    )
                samples_sum.append(samples)

        return samples_sum


    def get_delete_sample(self, physical_key: int) -> dict:
        """
        读取已删除的样本数据（通过物理键）

        Args:
            physical_key: 物理键值

        Returns:
            dict: 已删除的样本数据

        Raises:
            ValueError: 如果物理键不存在或未被标记为删除
        """
        if physical_key not in self.deleted_keys:
            raise ValueError(f"物理键 {physical_key} 未被标记删除或不存在")

        # 检查物理键是否在有效范围内
        if physical_key < 0 or physical_key >= len(self.physical_keys):
            raise ValueError(f"物理键 {physical_key} 超出有效范围")

        # 将物理键转换为带有尾随零的字符串
        key =  Base.encode_str("{:010}".format(physical_key))
        obj = {}
        with self._lmdb_env.begin(db=self.data_db) as txn:
            # 从LMDB读取msgpack,并解码其中的每个值
            _obj = msgpack.unpackb(txn.get(key), raw=False, use_list=True)
            for k in _obj:
                # 如果键存储为字节对象,则必须对其进行解码
                if isinstance(k, bytes):
                    _k =  Base.decode_str(k)
                else:
                    _k = str(k)
                obj[_k] = msgpack.unpackb(
                    _obj[_k], raw=False, use_list=False, object_hook= Base.decode_data
                )
        return obj

    def __getitem__(self, key:int) -> dict:

        return self.get_sample(key)

    def __len__(self) -> int:
        return self.nb_samples

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        if len(self) == 0:
            return "\033[93m数据库为空\033[0m"

        out = "\033[93m"
        out += "类名:\t\t{}\n".format(self.__class__.__name__)
        out += "位置:\t\t'{}'\n".format(os.path.abspath(self.dirpath))
        out += "样本数量:\t{}\n".format(len(self))
        out += "物理存储大小:\t{}\n".format(len(self.physical_keys))
        out += "已删除样本:\t{}\n".format(len(self.deleted_keys))
        out += f"第一个数据所有键:\n\t{self.get_data_keys(0)}\n"
        out += f"第一个数据大小:\n\t{self.get_data_size(0):5f} MB\n"
        out += f"元数据键:\n\t{self.get_meta_keys()}\n"



        out += "数据键(第0个样本):\n"
        spec = self.get_data_specification(0)
        for key in spec:
            out += f"\t键: '{key}'\n"
            details = spec[key]
            for detail_key, detail_val in details.items():
                out += f"\t  {detail_key}:{detail_val}\n"


        out += "\n提示:\t使用 get_mapping() 查看逻辑索引与物理键的映射关系; "
        out += "\n\t使用 get_delete_sample(physical_key) 读取已删除的样本;"
        out += "\n\t如果数据库文件在固态硬盘,这样可以避免内存占用,请使用 with Reader(db_path) as db: data=db[i];"
        out += "\033[0m\n"
        return out

    def close(self):
        self._lmdb_env.close()


class Writer(Base):
    """

    用于将数据集的对象 ('numpy.ndarray') 写入闪电内存映射数据库 (LMDB),并带有MessagePack压缩。

    功能特点：
    - 支持数据的保存、修改、删除、插入操作
    - 使用双键管理系统：物理键和逻辑键分离
    - 标记删除而非物理删除，支持数据恢复
    - 兼容旧版本数据库格式
    - 支持多进程模式

    Note:
        db = Writer(dirpath=r'datasets/lmdb.db', map_size_limit=1024*100)
        # 元数据操作
        db.put_meta_("描述信息", "xxxx")
        db.put_meta_("元信息",{"version”:"1.0.0","list":[1,2]})
        db.put_meta_("列表",[1,2,3])

        # 基本操作
        data = {xx:np.array(xxx)}
        db.put_sample(data)                    # 在末尾添加样本
        db.insert_sample(5, data)             # 在指定位置插入样本
        db.change_sample(3, data)              # 修改指定位置的样本
        db.delete_sample(2)                    # 标记删除指定位置的样本
        db.restore_sample(10)                  # 恢复已删除的样本
        db.close()
    """

    def __init__(self, dirpath: str, map_size_limit: int,multiprocessing:bool=False):
        """
        初始化

        Args:
            dirpath:  应该写入LMDB的目录的路径。
            map_size_limit: LMDB的map大小,单位为MB。必须足够大以捕获打算存储在LMDB中所有数据。
        """
        self.dirpath = dirpath
        self.map_size_limit = map_size_limit  # Megabytes (MB)
        self.multiprocessing=multiprocessing
        self.stats=None  # 记录数据库状态

        # 键管理系统---原则上不允许删除数据
        self.physical_keys = []      # 所有物理存在的键
        self.read_keys = []          # 当前有效的读取键
        self.deleted_keys = set()    # 已删除的键
        self.nb_samples = 0          # 数据库大小


        # 检测参数
        if self.map_size_limit <= 0:
            raise ValueError(
                "LMDB map 大小必须为正数:{}".format(self.map_size_limit)
            )

        # 将 `map_size_limit` 从 B 转换到 MB
        map_size_limit <<= 20


        # 打开LMDB环境，检测用户路径是否带尾缀，带就以文件形式打开。
        subdir_bool =False if  bool(os.path.splitext(dirpath)[1])  else True
        if subdir_bool:
            os.makedirs(dirpath,exist_ok=True)
        try:
            if multiprocessing:
                self._lmdb_env = lmdb.open(
                    dirpath,
                    map_size=map_size_limit,
                    max_dbs=Base.NB_DBS,
                    writemap=True,        # 启用写时内存映射
                    metasync=False,      # 关闭元数据同步
                    map_async=True,      # 异步内存映射刷新
                    lock=True,           # 启用文件锁
                    max_spare_txns=32,   # 事务缓存池大小
                    subdir=subdir_bool         # 使用文件而非目录
                )
            
            else:
                self._lmdb_env = lmdb.open(dirpath,
                                        map_size=map_size_limit,
                                        max_dbs=Base.NB_DBS,
                                        subdir=subdir_bool)
        except lmdb.Error as e :
            raise ValueError(f"创建错误：{e} \t(map_size_limit设置创建 {map_size_limit >> 20} MB数据库(可能原因:数据库被其他程序占用中)")
        
        # 打开与环境关联的默认数据库
        self.data_db = self._lmdb_env.open_db(Base.DATA_DB) # 数据集
        self.meta_db = self._lmdb_env.open_db(Base.META_DB) # 数据信息

        # 加载键信息
        self._load_keys()




    def _load_keys(self):
        """加载键管理信息，兼容旧版本数据库"""
        with self._lmdb_env.begin(db=self.meta_db) as txn:
            # 尝试加载新版本的键管理信息
            physical_data = txn.get(b"__physical_keys__")
            read_data = txn.get(b"__read_keys__")
            deleted_data = txn.get(b"__deleted_keys__")
            nb_samples_data = txn.get(b"nb_samples")


            if physical_data and read_data:

                # 新版本数据库：使用键管理系统
                self.physical_keys = msgpack.unpackb(physical_data)
                self.read_keys = msgpack.unpackb(read_data)
                if deleted_data:
                    self.deleted_keys = set(msgpack.unpackb(deleted_data))
                else:
                    self.deleted_keys = set()
                # nb_samples 与 read_keys 保持一致
                self.nb_samples = len(self.read_keys)
                self.stats = "update_stats"
                print(f"\n\033[92m检测到{self.dirpath}数据库\033[93m<已有数据存在>,\033[92m数据库大小: {self.nb_samples}, 物理存储大小: {self.physical_size} \033[0m\n")

            if not physical_data and nb_samples_data:
                # 旧版本数据库：从现有数据重建键管理系统
                print("\033[93m检测到旧版本数据库，正在重建键管理系统...\033[0m")
                # 从meta_db获取样本数
                with self._lmdb_env.begin(db=self.meta_db) as txn:
                    nb_samples_data = txn.get(b"nb_samples")
                    if nb_samples_data:
                        try:
                            self.nb_samples = int(Base.decode_str(nb_samples_data))
                        except:
                            # 如果不是字符串，尝试用msgpack解码
                            self.nb_samples = msgpack.unpackb(nb_samples_data)
                    else:
                        # 如果没有样本数信息，从数据中统计
                        with self._lmdb_env.begin(db=self.data_db) as data_txn:
                            cursor = data_txn.cursor()
                            self.nb_samples = sum(1 for _ in cursor)
                # 重建键列表
                self.physical_keys = list(range(self.nb_samples))
                self.read_keys = list(range(self.nb_samples))
                self.deleted_keys = set()
                # 保存新的键管理系统
                self._save_keys()
                print(f"\033[92m成功重建键管理系统，数据库大小: {self.nb_samples}\033[0m")
                self.stats = "update_stats"

            if not nb_samples_data:
                self.stats = "create_stats"
                print(f"\n\033[92m检测到{self.dirpath}数据库\033[93m<数据为空>,\033[92m 启动创建模式\033[0m\n")




    def put_meta(self, key:str, value):
        """
        将任意类型的数据写入元数据库

        Args:
            key: 键名
            value: 任意可序列化的数据（支持str、list、dict等）

        """

        if isinstance(key, str):
            _key = Base.encode_str(key)
        else:
            _key = key
        with self._lmdb_env.begin(write=True, db=self.meta_db) as txn:
            # 使用msgpack序列化任意类型数据
            txn.put(_key, msgpack.packb(value, use_bin_type=True))
    def _save_keys(self):
        """保存
        键管理信息到 meta_db"""
        with self._lmdb_env.begin(write=True, db=self.meta_db) as txn:
            txn.put(b"__physical_keys__", msgpack.packb(self.physical_keys))
            txn.put(b"__read_keys__", msgpack.packb(self.read_keys))
            txn.put(b"__deleted_keys__", msgpack.packb(list(self.deleted_keys)))
            txn.put(b"nb_samples", msgpack.packb(self.nb_samples))
    def _write_data(self,new_physical_key,sample):
        # 存储样本数据
        with self._lmdb_env.begin(write=True, db=self.data_db) as txn:
            msg_pkgs = {}
            for key in sample:
                obj = Base.encode_data(sample[key])
                msg_pkgs[key] = msgpack.packb(obj, use_bin_type=True)
            physical_key_str = Base.encode_str("{:010}".format(new_physical_key))
            pkg = msgpack.packb(msg_pkgs, use_bin_type=True)
            txn.put(physical_key_str, pkg)
    @property
    def size(self):
        """数据库大小（有效样本数）- 与 nb_samples 保持一致"""
        return self.nb_samples

    @property
    def physical_size(self):
        """物理存储大小"""
        return len(self.physical_keys)
    def insert_sample(self,key: int, sample: dict, safe_model: bool = True):
        """
        在指定逻辑索引位置插入样本

        Args:
           key: 要插入的逻辑索引位置
           sample: 要插入的样本数据
           safe_model: 安全模式，如果开启则会提示确认
        """
        if key < 0 or key > self.nb_samples:
            raise ValueError(f"插入索引 {key} 超出范围 [0, {self.nb_samples}]")

        if safe_model:
            _ok = input(f"\033[93m将在逻辑索引 {key} 处插入新样本。确认请输入 'yes': \033[0m")
            if _ok.strip().lower() != "yes":
                print("用户取消插入操作")
                return

        # 开始处理插入
        new_physical_key = len(self.physical_keys) #新位置=尾插
        self._write_data(new_physical_key,sample)
        # 更新键管理系统
        self.physical_keys.append(new_physical_key)
        self.read_keys.insert(key, new_physical_key)
        self.nb_samples = len(self.read_keys)
        # 保存信息
        self._save_keys()
        print(f"\033[92m成功在逻辑索引 {key} 处插入新样本，数据库大小: {self.nb_samples}, 物理存储大小: {self.physical_size}\033[0m")
    def delete_sample(self, key: int):
        """
        删除指定逻辑索引位置的样本（标记删除）

        Args:
            key: 要删除的逻辑索引位置
        """
        if key < 0 or key >= self.nb_samples:
            raise ValueError(f"删除索引 {key} 超出范围 [0, {self.nb_samples - 1}]")
        # 获取物理键
        physical_key = self.read_keys[key]
        # 标记删除--从读取里删除
        self.read_keys.pop(key)
        self.deleted_keys.add(physical_key)
        self.nb_samples = len(self.read_keys)
        # 更新保存的信息
        self._save_keys()
        print(f"\033[92m成功标记删除逻辑索引 {key} 处的样本，当前数据库大小: {self.nb_samples}\033[0m")

    def change_sample(self, key: int, sample: dict, safe_model: bool = True):
        """

         修改键值

        Args:
            key: 键
            sample:  字典类型数据
            safe_model: 安全模式,如果开启,则修改会提示;


        """

        if key < 0 or key >= self.nb_samples:
            raise ValueError(f"修改索引 {key} 超出范围 [0, {self.nb_samples - 1}]")

        if safe_model:
            _ok = input("\033[93m请确认你的行为,因为这样做,会强制覆盖数据,无法找回!\n"
                        f"当前数据库大小为<< {self.nb_samples} >>,索引从< 0 >>开始计数,现在准备将修改<< {key} >>的值,同意请输入yes! 请输入:\033[93m")
            if _ok.strip().lower() != "yes":
                print(f"用户选择退出! 您输入的是{_ok.strip().lower()}")
                return

        physical_key = self.read_keys[key]
        self._write_data(physical_key,sample)
        print(f"\033[92m成功修改逻辑索引 {key} 处的样本\033[0m")

    def put_sample(self,sample:dict):
        """
        将传入内容的键和值放入`data_db` LMDB中。

        Notes:
            put_samples({'key1': value1, 'key2': value2, ...})

        Args:
            sample: 由str为键,numpy类型为值组成

        """
        try:
            # 生成新的物理键
            new_physical_key = len(self.physical_keys)
            self._write_data(new_physical_key,sample)
            # 更新键管理系统
            self.physical_keys.append(new_physical_key)
            self.read_keys.append(new_physical_key)
            self.nb_samples = len(self.read_keys)
            # 保存信息
            self._save_keys()
        except lmdb.MapFullError as e:
            raise AttributeError(
                "LMDB 的map_size 太小:%s MB, %s" % (self.map_size_limit, e)
            )



    ####################其他功能###############################################

    def get_mapping(self,phy2log=True):
        """
        获取逻辑索引与物理键的映射关系
        Args:
            phy2log:True=物理键到逻辑索引的映射关系，False=逻辑索引到物理键的映射关系

        Returns:
            dict: 映射关系{物理键: 逻辑索引} or {逻辑索引: 物理键}
        """

        if phy2log:
            return {physical_key: logical_idx for logical_idx, physical_key in enumerate(self.read_keys)}
        else:
            return {logical_idx: physical_key for logical_idx, physical_key in enumerate(self.read_keys)}







    def restore_sample(self, physical_key: int):
        """
        恢复标记删除的样本

        Args:
            physical_key: 要恢复的物理键
        """
        if physical_key not in self.deleted_keys:
            print(f"物理键 {physical_key} 未被标记删除")
            return
        # 从删除标记中移除
        self.deleted_keys.remove(physical_key)
        # 将恢复的样本添加到读取键的末尾
        self.read_keys.append(physical_key)
        # 更新样本计数
        self.nb_samples = len(self.read_keys)
        # 保存键信息和元数据
        self._save_keys()
        print(f"\033[92m成功恢复物理键 {physical_key} 的样本，当前数据库大小: {self.nb_samples}\033[0m")



    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        out = "\033[94m"
        out += f"类名:\t\t\t{self.__class__.__name__}\n"
        out += f"位置:\t\t\t'{os.path.abspath(self.dirpath)}'\n"
        out += f"LMDB的map_size:\t\t{self.map_size_limit}MB\n"
        out += f"数据库大小:\t\t{self.nb_samples}\n"
        out += f"物理存储大小:\t\t{self.physical_size}\n"
        out += f"已删除样本:\t\t{len(self.deleted_keys)}\n"
        out += f"当前模式:\t\t{self.stats}\n"
        out += "\033[0m\n"
        return out

    def close(self):
        """
        关闭环境。
        在关闭之前,将样本数写入`meta_db`,使所有打开的迭代器、游标和事务无效。

        """
        self._save_keys()
        self._lmdb_env.close()
        if sys.platform.startswith('win') and not self.multiprocessing:
            print(f"检测到windows系统, 请运行  fix_lmdb_windows_size('{self.dirpath}') 修复文件大小问题")




#######################################################################
#######################################################################
###########################其他工具 #####################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
def get_data_size(samples:dict):
    """
    检测sample字典的大小

    Args:
        samples (_type_): 字典类型数据

    Return:
        gb_required : 字典大小(GB)
    """
    # 检查数据类型
    gb_required = 0
    for key in samples:
        # 所有数据对象的类型必须为`numpy.ndarray`
        if not isinstance(samples[key], np.ndarray):
            raise ValueError(
                "不支持的数据类型:" "`numpy.ndarray` != %s" % type(samples[key])
            )
        else:
            gb_required += np.uint64(samples[key].nbytes)

    # 确保用户指定的假设RAM大小可以容纳要存储的样本数
    gb_required = float(gb_required / 10 ** 9)

    return gb_required



def fix_lmdb_windows_size(dirpath: str):
    """
    修复lmdb在windows系统上创建大小异常问题(windows上lmdb没法实时变化大小);

    Args:
        dirpath:  lmdb目录路径

    Returns:

    """
    try:
        db = Writer(dirpath=dirpath, map_size_limit=1)
        db.close()
    except Exception as e:
        print(f"修复完成,",e)



def merge_lmdb(target_dir: str, source_dirs: list, map_size_limit: int, multiprocessing: bool = False):
    """
    将多个源LMDB数据库合并到目标数据库
    
    Args:
        target_dir: 目标LMDB路径
        source_dirs: 源LMDB路径列表
        map_size_limit: 目标LMDB的map大小限制（MB）
        multiprocessing: 是否启用多进程模式
        
        
    Example:
        ```
        # 合并示例
        MergeLmdb(
            target_dir="merged.db",
            source_dirs=["db1", "db2"],
            map_size_limit=1024  # 1GB
        )
        ```

    """
    # 计算总样本数
    total_samples = 0
    readers = []
    for src_dir in source_dirs:
        reader = Reader(src_dir)
        readers.append(reader)
        total_samples += len(reader)
    
    # 创建目标Writer实例
    writer = Writer(target_dir, map_size_limit=map_size_limit, multiprocessing=multiprocessing)
    
    # 带进度条的合并过程
    with tqdm(total=total_samples, desc="合并数据库", unit="sample") as pbar:
        for reader in readers:
            for i in range(len(reader)):
                sample = reader[i]
                writer.put_sample(sample)
                pbar.update(1)
                pbar.set_postfix({"当前数据库": os.path.basename(reader.dirpath)})
    
    # 关闭所有Reader和Writer
    for reader in readers:
        reader.close()
    writer.close()




def split_lmdb(source_dir: str, target_dirs: list, map_size_limit: int, multiprocessing: bool = False):
    """
    将源LMDB数据库均匀拆分到多个目标数据库
    
    Args:
        source_dir: 源LMDB路径
        target_dirs: 目标LMDB路径列表
        map_size_limit: 每个目标LMDB的map大小限制（MB）
        multiprocessing: 是否启用多进程模式
        
    
    Example:
        ```
        SplitLmdb(
        source_dir="large.db",
        target_dirs=[f"split_{i}.db" for i in range(4)],
        map_size_limit=256
        )
        ```
    """
    n = len(target_dirs)
    writers = [Writer(d, map_size_limit=map_size_limit, multiprocessing=multiprocessing) for d in target_dirs]
    
    with Reader(source_dir) as reader:
        total_samples = len(reader)
        
        # 带进度条的拆分过程
        with tqdm(total=total_samples, desc="拆分数据库", unit="sample") as pbar:
            samples_per_writer = total_samples // n
            remainder = total_samples % n
            
            writer_idx = 0
            count_in_writer = 0
            
            for i in range(total_samples):
                sample = reader[i]
                writers[writer_idx].put_sample(sample)
                count_in_writer += 1
                
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({
                    "目标库": os.path.basename(writers[writer_idx].dirpath),
                    "进度": f"{writer_idx+1}/{n}"
                })
                
                # 判断是否切换到下一个Writer
                threshold = samples_per_writer + 1 if writer_idx < remainder else samples_per_writer
                if count_in_writer >= threshold:
                    writer_idx += 1
                    count_in_writer = 0
                    if writer_idx >= n:
                        break
    
    # 关闭所有Writer实例
    for w in writers:
        w.close()



def parallel_write(output_dir: str, 
                        file_list: list, 
                        process: callable, 
                        map_size_limit: int, 
                        num_processes: int, 
                        multiprocessing: bool = False,
                        temp_root: str = "./tmp", 
                        cleanup_temp: bool = True):
    """
    多进程处理JSON文件并写入LMDB

    Args:
        output_dir: 最终输出LMDB路径
        file_list: 文件路径列表
        process: 数据处理函数
        map_size_limit: 总LMDB的map大小限制(MB)
        num_processes: 进程数量
        multiprocessing: 是否启用多进程模式
        temp_root: 临时目录根路径（默认./tmp，尽量写在SSD,方便快速转换
        cleanup_temp: 是否清理临时目录（默认True）
        
        
    Example:
        ```
        
        def process(json_file):
            with open(json_file,"r") as f:
                data = json.loads(f.read())
            id=data["id_patient"]
            jaw = data["jaw"]
            labels = data["labels"]
            
            mesh = vedo.load( json_file.replace(".json",".obj"))
            vertices = mesh.vertices
            faces = mesh.cells


            out = {
                'mesh_faces':faces,
                'mesh_vertices':vertices,
                'vertex_labels':labels,
                "jaw":jaw,

            }
            return out
    

        
        if __name__ == '__main__':
            json_file_list = glob.glob("./*/*/*.json")
            print(len(json_file_list))
            
            sindre.lmdb.parallel_write(
                output_dir=dirpath,
                file_list=json_file_list[:16],
                process=process,
                map_size_limit=map_size_limit,
                num_processes=8,
                temp_root="./processing_temp", 
                cleanup_temp=False  
            )
    
    
        ```
    
    
    """
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)
    temp_dirs = [os.path.join(temp_root,f"process_{i}.db") for i in range(num_processes)]

    # 启动进程
    manager = mp.Manager()
    progress_queue = manager.Queue()
    processes = []
    
    try:
        for i in range(num_processes):
            p = mp.Process(
                target=_worker_write,
                args=(temp_dirs[i], file_list, process, 
                     map_size_limit//num_processes, multiprocessing, i, 
                     num_processes, progress_queue)
            )
            processes.append(p)
            p.start()

        # 主进度条
        with tqdm(total=len(file_list), desc="多进程处理", unit="file") as main_pbar:
            while any(p.is_alive() for p in processes):
                while not progress_queue.empty():
                    main_pbar.update(progress_queue.get())
                time.sleep(0.1)

        # 合并临时数据库
        merge_lmdb(
                    target_dir=output_dir,
                    source_dirs=temp_dirs,
                    map_size_limit=map_size_limit, 
                    multiprocessing=multiprocessing
                )
    except Exception as e:
        cleanup_temp =False
        print(f"处理失败: {str(e)}")
        traceback.print_exc()
        print(f"请手动合并目录: \
              MergeLmdb(target_dir={output_dir},\
              source_dirs={temp_dirs},\
              map_size_limit={map_size_limit},\
              multiprocessing={multiprocessing})")


    finally:
        # 清理进程资源
        for p in processes:
            p.join()
        
        # 按需清理临时目录
        if cleanup_temp:
            for d in temp_dirs:
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
            print(f"已清理临时目录: {temp_root}")
        else:
            print(f"保留临时目录: {temp_root}")
            
            
            
def _worker_write(temp_dir: str, 
                json_file_list: list, 
                process: callable, 
                map_size_limit: int, 
                multiprocessing: bool,
                process_id: int, 
                num_processes: int, 
                progress_queue):
    """
    子进程处理函数 (适配你的数据处理逻辑)
    """
    writer = Writer(temp_dir, map_size_limit=map_size_limit, multiprocessing=multiprocessing)
    
    # 带错误处理的处理流程
    processed_count = 0
    for idx, json_file in enumerate(json_file_list):
        # 分配任务给当前进程
        if idx % num_processes != process_id:
            continue
        
        try:
            # 执行数据处理
            out = process(json_file)
            
            if out:
                # 写入数据库
                writer.put_sample(out)
            else:
                print(f"函数返回值异常: {out}")
            processed_count += 1
            
            # 每处理10个文件报告一次进度
            if processed_count % 10 == 0:
                progress_queue.put(10)
                
        except Exception as e:
            print(f"\n处理失败: {json_file}")
            print(f"错误信息: {str(e)}")
            traceback.print_exc()
            continue
    
    # 报告剩余进度
    if processed_count % 10 != 0:
        progress_queue.put(processed_count % 10)
    
    writer.close()




class ReaderList:
    """组合多个LMDB数据库进行统一读取的类，提供序列协议的接口

    该类用于将多个LMDB数据库合并为一个逻辑数据集，支持通过索引访问和获取长度。
    内部维护数据库索引映射表和真实索引映射表，实现跨数据库的透明访问。

    Attributes:
        db_list (List[Reader]): 存储打开的LMDB数据库实例列表
        db_mapping (List[int]): 索引到数据库索引的映射表，每个元素表示对应索引数据所在的数据库下标
        real_idx_mapping (List[int]): 索引到数据库内真实索引的映射表，每个元素表示数据在对应数据库中的原始索引
    """

    def __init__(self, db_path_list: list,multiprocessing:bool=True):
        """初始化组合数据库读取器

        Args:
            db_path_list (List[str]): LMDB数据库文件路径列表，按顺序加载每个数据库
        """
        self.db_list = []
        self.db_mapping = []  # 数据库索引映射表
        self.real_idx_mapping = []  # 真实索引映射表

        for db_idx, db_path in enumerate(db_path_list):
            db = Reader(db_path, multiprocessing)
            db_length = len(db)
            self.db_list.append(db)
            # 扩展映射表
            self.db_mapping.extend([db_idx] * db_length)
            self.real_idx_mapping.extend(range(db_length))
            print(f"load: {db_path} --> len: {db_length}")

    def __len__(self) -> int:
        """获取组合数据集的总条目数

        Returns:
            int: 所有LMDB数据库的条目数之和
        """
        return len(self.real_idx_mapping)

    def __getitem__(self, idx: int):
        """通过索引获取数据条目

        Args:
            idx (int): 数据条目在组合数据集中的逻辑索引

        Returns:
            object: 对应位置的数据条目，具体类型取决于LMDB存储的数据格式

        Raises:
            IndexError: 当索引超出组合数据集范围时抛出
        """
        db_idx = self.db_mapping[idx]
        real_idx = self.real_idx_mapping[idx]
        return self.db_list[db_idx][real_idx]

    def close(self):
        """关闭所有打开的LMDB数据库连接

        该方法应在使用完毕后显式调用，确保资源正确释放
        """
        for db in self.db_list:
            db.close()

    def __del__(self):
        """析构函数，自动调用close方法释放资源

        注意：不保证析构函数会被及时调用，建议显式调用close()
        """
        self.close()

class ReaderSSD:
    """针对SSD优化的LMDB数据库读取器，支持高效随机访问

    该类针对SSD存储特性优化，每次读取时动态打开数据库连接，
    适合需要高并发随机访问的场景，可充分利用SSD的IOPS性能。

    Attributes:
        db_len (int): 数据库条目总数
        db_path (str): LMDB数据库文件路径
        multiprocessing (bool): 是否启用多进程模式
    """

    def __init__(self, db_path: str, multiprocessing: bool = False):
        """初始化SSD优化的LMDB读取器

        Args:
            db_path (str): LMDB数据库文件路径
            multiprocessing (bool, optional): 是否启用多进程支持。
                启用后将允许在多个进程中同时打开数据库连接。默认为False。
        """
        self.db_len = 0
        self.db_path = db_path
        self.multiprocessing = multiprocessing
        with Reader(self.db_path, multiprocessing=self.multiprocessing) as db:
            self.db_len = len(db)  # 修正: 使用传入的db变量

    def __len__(self) -> int:
        """获取数据库的总条目数

        Returns:
            int: 数据库中的条目总数
        """
        return self.db_len

    def __getitem__(self, idx: int) -> object:
        """通过索引获取单个数据条目

        每次调用时动态打开数据库连接，读取完成后立即关闭。
        适合随机访问模式，特别是在SSD存储上。

        Args:
            idx (int): 数据条目索引

        Returns:
            object: 索引对应的数据条目

        Raises:
            IndexError: 当索引超出有效范围时抛出
        """
        with Reader(self.db_path, multiprocessing=self.multiprocessing) as db:
            return db[idx]

    def get_batch(self, indices: list) :
        """批量获取多个数据条目

        优化的批量读取接口，在一个数据库连接中读取多个条目，
        减少频繁打开/关闭连接的开销。

        Args:
            indices (list[int]): 数据条目索引列表

        Returns:
            list[object]: 索引对应的数据条目列表

        Raises:
            IndexError: 当任何索引超出有效范围时抛出
        """
        with Reader(self.db_path, multiprocessing=self.multiprocessing) as db:
            return [db[idx] for idx in indices]


class ReaderSSDList:
    """组合多个SSD优化的LMDB数据库进行统一读取的类，提供序列协议的接口

    该类用于将多个SSD优化的LMDB数据库合并为一个逻辑数据集，支持通过索引访问和获取长度。
    内部维护数据库索引映射表和真实索引映射表，实现跨数据库的透明访问，同时保持SSD优化特性。

    Attributes:
        db_path_list (List[str]): LMDB数据库文件路径列表
        db_mapping (List[int]): 索引到数据库索引的映射表，每个元素表示对应索引数据所在的数据库下标
        real_idx_mapping (List[int]): 索引到数据库内真实索引的映射表，每个元素表示数据在对应数据库中的原始索引
        multiprocessing (bool): 是否启用多进程模式
    """

    def __init__(self, db_path_list: list, multiprocessing: bool = False):
        """初始化组合SSD优化数据库读取器

        Args:
            db_path_list (List[str]): LMDB数据库文件路径列表，按顺序加载每个数据库
            multiprocessing (bool, optional): 是否启用多进程支持。默认为False。
        """
        self.db_path_list = db_path_list
        self.db_mapping = []  # 数据库索引映射表
        self.real_idx_mapping = []  # 真实索引映射表
        self.multiprocessing = multiprocessing

        for db_idx, db_path in enumerate(db_path_list):
            # 使用ReaderSSD获取数据库长度而不保持连接
            db = ReaderSSD(db_path, multiprocessing)
            db_length = len(db)
            # 扩展映射表
            self.db_mapping.extend([db_idx] * db_length)
            self.real_idx_mapping.extend(range(db_length))
            print(f"load: {db_path} --> len: {db_length}")

    def __len__(self) -> int:
        """获取组合数据集的总条目数

        Returns:
            int: 所有LMDB数据库的条目数之和
        """
        return len(self.real_idx_mapping)

    def __getitem__(self, idx: int):
        """通过索引获取数据条目

        Args:
            idx (int): 数据条目在组合数据集中的逻辑索引

        Returns:
            object: 对应位置的数据条目，具体类型取决于LMDB存储的数据格式

        Raises:
            IndexError: 当索引超出组合数据集范围时抛出
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range")
        db_idx = self.db_mapping[idx]
        real_idx = self.real_idx_mapping[idx]
        db_path = self.db_path_list[db_idx]
        # 使用ReaderSSD动态打开数据库并获取条目
        db = ReaderSSD(db_path, self.multiprocessing)
        return db[real_idx]

    def get_batch(self, indices: list):
        """批量获取多个数据条目

        对同一数据库中的索引进行分组，然后使用对应数据库的get_batch方法批量读取，
        减少频繁打开/关闭连接的开销。

        Args:
            indices (list[int]): 数据条目索引列表

        Returns:
            list[object]: 索引对应的数据条目列表

        Raises:
            IndexError: 当任何索引超出有效范围时抛出
        """
        # 检查所有索引是否有效
        for idx in indices:
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of range")

        # 按数据库分组索引
        db_groups = {}
        for idx in indices:
            db_idx = self.db_mapping[idx]
            real_idx = self.real_idx_mapping[idx]
            if db_idx not in db_groups:
                db_groups[db_idx] = []
            db_groups[db_idx].append(real_idx)

        # 对每个数据库批量读取
        results = [None] * len(indices)
        for db_idx, real_indices in db_groups.items():
            db_path = self.db_path_list[db_idx]
            db = ReaderSSD(db_path, self.multiprocessing)
            # 获取该数据库中所有索引对应的数据
            batch_results = db.get_batch(real_indices)
            # 将结果放入正确的位置
            for i, real_idx in enumerate(real_indices):
                # 找到原始索引在indices中的位置
                original_idx_pos = indices.index(self._find_original_index(db_idx, real_idx))
                results[original_idx_pos] = batch_results[i]

        return results

    def _find_original_index(self, db_idx, real_idx):
        """根据数据库索引和真实索引找到原始索引"""
        # 找到第一个属于该数据库的索引位置
        first_db_idx = self.db_mapping.index(db_idx)
        # 计算该数据库内的偏移量
        return first_db_idx + real_idx




def get_data_value(current, key):
    """
    Args:
        key: 该索引的键（支持多层路径，如"mesh.v"）
    Returns:
        对应的值
    Raises:
        KeyError: 键不存在或路径无效时抛出
    """
    # 拆分键路径（如"mesh.v" → ["mesh", "v"]）
    keys = key.split(".")
    # 逐层访问嵌套结构
    for sub_key in keys:
        # 检查当前层级是否为字典，且子键存在
        if not isinstance(current, dict) or sub_key not in current:
            raise KeyError(f"路径无效：'{key}'（子键 '{sub_key}' 不存在或中间值非字典）")
        # 进入下一层级
        current = current[sub_key]
    # 返回最终值
    return current