# -*- coding: UTF-8 -*-

import os
import numpy as np
import sys
import pickle
try:
    import lmdb
    import msgpack
except ImportError:
    raise ImportError(
        "Could not import the LMDB library `lmdb` or  `msgpack`. Please refer "
        "to https://github.com/dw/py-lmdb/  or https://github.com/msgpack/msgpack-python or https://github.com/python-lz4/python-lz4 for installation "
        "instructions."
    )

__all__ = [ "Reader", "Writer"]

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
                # 修复用户用list包裹多个对象问题;
                if data.size==1:
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
                if not self.multiprocessing:
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

    def put_batch_sample(self,samples:list[dict]):
        """
        将传入内容的键和值放入`data_db` LMDB中。

        Notes:
            put_batch_sample({'key1': value1},{'key2': value2})

        Args:
            samples: 由str为键,numpy类型为值组成的list

        """
        try:
            with self._lmdb_env.begin(write=True, db=self.data_db) as txn:
                for sample in samples:
                    # 生成新的物理键
                    new_physical_key = len(self.physical_keys)
                    # 写入数据
                    msg_pkgs = {}
                    for key in sample:
                        obj = Base.encode_data(sample[key])
                        msg_pkgs[key] = msgpack.packb(obj, use_bin_type=True)
                    physical_key_str = Base.encode_str("{:010}".format(new_physical_key))
                    pkg = msgpack.packb(msg_pkgs, use_bin_type=True)
                    txn.put(physical_key_str, pkg)
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



