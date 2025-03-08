# -*- coding: UTF-8 -*-
import shutil
import time
import traceback
from tqdm import tqdm
from sindre.lmdb.tools import *
import multiprocessing as mp
try:
    import lmdb
    import msgpack
except ImportError:
    raise ImportError(
        "Could not import the LMDB library `lmdb` or  `msgpack`. Please refer "
        "to https://github.com/dw/py-lmdb/  or https://github.com/msgpack/msgpack-python for installation "
        "instructions."
    )

__all__ = ["Reader", "Writer", "merge_lmdb", "repair_windows_size", "split_lmdb","parallel_write"]


class Reader(object):
    """
    用于读取包含张量(`numpy.ndarray`)数据集的对象。
    这些张量是通过使用MessagePack从Lightning Memory-Mapped Database (LMDB)中读取的。


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

        # 以只读模式打开LMDB环境
        subdir_bool = True if os.path.isdir(dirpath)  else False
        if multiprocessing:
            self._lmdb_env = lmdb.open(dirpath,
                    readonly=True, 
                    meminit=False,
                    max_dbs=NB_DBS,
                    max_spare_txns=32,
                    subdir=subdir_bool, 
                    lock=False)
        else:
            self._lmdb_env = lmdb.open(dirpath,
                                       readonly=True,
                                       max_dbs=NB_DBS,
                                       subdir=subdir_bool, 
                                       lock=True)

        # 打开与环境关联的默认数据库
        self.data_db = self._lmdb_env.open_db(DATA_DB)
        self.meta_db = self._lmdb_env.open_db(META_DB)

        # 读取元数据,BODGE:修复读取空数据库报错
        try:
            self.nb_samples = int(self.get_meta_str(NB_SAMPLES))
        except ValueError:
            self.nb_samples = 0

    def get_meta_key_info(self) -> set:
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
                key_set.add(decode_str(key))
        return key_set

    def get_data_key_info(self) -> set:
        """

        Returns:
            获取元数据库所有键

        """
        key_set = set()
        # 创建一个读事务和游标
        with self._lmdb_env.begin(db=self.data_db) as txn:
            cursor = txn.cursor()
            # 遍历游标并获取键值对
            for key, value in cursor:
                dict_v = msgpack.unpackb(value, raw=False, use_list=True)
                for k in dict_v.keys():
                    key_set.add(k)
        return key_set

    def get_meta_str(self, key) -> str:
        """
        将输入键对应的值作为字符串返回。
        该值从`meta_db`中检索。
        Args:
            key: 字符串或字节字符串

        Returns:
            str,输入键对应的值

        """

        if isinstance(key, str):
            _key = encode_str(key)
        else:
            _key = key

        with self._lmdb_env.begin(db=self.meta_db) as txn:
            _k = txn.get(_key)
            if isinstance(_k, bytes):
                return decode_str(_k)
            else:
                return str(_k)

    def get_data_keys(self, i: int = 0) -> list:
        """
        返回第i个样本在`data_db`中的所有键的列表。
        如果所有样本包含相同的键,则只需要检查第一个样本,因此默认值为`i=0`

        Args:
            i: 索引

        Returns:
            list类型对象

        """

        return list(self[i].keys())

    def get_data_value(self, i: int, key: str):
        """
        返回第i个样本对应于输入键的值。

        该值从`data_db`中检索。

        因为每个样本都存储在一个msgpack中,所以在返回值之前,我们需要先读取整个msgpack。

        Args:
            i: 索引
            key: 该索引的键

        Returns:
            对应的值

        """
        try:
            return self[i][key]
        except KeyError:
            raise KeyError("键不存在:{}".format(key))

    def get_data_specification(self, i: int) -> dict:
        """
        返回第i个样本的所有数据对象的规范。
        规范包括形状和数据类型。这假设每个数据对象都是`numpy.ndarray`。

        Args:
            i: 索引

        Returns:
            数据字典

        """
        spec = {}
        sample = self[i]
        for key in sample.keys():
            spec[key] = {}
            try:
                spec[key]["dtype"] = sample[key].dtype
                spec[key]["shape"] = sample[key].shape
            except KeyError:
                raise KeyError("键不存在:{}".format(key))

        return spec

    def get_sample(self, i: int) -> dict:
        """
        从`data_db`返回第i个样本。
        Args:
            i:  索引

        Returns:
            字典类型对象

        """

        if 0 > i or self.nb_samples <= i:
            raise IndexError("所选样本编号超出范围: %d" % i)

        # 将样本编号转换为带有尾随零的字符串
        key = encode_str("{:010}".format(i))

        obj = {}
        with self._lmdb_env.begin(db=self.data_db) as txn:
            # 从LMDB读取msgpack,并解码其中的每个值
            _obj = msgpack.unpackb(txn.get(key), raw=False, use_list=True)
            for k in _obj:
                # 如果键存储为字节对象,则必须对其进行解码
                if isinstance(k, bytes):
                    _k = decode_str(k)
                else:
                    _k = str(k)
                obj[_k] = msgpack.unpackb(
                    _obj[_k], raw=False, use_list=False, object_hook=decode_data
                )

        return obj

    def get_samples(self, i: int, size: int) -> list:
        """
        返回从`i`到`i + size`的所有连续样本。

        Notes:
         假设:
            * 从`i`到`i + size`的每个样本具有相同的键集。
            * 样本中的所有数据对象都是`numpy.ndarray`类型。
            * 与同一个键关联的值具有相同的张量形状和数据类型。


        Args:
            i: int, 开始索引
            size: int, 索引长度

        Returns:
            所有样本组成的list


        """
        if 0 > i or self.nb_samples <= i + size - 1:
            raise IndexError(
                "所选样本编号超出范围: %d 到 %d(大小:%d)" % (i, i + size, size)
            )

        # 基于第i个样本做出关于数据的假设
        samples_sum = []
        with self._lmdb_env.begin(db=self.data_db) as txn:
            for _i in range(i, i + size):
                samples = {}
                # 将样本编号转换为带有尾随零的字符串
                key = encode_str("{:010}".format(_i))
                # 从LMDB读取msgpack,解码其中的每个值,并将其添加到检索到的样本集合中
                obj = msgpack.unpackb(txn.get(key), raw=False, use_list=True)
                for k in obj:
                    # 如果键存储为字节对象,则必须对其进行解码
                    if isinstance(k, bytes):
                        _k = decode_str(k)
                    else:
                        _k = str(k)
                    samples[_k] = msgpack.unpackb(
                        obj[_k], raw=False, use_list=False, object_hook=decode_data
                    )
                samples_sum.append(samples)

        return samples_sum

    def __getitem__(self, key) -> list:
        """
        使用`get_sample()`从`data_db`返回样本。

        Args:
            key: int/slice类型的值

        Returns:
            对应索引对象

        """
        if isinstance(key, (int, np.integer)):
            _key = int(key)
            if 0 > _key:
                _key += len(self)
            if 0 > _key or len(self) <= _key:
                raise IndexError("所选样本超出范围:`{}`".format(key))
            return self.get_sample(_key)
        elif isinstance(key, slice):
            return [self[i] for i in range(*key.indices(len(self)))]
        else:
            raise TypeError("无效的参数类型:`{}`".format(type(key)))

    def __len__(self) -> int:
        """

        Returns:
            返回数据集中的样本数量。

        """
        return self.nb_samples

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        spec = self.get_data_specification(0)
        # 使用ANSI转义序列将输出文本设置为黄色
        out = "\033[93m"
        out += "类名:\t\t{}\n".format(self.__class__.__name__)
        out += "位置:\t\t'{}'\n".format(os.path.abspath(self.dirpath))
        out += "样本数量:\t{}\n".format(len(self))
        if len(self)<100:
            out += f"data_db所有键:\n\t{self.get_data_key_info()}\n"
            out += f"meta_db所有键:\n\t{self.get_meta_key_info()}\n"
        out += "数据键(第0个样本):"
        for key in self.get_data_keys():
            out += "\n\t'{}' <- 数据类型: {}, 形状: {}".format(
                key, spec[key]["dtype"], spec[key]["shape"]
            )
        out += "\n\t提示:如果需要查看更多键类型可以使用-->get_data_specification(i=1)查看. "
        out += "\033[0m\n"
        return out

    def close(self):
        """

        Returns:
            关闭环境。使打开的任何迭代器、游标和事务无效。

        """
        self._lmdb_env.close()


class Writer(object):
    """
    用于将数据集的对象 ('numpy.ndarray') 写入闪电内存映射数据库 (LMDB),并带有MessagePack压缩。
    Note:
    
    
        db =  sindre.lmdb.Writer(dirpath=r'datasets/lmdb', map_size_limit=1024*100,ram_gb_limit=3.0)
        db.set_meta_str("描述信息", "xxxx")
        
        data = {xx:np.array(xxx)} # 尽量占用ram_gb_limit内存
        
        gb_required = db.check_sample_size(data) # 计算数据占用内存(GB)

        db.put_samples(data) # 一次性写入,注意gb_required<ram_gb_limit限制
            
       
        db.close()
            
    
    """

    def __init__(self, dirpath: str, map_size_limit: int,multiprocessing:bool=False):
        """
        初始化

        Args:
            dirpath:  应该写入LMDB的目录的路径。
            map_size_limit: LMDB的map大小,单位为MB。必须足够大以捕获打算存储在LMDB中所有数据。
            multiprocessing: 是否开启多进程。
        """
        self.dirpath = dirpath
        self.map_size_limit = map_size_limit  # Megabytes (MB)
        #self.ram_gb_limit = ram_gb_limit  # Gigabytes (GB)
        self.keys = []
        self.nb_samples = 0
        self.multiprocessing=multiprocessing

        # 检测参数
        if self.map_size_limit <= 0:
            raise ValueError(
                "LMDB map 大小必须为正:{}".format(self.map_size_limit)
            )
        # if self.ram_gb_limit <= 0:
        #     raise ValueError(
        #         "每次写入的RAM限制 (GB) 必须为为正:{}".format(self.ram_gb_limit)
        #     )

        # 将 `map_size_limit` 从 B 转换到 MB
        map_size_limit <<= 20
        
         # 将 `map_size_limit` 从 B 转换到 GB
        #map_size_limit <<= 30

        # 打开LMDB环境
        subdir_bool = True if os.path.isdir(dirpath)  else False
        # 检测目录是否存在
        if not os.path.exists(os.path.dirname(dirpath)) and not subdir_bool:
            os.makedirs(os.path.dirname(dirpath),exist_ok=True)
        try:
            if multiprocessing:
                self._lmdb_env = lmdb.open(
                    dirpath,
                    map_size=map_size_limit,
                    max_dbs=NB_DBS,
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
                                        max_dbs=NB_DBS,
                                        subdir=subdir_bool)
        except lmdb.Error as e :
            raise ValueError(f"创建错误：{e} \t(map_size_limit设置创建 {map_size_limit >> 30} GB 数据库)")
        
        # 打开与环境关联的默认数据库
        self.data_db = self._lmdb_env.open_db(DATA_DB)
        self.meta_db = self._lmdb_env.open_db(META_DB)

        # 启动检测服务
        self.check_db_stats()

    def change_db_value(self, key: int, value: dict, safe_model: bool = True):
        """

         修改键值

        Args:
            key: 键
            value:  内容
            safe_model: 安全模式,如果开启,则修改会提示;


        """

        num_size = self.nb_samples
        if key < num_size:
            if safe_model:
                _ok = input("\033[93m请确认你的行为,因为这样做,会强制覆盖数据,无法找回!\n"
                            f"当前数据库大小为<< {num_size} >>,索引从< 0 >>0开始计数,现在准备将修改<< {key} >>的值,同意请输入yes! 请输入:\033[93m")
                if _ok.strip().lower() != "yes":
                    print(f"用户选择退出! 您输入的是{_ok.strip().lower()}")
                    sys.exit(0)
            self.change_value(key, value)
        else:
            raise ValueError(
                f"当前数据库大小为<< {num_size} >>,将修改<< {key} >>应该小于当前数据库大小,索引从<< 0 >>开始计数! \033[0m\n")

    def change_value(self, num_id: int, samples: dict):
        """

        通过指定索引,修改内容
        Args:
            num_id: 索引
            samples: 内容

        Returns:

        """
        # 对于每个样本,构建一个msgpack并将其存储在LMDB中
        with self._lmdb_env.begin(write=True, db=self.data_db) as txn:
            # 为每个数据对象构建一个msgpack
            msg_pkgs = {}
            for key in samples:
                # 确保当前样本是`numpy.ndarray`
                obj = samples[key]
                if not isinstance(obj, np.ndarray):
                    obj = np.array(obj)
                # 创建msgpack
                msg_pkgs[key] = msgpack.packb(obj, use_bin_type=True, default=encode_data)

                # LMDB键:样本编号作为带有尾随零的字符串
                key = encode_str("{:010}".format(num_id))

                # 构建最终的msgpack并将其存储在LMDB中
                pkg = msgpack.packb(msg_pkgs, use_bin_type=True)
                txn.put(key, pkg)

    def check_db_stats(self):
        """
        检查lmdb是继续写,还是新写

        """

        with self._lmdb_env.begin(db=self.meta_db) as txn:
            _k = txn.get(encode_str("nb_samples"))
            if not _k:
                self.db_stats = "create_stats"
                print(
                    f"\n\033[92m检测到{self.dirpath}数据库\033[93m<数据为空>,\033[92m 启动创建模式,键从<< {self.nb_samples} >>开始 \033[0m\n")
            else:
                if isinstance(_k, bytes):
                    self.nb_samples = int(decode_str(_k))
                else:
                    self.nb_samples = int(_k)
                self.db_stats = "auto_update_stats"
                if not self.multiprocessing:
                    print(
                    f"\n\033[92m检测到{self.dirpath}数据库\033[93m<已有数据存在>,\033[92m启动自动增量更新模式,键从<< {self.nb_samples} >>开始\033[0m\n")


    def check_sample_size(self,samples:dict):
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


    def put_samples(self, samples: dict):
        """
        将传入内容的键和值放入`data_db` LMDB中。

        Notes:
            函数假设所有值的第一个轴表示样本数。因此,单个样本必须在`numpy.newaxis`之前。

            作为Python字典:

                put_samples({'key1': value1, 'key2': value2, ...})

        Args:
            samples: 由字符串和numpy数组组成

        """
        try:
            # 对于每个样本,构建一个msgpack并将其存储在LMDB中
            with self._lmdb_env.begin(write=True, db=self.data_db) as txn:
                # 为每个数据对象构建一个msgpack
                msg_pkgs = {}
                for key in samples:
                    # 确保当前样本是`numpy.ndarray`
                    obj = samples[key]
                    if not isinstance(obj, np.ndarray):
                        obj = np.array(obj)
                        try:
                            # 检查是否存在 NaN 或 Inf
                            if np.isnan(obj).any() or np.isinf(obj).any() or obj.dtype==np.object_:
                                raise ValueError("\033[91m 数据中包含 NaN 或 Inf 或 obj,请检查数据.\033[0m\n")
                        except Exception as e:
                            # 不支持校验
                            pass 
                    # 创建msgpack
                    msg_pkgs[key] = msgpack.packb(obj, use_bin_type=True, default=encode_data)

                    # LMDB键:样本编号作为带有尾随零的字符串
                    key = encode_str("{:010}".format(self.nb_samples))

                    # 构建最终的msgpack并将其存储在LMDB中
                    pkg = msgpack.packb(msg_pkgs, use_bin_type=True)
                    txn.put(key, pkg)

                # 增加全局样本计数器
                self.nb_samples += 1
        except lmdb.MapFullError as e:
            raise AttributeError(
                "LMDB 的map_size 太小:%s MB, %s" % (self.map_size_limit, e)
            )

        # 将当前样本数写入`meta_db`
        self.set_meta_str(NB_SAMPLES, str(self.nb_samples))

    def set_meta_str(self, key, string: str):
        """
        将输入的字符串写入`meta_db`中的输入键。

        Args:
            key: string or bytestring
            string:  string

        """

        if isinstance(key, str):
            _key = encode_str(key)
        else:
            _key = key

        with self._lmdb_env.begin(write=True, db=self.meta_db) as txn:
            txn.put(_key, encode_str(string))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __repr__(self):
        out = "\033[94m"
        out += f"类名:\t\t\t{self.__class__.__name__}\n"
        out += f"位置:\t\t\t'{os.path.abspath(self.dirpath)}'\n"
        out += f"LMDB的map_size:\t\t{self.map_size_limit}MB\n"
        #out += f"每份数据RAM限制:\t\t{self.ram_gb_limit}GB\n"
        out += f"当前模式:\t\t{self.db_stats}\n"
        out += f"当前开始序号为:\t\t{self.nb_samples}\n"
        out += "\033[0m\n"
        return out

    def close(self):
        """
        关闭环境。
        在关闭之前,将样本数写入`meta_db`,使所有打开的迭代器、游标和事务无效。

        """
        self.set_meta_str(NB_SAMPLES, str(self.nb_samples))
        self._lmdb_env.close()
        if sys.platform.startswith('win') and not self.multiprocessing:
            print(f"检测到windows系统, 请运行  repair_windows_size({self.dirpath}) 修复文件大小问题")
           
            

def repair_windows_size(dirpath: str):
    """
    解决windows没法实时变化大小问题;

    Args:
        dirpath:  lmdb目录路径

    Returns:

    """

    db = Writer(dirpath=dirpath, map_size_limit=1)
    db.close()



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
        merge_lmdb(
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
                writer.put_samples(sample)
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
        split_lmdb(
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
                writers[writer_idx].put_samples(sample)
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
              merge_lmdb(target_dir={output_dir},\
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
                writer.put_samples(out)
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
    