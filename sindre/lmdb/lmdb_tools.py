# -*- coding: UTF-8 -*-
import numpy as np
import os
from tqdm import tqdm
from sindre.lmdb import Writer, Reader

__all__ = ["get_data_value","get_dict_size", "ReaderList","ReaderSSDList","ReaderSSD",  "split_lmdb", "merge_lmdb","fix_lmdb_windows_size"]




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
        with Reader(db_path, multiprocessing=self.multiprocessing) as db:
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





def check_filesystem_is_ext4(current_path:str)->bool:
    """
    检测硬盘是否为ext4

    Args:
        current_path: 需要检测的磁盘路径

    Returns:
        True: 当前为ext4磁盘，支持自适应容量分配
        False: 当前不是ext4磁盘，不支持自适应容量分配

    """
    import psutil

    current_path = os.path.abspath(current_path)

    partitions = psutil.disk_partitions()

    for partition in partitions:
        if current_path.startswith(partition.mountpoint):
            fs_type = partition.fstype
            if fs_type == 'NTFS':
                print(f"当前路径<<{current_path}>>的文件系统类型是NTFS,不是ext4"
                      f"\n\033[91m注意lmdb会在window上无法按实际大小变化,mapsize为多,则申请多少空间（建议按需要写入的文件大小申请空间)\033[0m\n")
                return True
            else:
                print(f"\n当前路径<<{current_path}>>的文件系统类型不是NTFS.\033[92m\n可将mapsize最大化,db大小会按实际大小变化\033[0m\n")
                return False





def get_dict_size(samples:dict):
    """
    检测sample字典的占用空间

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







def merge_lmdb(target_dir: str,
               source_dirs: list,
               map_size_limit: int,
               multiprocessing: bool = False,
               batchsize=100):
    """
    将多个源LMDB数据库合并到目标数据库

    Args:
        target_dir: 目标LMDB路径
        source_dirs: 源LMDB路径列表
        map_size_limit: 目标LMDB的map大小限制（MB）
        multiprocessing: 是否启用多进程模式
        batchsize:一次性提交写入量

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
    cache_samples =[]
    with tqdm(total=total_samples, desc="合并数据库", unit="sample") as pbar:
        for reader in readers:
            for i in range(len(reader)):
                sample = reader[i]
                if len(cache_samples)>=batchsize:
                    writer.put_batch_sample(cache_samples)
                    cache_samples.clear()
                else:
                    cache_samples.append(sample)
                pbar.update(1)
                pbar.set_postfix({"当前数据库": os.path.basename(reader.dirpath)})
        # 剩余数据
        if len(cache_samples)>0:
            writer.put_batch_sample(cache_samples)

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


