# -*- coding: UTF-8 -*-
import numpy as np
import os
import sys


def encode_str(string, encoding="utf-8", errors="strict"):
    """返回输入字符串的编码字节对象。

    Parameters
    ----------
    string : string
    encoding : string
        Default is `utf-8`.
    errors : string
       指定应如何处理编码错误。默认为 `strict`.
    """
    return str(string).encode(encoding=encoding, errors=errors)


def decode_str(obj, encoding="utf-8", errors="strict"):
    """将输入字节对象解码为字符串。

    Parameters
    ----------
    obj : byte object
    encoding : string
        Default is `utf-8`.
    errors : string
       指定应如何处理编码错误。默认为 `strict`.
    """
    return obj.decode(encoding=encoding, errors=errors)

# 支持的序列化类型
TYPES = {"str": 1, "ndarray": 2}

# 默认数据库数
NB_DBS = 2

# 默认数据库的名称
DATA_DB = encode_str("data_db")
META_DB = encode_str("meta_db")

# 元数据的默认键
NB_SAMPLES = encode_str("nb_samples")


    
    
def encode_data(obj):
    """
    返回一个字典，该字典包含对输入数据对象进行编码后的信息。

    Args:
        obj : 数据对象
            如果传入的数据对象既不是字符串也不是普通的 NumPy 数组，
            则该对象将按原样返回。
    """
    if isinstance(obj, str):
        # 如果是字符串，返回一个字典，包含类型标识和字符串数据
        return {b"type": TYPES["str"], b"data": obj}
    elif isinstance(obj, np.ndarray):
        return {
            b"type": TYPES["ndarray"],
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
            b"data": obj.tobytes()
        }
    
    else:
        # 对于其他类型的对象，假设用户知道自己在做什么，按原样返回对象
        return obj



def decode_data(obj):
    """
    解码一个序列化的数据对象。

    Args:
        obj : Python 字典
            一个描述序列化数据对象的字典。
    """
    try:
        if TYPES["str"] == obj[b"type"]:
            return obj[b"data"]
        elif TYPES["ndarray"] == obj[b"type"]:
            return np.frombuffer(obj[b"data"], dtype=np.dtype(obj[b"dtype"])).reshape(obj[b"shape"])
        else:
            # # 对于其他类型的对象，假设用户知道自己在做什么，按原样返回对象
            return obj
        
    except KeyError:
        return obj
    except ValueError as ve:
        print(f"解码时出现值错误: {ve}")
        return obj
    except TypeError as te:
        print(f"解码时出现类型错误: {te}")
        return obj



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






if __name__ == '__main__':
    # 创建一个 numpy 包裹的字符串
    arr_str = np.array(["是","ss",1,1e-4]).reshape(2,2)
    encoded = encode_data(arr_str)
    decoded =  decode_data(encoded)
    print(type(arr_str))
    #print(np.array(arr_str.tolist()))
    print(encoded,"\n")
    print(decoded)