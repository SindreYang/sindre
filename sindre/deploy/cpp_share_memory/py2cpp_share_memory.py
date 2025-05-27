import numpy as np
import time
import struct
from multiprocessing import shared_memory

class SimpleSharedMemory:
    def __init__(self, name="SimpleShm", shape=(512, 512), dtype=np.float32, create=False):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.ndim = len(shape)
        # 元数据大小：4字节维度数 + 2*4字节形状 + 4字节状态标志
        self.metadata_size = 4 + self.ndim * 4 + 4
        
        # 计算总大小（元数据 + 数据）
        self.data_size = int(np.prod(shape) * self.dtype.itemsize)
        self.total_size = self.metadata_size + self.data_size
        
        self.shm = shared_memory.SharedMemory(name=name, create=create, size=self.total_size)
        if create:
            self._write_metadata()
            # 初始化状态：0表示空
            self.set_status(0)
        else:
            self._read_metadata()
    
    def _write_metadata(self):
        ptr = 0
        # 写入维度数（4字节int）
        struct.pack_into('i', self.shm.buf, ptr, self.ndim)
        ptr += 4
        # 写入形状（每个维度4字节int）
        struct.pack_into(f'{self.ndim}i', self.shm.buf, ptr, *self.shape)
        ptr += self.ndim * 4
        
    def _read_metadata(self):
        ptr = 0
        # 读取维度数
        self.ndim = struct.unpack_from('i', self.shm.buf, ptr)[0]
        ptr += 4
        # 读取形状
        self.shape = struct.unpack_from(f'{self.ndim}i', self.shm.buf, ptr)
        ptr += self.ndim * 4
        self.dtype = self.dtype  # 保持类型一致
    
    def _get_status_offset(self):
        return 4 + self.ndim * 4  # 维度数和形状之后的位置
    
    def set_status(self, status):
        """设置状态标志：0-空，1-已写入C++，2-已写入Python"""
        offset = self._get_status_offset()
        struct.pack_into('i', self.shm.buf, offset, status)
    
    def get_status(self):
        """获取状态标志"""
        offset = self._get_status_offset()
        return struct.unpack_from('i', self.shm.buf, offset)[0]
    
    def read_from_cpp(self) -> np.ndarray:
        """从C++读取数据"""
        while self.get_status() != 1:  # 等待C++写入
            time.sleep(0.01)
            
        ptr = self.metadata_size
        array = np.frombuffer(
            self.shm.buf[ptr:ptr+self.data_size], 
            dtype=self.dtype
        ).reshape(self.shape)
        
        # 设置状态为空
        self.set_status(0)
        
        return array
        
    def write_to_cpp(self, array: np.ndarray):
        """写入数据给C++读取"""
        while self.get_status() != 0:  # 等待C++读取
            time.sleep(0.01)
            
        array = array.astype(self.dtype).reshape(self.shape)
        ptr = self.metadata_size
        self.shm.buf[ptr:ptr+array.nbytes] = array.tobytes()
        
        # 设置状态为已写入Python
        self.set_status(2)
        
    def cleanup(self):
        self.shm.close()
        if hasattr(self.shm, 'unlink'):
            self.shm.unlink()

# Python 端处理示例
if __name__ == "__main__":
    # 打开现有共享内存（由C++创建）
    shm = SimpleSharedMemory(create=False, shape=(512, 512), dtype=np.float32)
    try:
        while True:
            # 从C++读取数据
            data = shm.read_from_cpp()
            print(f"读取数据，形状: {data.shape}, 唯一值: {np.unique(data)}")
            
            # 处理数据（这里简单地将每个元素加1）
            processed_data = data + 1.0
            
            # 写回给C++
            shm.write_to_cpp(processed_data)
            print(f"写回数据，形状: {processed_data.shape}, 唯一值: {np.unique(processed_data)}")
            
    except KeyboardInterrupt:
        shm.cleanup()