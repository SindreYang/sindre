import numpy as np
import time
import struct
from multiprocessing import shared_memory, Lock

class SimpleSharedMemory:
    def __init__(self, name="SimpleShm", shape=(512, 512), dtype=np.float32, create=True):
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.ndim = len(shape)
        # 元数据大小：4字节维度数 + ndim*4字节形状 + 4字节状态标志 + 4字节锁
        self.metadata_size = 4 + self.ndim * 4 + 4 + 4
        
        # 计算总大小（元数据 + 数据）
        self.data_size = int(np.prod(shape) * self.dtype.itemsize)
        self.total_size = self.metadata_size + self.data_size
        
        # 用于状态标志的锁
        self.status_lock = Lock()
        
        if create:
            self.shm = shared_memory.SharedMemory(
                name=name, create=True, size=self.total_size
            )
            self._write_metadata()
            # 初始化状态：0表示空，1表示已写入
            self.set_status(0)
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)
            self._read_metadata()
        
      
    
    def _get_status_offset(self):
        return 4 + self.ndim * 4  # 维度数和形状之后的位置
    
    def _get_lock_offset(self):
        return self._get_status_offset() + 4  # 状态标志之后的位置
    
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
    
    def set_status(self, status):
        """设置状态标志：0-空，1-已写入"""
        with self.status_lock:
            offset = self._get_status_offset()
            struct.pack_into('i', self.shm.buf, offset, status)
    
    def get_status(self):
        """获取状态标志"""
        with self.status_lock:
            offset = self._get_status_offset()
            return struct.unpack_from('i', self.shm.buf, offset)[0]
    
    def write(self, array: np.ndarray):
        """写入数据，如果内存未清空则等待"""
        while self.get_status() == 1:  # 等待内存被清空
            time.sleep(0.01)  # 短暂休眠避免忙等待
            
        array = array.astype(self.dtype).reshape(self.shape)
        ptr = self.metadata_size  # 元数据后的偏移量
        # 直接写入内存视图
        self.shm.buf[ptr:ptr+array.nbytes] = array.tobytes()
        # 设置状态为已写入
        self.set_status(1)
        
    def read(self) -> np.ndarray:
        """读取数据并清空内存"""
        while self.get_status() == 0:  # 等待数据写入
            time.sleep(0.01)  # 短暂休眠避免忙等待
            
        ptr = self.metadata_size
        array = np.frombuffer(
            self.shm.buf[ptr:ptr+self.data_size], 
            dtype=self.dtype
        ).reshape(self.shape)
        
        # 清空内存（可选：用零填充）
        # self.shm.buf[ptr:ptr+self.data_size] = bytes(self.data_size)
        
        # 设置状态为空
        self.set_status(0)
        
        return array
        
    def cleanup(self):
        self.shm.close()
        self.shm.unlink()

# 写入端示例
if __name__ == "__main__":
    shm = SimpleSharedMemory(create=True, shape=(512, 512), dtype=np.float32)
    try:
        count =1e-6
        while True:
            data = (np.ones((512, 512))*count).astype(np.float32)
            shm.write(data)
            print(f"写入数据，形状：{data.shape}，{np.unique(data)}")
            time.sleep(0.1)
            count+=1
    except KeyboardInterrupt:
        shm.cleanup()

"""
# 读取端（另一个脚本）
import time
from memory_writer import SimpleSharedMemory
import numpy as np


# 读取端示例（另一个脚本）
if __name__ == "__main__":
    shm = SimpleSharedMemory(create=False, shape=(512, 512), dtype=np.float32)
    try:
        while True:
            data = shm.read()
            print(f"读取数据，值：{np.unique(data)} {data.shape}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        shm.cleanup()

        
"""