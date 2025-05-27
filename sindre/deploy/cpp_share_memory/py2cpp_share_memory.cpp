#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <windows.h> // Windows 平台共享内存API
#include <atomic>

// 共享内存结构定义
struct SharedMemoryHeader {
    int ndim;               // 维度数
    int shape[2];           // 数组形状（最大二维）
    std::atomic<int> status; // 状态标志：0-空，1-已写入C++，2-已写入Python
    char padding[4];        // 对齐
};

class SimpleSharedMemory {
private:
    std::string name;
    HANDLE fileHandle;
    void* dataPtr;
    size_t dataSize;
    SharedMemoryHeader* header;
    char* arrayData;
    
public:
    SimpleSharedMemory(const std::string& name, int width = 512, int height = 512, bool create = true)
        : name(name) {
        
        // 计算总大小
        dataSize = width * height * sizeof(float);
        size_t totalSize = sizeof(SharedMemoryHeader) + dataSize;
        
        if (create) {
            // 创建共享内存
            fileHandle = CreateFileMapping(
                INVALID_HANDLE_VALUE,
                NULL,
                PAGE_READWRITE,
                0,
                static_cast<DWORD>(totalSize),
                name.c_str()
            );
            
            if (fileHandle == NULL) {
                std::cerr << "创建共享内存失败: " << GetLastError() << std::endl;
                exit(1);
            }
            
            // 映射视图
            dataPtr = MapViewOfFile(
                fileHandle,
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                totalSize
            );
            
            if (dataPtr == NULL) {
                std::cerr << "映射共享内存失败: " << GetLastError() << std::endl;
                CloseHandle(fileHandle);
                exit(1);
            }
            
            // 初始化头部
            header = static_cast<SharedMemoryHeader*>(dataPtr);
            header->ndim = 2;
            header->shape[0] = width;
            header->shape[1] = height;
            header->status.store(0); // 初始状态为空
            
            arrayData = reinterpret_cast<char*>(dataPtr) + sizeof(SharedMemoryHeader);
        } else {
            // 打开现有共享内存
            fileHandle = OpenFileMapping(
                FILE_MAP_ALL_ACCESS,
                FALSE,
                name.c_str()
            );
            
            if (fileHandle == NULL) {
                std::cerr << "打开共享内存失败: " << GetLastError() << std::endl;
                exit(1);
            }
            
            // 映射视图
            dataPtr = MapViewOfFile(
                fileHandle,
                FILE_MAP_ALL_ACCESS,
                0,
                0,
                0
            );
            
            if (dataPtr == NULL) {
                std::cerr << "映射共享内存失败: " << GetLastError() << std::endl;
                CloseHandle(fileHandle);
                exit(1);
            }
            
            header = static_cast<SharedMemoryHeader*>(dataPtr);
            arrayData = reinterpret_cast<char*>(dataPtr) + sizeof(SharedMemoryHeader);
            
            // 验证尺寸
            if (header->ndim != 2 || 
                header->shape[0] != width || 
                header->shape[1] != height) {
                    std::cerr << "共享内存尺寸不匹配 " << std::endl;
                    cleanup();
                    exit(1);
            }
        }
    }
    
    ~SimpleSharedMemory() {
        cleanup();
    }
    
    void cleanup() {
        if (dataPtr) {
            UnmapViewOfFile(dataPtr);
            dataPtr = nullptr;
        }
        
        if (fileHandle) {
            CloseHandle(fileHandle);
            fileHandle = nullptr;
        }
    }
    
    void write(const std::vector<float>& data) {
        // 等待Python读取完成
        while (header->status.load() != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // 复制数据
        memcpy(arrayData, data.data(), dataSize);
        
        // 设置状态为已写入C++
        header->status.store(1);
    }
    
    std::vector<float> read() {
        // 等待Python写入完成
        while (header->status.load() != 2) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // 读取数据
        std::vector<float> result(header->shape[0] * header->shape[1]);
        memcpy(result.data(), arrayData, dataSize);
        
        // 设置状态为空
        header->status.store(0);
        
        return result;
    }
    
    int getWidth() const {
        return header->shape[0];
    }
    
    int getHeight() const {
        return header->shape[1];
    }
};

// C++ 生产者/消费者示例
int main() {
    SetConsoleOutputCP(CP_UTF8);
    // 创建共享内存
    SimpleSharedMemory shm("SimpleShm", 512, 512, true);
    
    try {
        float counter = 1e-6;
        
        while (true) {
            // 生成数据
            std::vector<float> data(shm.getWidth() * shm.getHeight(), counter);
            
            // 写入数据到共享内存
            shm.write(data);
            std::cout << "写入数据，形状: [" << shm.getWidth() << ", " << shm.getHeight() << "], 值: " << counter << std::endl;
            
            // 等待Python处理并写入结果
            std::vector<float> result = shm.read();
            
            // 验证结果（这里只是简单打印第一个元素）
            std::cout << "读取数据，形状: [" << shm.getWidth() << ", " << shm.getHeight() << "], 第一个元素: " << result[0] << std::endl;
            
            // 增加计数器
            counter += 1.0f;
            
            // 控制循环速度
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
    }
    
    return 0;
}