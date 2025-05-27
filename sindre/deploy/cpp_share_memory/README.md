C++ 与 Python 共享内存通信系统



项目概述



本项目实现了 C++ 与 Python 之间通过共享内存进行高效数据通信的功能，支持以下流程：

`C++ 生成数据 → 写入共享内存 → Python 读取并处理数据 → 写回共享内存 → C++ 读取结果`

适用于需要跨语言传输大数据的场景（如图像处理、数值计算等），避免传统序列化方式的性能损耗。


环境要求



### C++ 端&#xA;



*   **操作系统**：Windows / Linux /macOS（本示例以 Windows 为主）


*   **工具链**：



    *   CMake ≥ 3.10


    *   C++17 编译器（如 MSVC、GCC、Clang）


*   **依赖库**：无（仅使用标准库和系统 API）


### Python 端&#xA;



*   **环境**：Python ≥ 3.6


*   **依赖库**：




```
pip install numpy
```

目录结构





```
project/


├─ cpp/                # C++ 源代码


│  ├─ main.cpp         # 主程序


│  └─ CMakeLists.txt   # 构建配置


├─ python/             # Python 脚本


│  └─ main.py          # 数据处理程序


└─ README.md           # 说明文档
```

编译与运行步骤



### 1. 编译 C++ 程序（以 Windows 为例）&#xA;

#### 步骤 1：创建构建目录并进入&#xA;



```
cd cpp


mkdir build && cd build
```

#### 步骤 2：生成构建文件（使用 CMake）&#xA;



```
cmake ..
```

#### 步骤 3：编译程序&#xA;



```
cmake --build . --config Release  # 生成 Release 版本（可选）
```

#### 步骤 4：运行 C++ 程序&#xA;



```
.\shared_memory_example.exe  # Windows


\# 或


./shared_memory_example     # Linux/macOS
```

### 2. 运行 Python 数据处理脚本&#xA;



```
cd ../python


python main.py
```

数据流程说明





1.  **C++ 生成数据**C++ 程序生成一个形状为 `(512, 512)` 的浮点型数组，初始值为 `1e-6`，每次循环递增 `1`，并写入共享内存。


2.  **Python 读取并处理数据**Python 脚本从共享内存读取数据，打印形状和唯一值，对数据执行 `+1.0` 操作后写回共享内存。


3.  **C++ 读取处理结果**C++ 程序读取 Python 处理后的数据，验证第一个元素并继续循环。


关键代码说明



### C++ 共享内存核心逻辑&#xA;



```
// 写入数据（等待 Python 清空内存）


void write(const std::vector\<float>& data) {


    while (header->status.load() != 0) {


    std::this_thread::sleep_for(std::chrono::milliseconds(10));


  }


    memcpy(arrayData, data.data(), dataSize);


    header->status.store(1); // 标记为 C++ 已写入


}


// 读取数据（等待 Python 写入结果）


std::vector\<float> read() {


    while (header->status.load() != 2) {


    std::this_thread::sleep_for(std::chrono::milliseconds(10));

 }


    std::vector\<float> result(dataSize / sizeof(float));


    memcpy(result.data(), arrayData, dataSize);


    header->status.store(0); // 清空状态


    return result;


}
```

### Python 共享内存核心逻辑&#xA;



```
def read_from_cpp(self) -> np.ndarray:


     while self.get_status() != 1:  # 等待 C++ 写入


         time.sleep(0.01)


     ptr = self.metadata_size


     array = np.frombuffer(self.shm.buf\[ptr:], dtype=self.dtype).reshape(self.shape)


     self.set_status(0)  # 清空状态


     return array


def write_to_cpp(self, array: np.ndarray):


     while self.get_status() != 0:  # 等待 C++ 读取


         time.sleep(0.01)


     self.shm.buf\[self.metadata_size:] = array.tobytes()


     self.set_status(2)  # 标记为 Python 已写入
```

常见问题解决



### 1. 中文显示乱码（Windows 平台）&#xA;

#### 原因&#xA;

控制台默认编码（GBK）与代码文件编码（UTF-8）不兼容。


#### 解决方案&#xA;



*   **方案一**：在 C++ 代码中设置控制台为 UTF-8 编码




```
\#include \<Windows.h>


int main() {


     SetConsoleOutputCP(CP_UTF8); // 设置输出编码为 UTF-8


     std::cout << "中文测试: 共享内存示例" << std::endl;


     return 0;


}
```



*   **方案二**：使用宽字符输出




```
std::wcout << L"中文测试: 共享内存示例" << std::endl;
```



*   **方案三**：创建批处理文件运行程序




```
@echo off


chcp 65001 > nul  // 设置为 UTF-8 编码


your_program.exe
```

### 2. 共享内存未找到&#xA;



*   确保 C++ 程序先运行（创建共享内存），再启动 Python 脚本。


*   检查共享内存名称是否一致（默认 `SimpleShm`），可在构造函数中修改名称。


### 3. 数据形状不匹配&#xA;



*   确保 C++ 和 Python 代码中指定的形状（`shape=(512, 512)`）一致。


*   检查维度数（`ndim=2`）是否匹配。


