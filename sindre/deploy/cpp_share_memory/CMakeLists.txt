cmake_minimum_required(VERSION 3.20)

# 设置项目名称
project(SharedMemory)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 添加可执行文件
add_executable(shared_memory py2cpp_share_memory.cpp)

# 如果需要链接额外的库（根据平台可能需要）
if(WIN32)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/source-charset:UTF-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:UTF-8>")
else()
    # Linux/macOS 平台可能需要链接 pthread
    find_package(Threads REQUIRED)
    target_link_libraries(shared_memory PRIVATE Threads::Threads)
endif()

# 设置编译选项
if(MSVC)
    # 针对 Visual Studio 的编译选项
    add_compile_options(/W4 /WX /utf8)
else()
    # 针对 GCC/Clang 的编译选项
    add_compile_options(-Wall -Wextra -Werror)
endif()