# win工具库

## 1. 简介

1. 集成一些针对windows系统的工具及应用


## 2. 内容

#### windows窗口设置为透明

- 原理
    - 通过win api 查找窗口，强制修改窗口透明度

- 用法

```python
from sindre.win_tools import taskbar
# 设置任务栏，透明度为255，
taskbar.set_windows_alpha(255,"Shell_TrayWnd")
```


#### 将目录下所有py文件编译为pyd

```python
from sindre.win_tools import tools
tools.py2pyd(r"C:\Users\sindre\Downloads\55555",clear_py=False)
```

#### 在python中调用pip安装包

```python
from sindre.win_tools import tools
# 如安装到指定目录下，需要调用，调用时sys.path.insert(0, target_dir)
tools.pip_install(package_name="numpy==1.24.1",target_dir="./target")
tools.pip_install(requirements_path="./requirements.txt",target_dir="./target")
tools.pip_install(package_name="numpy==1.24.1")
```


#### 用exe运行  python xx.py

1. 创建个app.c文件
```C
#include <windows.h>

int main() {
    // 获取当前可执行文件的路径
    char exePath[MAX_PATH];
    GetModuleFileNameA(NULL, exePath, MAX_PATH);

    // 去除文件名部分，只保留目录路径
    char drive[_MAX_DRIVE];
    char dir[MAX_PATH];
    _splitpath_s(exePath, drive, sizeof(drive), dir, sizeof(dir), NULL, 0, NULL, 0);

    // 组合目录路径
    char dirPath[MAX_PATH];
    sprintf_s(dirPath, sizeof(dirPath), "%s%s", drive, dir);

    // 构建app.py的绝对路径
    char appPath[MAX_PATH];
    sprintf_s(appPath, sizeof(appPath), "%sapp.py", dirPath);

    // 构建python的绝对路径
    char pythonPath[MAX_PATH];
    sprintf_s(pythonPath, sizeof(pythonPath), "%spy/python.exe", dirPath);

    // 构建启动命令
    char command[MAX_PATH * 2];
    sprintf_s(command, sizeof(command), "\"%s\" \"%s\"", pythonPath, appPath);

    // 创建进程并执行命令
    STARTUPINFOA si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    
    if (CreateProcessA(NULL, command, NULL, NULL, FALSE, 0, NULL, NULL, &si, &pi)) {
        // 等待进程结束
        WaitForSingleObject(pi.hProcess, INFINITE);
        
        // 关闭进程和线程的句柄
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    }

    return 0;
}
        
```
2. 创建图片，新建logo.rc文件
```
IDI_ICON1  ICON  DISCARDABLE  F:/My_Github/ToolKit/py2nsis/py2nsis/config/logo.ico'
        
```

3. 创建app_setup.py脚本，用于编译app.c
```python
from distutils.ccompiler import new_compiler
import distutils.sysconfig
import sys
import os
from pathlib import Path

def compile(src):
    src = Path(src)
    cc = new_compiler()
    exe = src.stem
    cc.add_include_dir(distutils.sysconfig.get_python_inc())
    cc.add_library_dir(os.path.join(sys.base_exec_prefix, 'libs'))
    # First the CLI executable
    objs = cc.compile([str(src),"logo.rc"])
    cc.link_executable(objs, exe)
    # Now the GUI executable
    # cc.define_macro('WINDOWS')
    # objs = cc.compile([str(src)])
    # cc.link_executable(objs, exe + 'w', extra_preargs=['/ICON:{self.icon}']) 
```


4. 运行 "python.exe   app_setup.py" 即可生成app.exe, 
   1. 双击即可代替运行 "py/python.exe   app.py"



#### 自动安装python
```python

from sindre.win_tools import tools
tools.python_installer(install_dir=r"C:\Users\sindre\Downloads\55555",version='3.9.6')
```


#### 进行nsis封装

```python

from sindre.win_tools import tools
tools.exe2nsis(work_dir=r"C:/55555",
         files_to_compress=[f"C:/55555/{i}" for i in  ["app", "app.exe", "app.py"]],
         exe_name="app.exe")
```
将生成![exe2nsis.png](img/exe2nsis.png)

## 3. API:

::: win_tools.tools
::: win_tools.taskbar


