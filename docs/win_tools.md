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



#### win32提示框
* 针对不想安装其他gui包，但需要弹窗提示的现象

```python

import win32api,win32con
 
 
##提醒OK消息框
win32api.MessageBox(0, "这是一个测试提醒OK消息框", "提醒",win32con.MB_OK)
 
##是否信息框
win32api.MessageBox(0, "这是一个测试是否信息框", "提醒",win32con.MB_YESNO)
 
##说明信息框
win32api.MessageBox(0, "这是一个测试说明信息框", "提醒",win32con.MB_HELP)
 
####警告信息框
win32api.MessageBox(0, "这是一个测试警告信息框", "提醒",win32con.MB_ICONWARNING)
 
##疑问信息框
win32api.MessageBox(0, "这是一个测试疑问信息框", "提醒",win32con.MB_ICONQUESTION)
 
##提示信息框
win32api.MessageBox(0, "这是一个测试提示信息框", "提醒",win32con.MB_ICONASTERISK)
 
##确认信息框
win32api.MessageBox(0, "这是一个测试确认信息框", "提醒",win32con.MB_OKCANCEL)
 
##重试信息框
win32api.MessageBox(0, "这是一个测试重试信息框", "提醒",win32con.MB_RETRYCANCEL)
 
##是否取消信息框
win32api.MessageBox(0, "这是一个测试是否取消信息框", "提醒",win32con.MB_YESNOCANCEL)


# 弹出带有三个按钮的消息框  
result = win32api.MessageBox(0, "这是一个带有三个按钮的测试消息框", "标题", win32con.MB_YESNOCANCEL)

# 检测用户点击的按钮  
if result == win32con.IDYES:
    print("用户点击了 Yes 按钮")
elif result == win32con.IDNO:
    print("用户点击了 No 按钮")
elif result == win32con.IDCANCEL:
    print("用户点击了 Cancel 按钮")
```

```
WIN32CON.MB_OK = 0
WIN32CON.MB_OKCANCEL = 1
WIN32CON.MB_ABORTRETRYIGNORE = 2
WIN32CON.MB_YESNOCANCEL = 3
WIN32CON.MB_YESNO = 4
WIN32CON.MB_RETRYCANCEL = 5
WIN32CON.MB_ICONHAND = 16
WIN32CON.MB_ICONQUESTION = 32
WIN32CON.MB_ICONEXCLAMATION = 48
WIN32CON.MB_ICONASTERISK = 64
WIN32CON.MB_ICONWARNING = WIN32CON.MB_ICONEXCLAMATION
WIN32CON.MB_ICONERROR = WIN32CON.MB_ICONHAND
WIN32CON.MB_ICONINFORMATION = WIN32CON.MB_ICONASTERISK
WIN32CON.MB_ICONSTOP = WIN32CON.MB_ICONHAND
WIN32CON.MB_DEFBUTTON1 = 0
WIN32CON.MB_DEFBUTTON2 = 256
WIN32CON.MB_DEFBUTTON3 = 512
WIN32CON.MB_DEFBUTTON4 = 768
WIN32CON.MB_APPLMODAL = 0
WIN32CON.MB_SYSTEMMODAL = 4096
WIN32CON.MB_TASKMODAL = 8192
WIN32CON.MB_HELP = 16384
WIN32CON.MB_NOFOCUS = 32768
WIN32CON.MB_SETFOREGROUND = 65536
WIN32CON.MB_DEFAULT_DESKTOP_ONLY = 131072
WIN32CON.MB_TOPMOST = 262144
WIN32CON.MB_RIGHT = 524288
WIN32CON.MB_RTLREADING = 1048576
WIN32CON.MB_SERVICE_NOTIFICATION = 2097152
WIN32CON.MB_TYPEMASK = 15
WIN32CON.MB_USERICON = 128
WIN32CON.MB_ICONMASK = 240
WIN32CON.MB_DEFMASK = 3840
WIN32CON.MB_MODEMASK = 12288
WIN32CON.MB_MISCMASK = 49152

```


#### 快速创建托盘

```python

from sindre.win_tools import stray
import time
def setup_fun(icon):
    icon.visible = True

    i = 0
    while icon.visible:
        # Some payload code
        print(i)
        i += 1
        time.sleep(5)
def fun0():
    print("fun0")


def fun1():
    print("fun1")


def fun2():
    print("fun2")


def fun3():
    print("fun3")


if __name__ == '__main__':
    data = [
        # 不通知执行
        {"按钮1": ["", fun1]},
        # 通知执行
        {"按钮0": ["执行fun0", fun0]},
        # 二级嵌套
        {"子菜单": [
            {"按钮2": ["", fun2]},
            {"按钮3": ["执行fun3", fun3]}
        ]
        }
    ]
    s = stray(data)
    # 自定义函数启动
    # s(setup_fun)
    s()

```


## 3. API:

::: win_tools.tools

::: win_tools.taskbar

<!-- ::: win_tools.stray -->


