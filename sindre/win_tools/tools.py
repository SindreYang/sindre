# -*- coding: UTF-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@path   ：sindre_package -> py2pyd.py
@IDE    ：PyCharm
@Author ：sindre
@Email  ：yx@mviai.com
@Date   ：2024/6/17 16:32
@Version: V0.1
@License: (C)Copyright 2021-2023 , UP3D
@Reference: 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
__author__ = 'sindre'

import glob
import os
import shutil
import socket
import subprocess
import threading
import time
import zipfile

import requests
from setuptools import Extension, setup

from collections import Counter

from Cython.Build import cythonize
from tqdm import tqdm



"""
def py2pyd(source_path: str, copy_dir: bool = False, clear_py=False):
    tmp_path = os.path.join(source_path, "tmp")
    if not os.path.exists(tmp_path):
        print(f"创建临时目录:{tmp_path}")
        os.mkdir(tmp_path)

    extensions = []
    py_files = []
    pyd_files = {}
    repeatList = []
    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(source_path):
        for file in files:
            # 判断文件名是否以 .py 结尾
            if file.endswith('.py'):
                if file == "__init__.py":
                    continue
                else:
                    # 构建文件的完整路径
                    file_path = os.path.join(root, file)
                    py_files.append(file_path)
                    repeatList.append(file)
                    new_name = file.replace(".py", ".pyd")
                    pyd_files[new_name] = os.path.join(root, new_name)

                    # 构建扩展模块名称
                    module_name = os.path.splitext(file)[0]

                    # 构建扩展模块对象
                    extension = Extension(module_name, sources=[file_path])
                    extensions.append(extension)
            else:
                print("不支持的文件类型：", file)

    # 统计列表重复项的数量并转为字典
    dict1 = dict(Counter(repeatList))

    # 列表推导式查找字典中值大于1的键值
    dict2 = {key: value for key, value in dict1.items() if value > 1}
    if len(dict2) > 0:
        print(f"存在重复文件名：{dict2} \n ")
        return False
    print("编译：", extensions)

    setup(
        ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}, force=True),
        script_args=["build_ext",  # "--inplace",
                     "--build-lib", f"{tmp_path}", "--build-temp", f"{tmp_path}", ])

    for file in glob.glob(os.path.join(source_path, "**/*"), recursive=True):
        if file.endswith((".c")):
            print("删除: ", file)
            os.remove(file)

    print("处理pyd文件")
    for file in os.listdir(tmp_path):
        if file.endswith(".pyd"):
            new_name = file.split(".")[0] + ".pyd"
            old_path = os.path.join(tmp_path, file)
            print(f"移动{file}-->{pyd_files[new_name]}：")
            try:
                os.rename(old_path, pyd_files[new_name])
            except FileExistsError:
                print(pyd_files[new_name], "已存在")
            except Exception as e:
                print("未知错误：", e)

    if clear_py:
        for f in py_files:
            if os.path.exists(f):
                os.remove(f)
                print("删除py文件：", f)

    if os.path.exists(tmp_path):
        print("删除临时目录：", tmp_path)
        shutil.rmtree(tmp_path)
    return True
"""


def py2pyd(source_path: str, clear_py: bool = False):
    """
        将目录下所有py文件编译成pyd文件。

    Args:
        source_path: 源码目录
        clear_py: 是否编译后清除py文件,注意备份。


    """
    tmp_path = os.path.join(source_path, "tmp")
    if not os.path.exists(tmp_path):
        print(f"mkdir tmp dir:{tmp_path}")
        os.mkdir(tmp_path)

    # 遍历目录下的所有文件
    for root, dirs, files in os.walk(source_path):
        if dirs != "tmp":
            for file in files:
                # 判断文件名是否以 .py 结尾
                if file.endswith('.py'):
                    if file == "__init__.py":
                        continue
                    else:
                        # 构建文件的完整路径
                        file_path = os.path.join(root, file)
                        # 构建扩展模块名称
                        module_name = os.path.splitext(file)[0]

                        # 构建扩展模块对象
                        extension = Extension(module_name, sources=[file_path])
                        print("build:", extension)

                        setup(
                            ext_modules=cythonize(extension, compiler_directives={'language_level': "3"}, force=True),
                            script_args=["build_ext",  # "--inplace",
                                         "--build-lib", f"{tmp_path}", "--build-temp", f"{tmp_path}", ])

                        # 移动pyd
                        for f_pyd in os.listdir(tmp_path):
                            if f_pyd.endswith('.pyd'):
                                if f_pyd.split(".")[0] == module_name:
                                    # 保证只一次只处理一个文件
                                    pyd_name = f_pyd.split(".")[0] + ".pyd"
                                    old_path = os.path.join(tmp_path, f_pyd)
                                    new_path = os.path.join(root, pyd_name)
                                    try:
                                        print(f"move{old_path}-->{new_path}：")
                                        os.rename(old_path, new_path)
                                        if clear_py:
                                            print(f"clear:{file_path}")
                                            os.remove(file_path)
                                    except Exception as e:
                                        print("Exception:", e)

                        # 删除.c文件
                        c_file = file_path.replace(".py", ".c")
                        print("del:", c_file)
                        os.remove(c_file)

    if os.path.exists(tmp_path):
        print("del tmp dir:", tmp_path)
        shutil.rmtree(tmp_path)


def pip_install(package_name: str = "", target_dir: str = "", requirements_path: str = ""):
    """
        模拟pip安装

    Args:
        package_name: 包名
        target_dir: 安装目录，为空，则自动安装到当前环境下
        requirements_path: requirementsTxT路径

    """
    from pip._internal import main as pip_main
    # pip_main(['install', "pyinstaller", '--target', self.tmp_path])

    if requirements_path != "":
        # 读取 requirements.txt 文件
        with open(requirements_path, 'r') as file:
            requirements = file.readlines()
        # 安装所有的whl文件到指定目录下
        for requirement in requirements:
            if target_dir != "":
                pip_main(['install', requirement.strip(), '--target', target_dir])
            else:
                pip_main(['install', requirement.strip()])

    if target_dir != "":
        pip_main(['install', package_name, '--target', target_dir])
    if package_name != "":
        pip_main(['install', package_name])


def python_installer(install_dir: str, version: str = '3.9.6'):
    """
        python自动化安装

    Notes:
        默认从 https://mirrors.huaweicloud.com/python/{version}/python-{version}-embed-amd64.zip 下载安装

    Args:
        install_dir: 安装位置
        version: 版本号


    """
    # url = f'https://www.python.org/ftp/python/{version}/python-{version}-embed-amd64.zip'
    url = f'https://mirrors.huaweicloud.com/python/{version}/python-{version}-embed-amd64.zip'
    file_path = os.path.join(install_dir, 'tmp')
    python_path = os.path.join(file_path, f"python.zip")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    if not os.path.exists(python_path):
        try:
            # 发送下载请求
            print("Python安装包开始下载！")
            with requests.get(url, stream=True) as r, open(python_path, 'wb') as f:
                total_size = int(r.headers.get('Content-Length', 0))
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, ncols=80)
                for data in r.iter_content(chunk_size=8192):
                    progress_bar.update(len(data))
                    f.write(data)
                progress_bar.close()
            print("Python安装包下载完成！")
        except Exception as e:
            print("下载过程出现错误:", str(e))
            return 0

    try:
        # 执行安装命令
        # install_command = [
        #     python_path,
        #     '/quiet',
        #     'InstallAllUsers=0',
        #     'DefaultJustForMeTargetDir=' + install_dir,
        #     'AssociateFiles=0',
        #     'CompileAll=1',
        #     'AppendPath=0',
        #     'Shortcuts=0',
        #     'Include_doc=0',
        #     'Include_dev=0',
        #     'Include_exe=0',
        #     'Include_launcher=0',
        #     'Include_lib=1',
        #     'Include_tcltk=0',
        #     'Include_pip=1',
        #     'Include_test=0',
        #     'Include_tools=0',
        # ]
        # uninstall_command = [
        #     python_path,
        #     '/quiet',
        #     '/uninstall',
        #     'DefaultJustForMeTargetDir=' + install_dir,
        #     ]
        # subprocess.run(uninstall_command, check=True,capture_output=True)
        # print("Python开始安装！")
        # result = subprocess.run(install_command, check=True,capture_output=True)
        # print(result.stdout.decode())
        # print("Python安装完成！")
        # shutil.rmtree(file_path)

        print("Python开始安装！")
        with zipfile.ZipFile(python_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
    except subprocess.CalledProcessError as e:
        print("安装过程出现错误:", str(e))


def exe2nsis(work_dir: str,
             files_to_compress: list,
             exe_name: str,
             appname: str = "AI",
             version: str = "1.0.0.0",
             author: str = "SindreYang",
             license: str = "",
             icon_old: str = ""):
    """
        将exe进行nsis封装成安装程序；

    Notes:
        files_to_compress =[f"{self.work_dir}/{i}" for i in  ["app", "py", "third", "app.exe", "app.py", "requirements.txt"]]

    Args:
        work_dir: 生成的路径
        files_to_compress: 需要转换的文件夹/文件列表
        exe_name: 指定主运行程序，快捷方式也是用此程序生成
        appname: 产品名
        version: 版本号--必须为 X.X.X.X
        author: 作者
        license: licence.txt协议路径
        icon_old: 图标


    """
    # 获取当前脚本的绝对路径
    exe_7z_path = os.path.abspath("./bin/7z/7z.exe")
    exe_nsis_path = os.path.abspath("./bin/NSIS/makensis.exe")
    config_path = os.path.abspath("./bin/config")
    print(exe_7z_path)
    # 压缩app目录
    app_7z_path = f"{work_dir}/app.7z"
    if os.path.exists(app_7z_path):
        print(f"已存在{app_7z_path},跳过压缩步骤")
    else:
        print(f"不存在{app_7z_path},开始压缩步骤")
        subprocess.run([f"{exe_7z_path}", "a", app_7z_path] + files_to_compress)
    # 替换文件
    nsis_code = f"""
# ====================== 自定义宏 产品信息==============================
!define PRODUCT_NAME           		"{appname}"
!define PRODUCT_PATHNAME           	"{appname}"     #安装卸载项用到的KEY
!define INSTALL_APPEND_PATH         "{appname}"     #安装路径追加的名称 
!define INSTALL_DEFALT_SETUPPATH    ""       #默认生成的安装路径 
!define EXE_NAME               		"{exe_name}" # 指定主运行程序，快捷方式也是用此程序生成
!define PRODUCT_VERSION        		"{version}"
!define PRODUCT_PUBLISHER      		"{author}"
!define PRODUCT_LEGAL          		"${{PRODUCT_PUBLISHER}} Copyright（c）2023"
!define INSTALL_OUTPUT_NAME    		"{appname}_V{version}.exe"

# ====================== 自定义宏 安装信息==============================
!define INSTALL_7Z_PATH 	   		"{work_dir}\\app.7z"
!define INSTALL_7Z_NAME 	   		"app.7z"
!define INSTALL_RES_PATH       		"skin.zip"
!define INSTALL_LICENCE_FILENAME    "{os.path.join(config_path, "license.txt") if license == "" else license}"
!define INSTALL_ICO 				"{os.path.join(config_path, "logo.ico") if icon_old == "" else icon_old}"


!include "{os.path.join(config_path, "ui.nsh")}"

# ==================== NSIS属性 ================================

# 针对Vista和win7 的UAC进行权限请求.
# RequestExecutionLevel none|user|highest|admin
RequestExecutionLevel admin

#SetCompressor zlib

; 安装包名字.
Name "${{PRODUCT_NAME}}"

# 安装程序文件名.

OutFile "{work_dir}\\{appname}_V{version}.exe"

InstallDir "1"

# 安装和卸载程序图标
Icon              "${{INSTALL_ICO}}"
UninstallIcon     "uninst.ico"

        
        """

    # 执行封装命令
    nsis_path = os.path.join(config_path, "output.nsi")
    with open(nsis_path, "w", encoding="gb2312") as file:
        file.write(nsis_code)
    print([f"{exe_nsis_path}", nsis_path])
    try:  # 生成exe
        subprocess.run([f"{exe_nsis_path}", nsis_path])
    except Exception as e:
        print(e)
        print([f"{exe_nsis_path}", nsis_path])
    # 清理文件
    os.remove(nsis_path)
    if os.path.exists(os.path.join(work_dir, f"{appname}_V{version}.exe")):
        os.remove(f"{work_dir}/app.7z")

        return True
    else:
        return False


def is_service_exists(service_name: str) -> bool:
    """
        使用sc query命令来查询服务

    Args:
        service_name: 服务名

    Returns:
        返回是否存在服务

    """

    # 使用sc query命令来查询服务
    command = ['sc', 'query', service_name]
    try:
        # 运行命令并获取输出
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # 检查命令是否成功执行（返回码为0）
        if result.returncode == 0:
            # 在标准输出中查找服务名，确认服务存在
            if service_name in result.stdout:
                return True
    except Exception as e:
        # 处理其他可能的异常
        return False


def check_port(port: int) -> bool:
    """
        检测win端口是否被占用

    Args:
        port: 端口号

    Returns:
        是否被占用

    """
    cmd = f"netstat -ano | findstr :{port} | findstr LISTENING"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return len(result.stdout.strip()) > 0


def download_url_file(url: str, package_path: str = "test.zip") -> bool:
    """
        下载网络文件

    Args:
        url: 文件下载地址
        package_path:  保存路径

    Returns:
        下载是否成功

    """

    try:
        # 发送下载请求
        with requests.get(url, stream=True) as r, open(package_path, 'wb') as f:
            total_size = int(r.headers.get('Content-Length', 0))
            for data in r.iter_content(chunk_size=8192):
                f.write(data)
            return True
    except Exception as e:
        print(e)
        return False


def zip_extract(zip_path: str, install_dir: str) -> bool:
    """
         将zip文件解压
    Args:
        zip_path: zip文件路径
        install_dir: 解压目录

    Returns:
        解压是否成功

    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(install_dir)
            return True
    except Exception as e:
        print(e)
        return False


def kill_process_using_port(server_port: int) -> bool:
    """
        请求管理员权限，并强制释放端口

    Args:
        server_port: 端口号

    Returns:
        端口是否成功释放

    """

    kill_code = rf"""
@echo off  
setlocal enabledelayedexpansion  
  
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"  
if '%errorlevel%' NEQ '0' (  
    goto UACPrompt  
) else (  
    goto gotAdmin  
)  
  
:UACPrompt  
echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"  
echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"  
"%temp%\getadmin.vbs"  
exit /B  
  
:gotAdmin  
if exist "%temp%\getadmin.vbs" (  
    del "%temp%\getadmin.vbs"  
)  
pushd "%CD%"  
CD /D "%~dp0"  
  
set PORT={server_port}
set /a TIMEOUT=2  
  
echo Checking for processes running on port %PORT%...  
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%PORT%') do (  
    set PID=%%a  
    goto foundProcess  
)  
echo No process found running on port %PORT%.  
goto endScript  
  
:foundProcess  
echo Found process with PID: !PID!  
taskkill /F /PID !PID!  
echo Process on port %PORT% has been killed.  

  
:endScript  
timeout /t %TIMEOUT% >nul  
echo Script will exit after %TIMEOUT% seconds...  
endlocal  
exit
        """
    with open("kill.bat", 'w') as f:
        f.write(kill_code)
    subprocess.Popen("kill.bat")  # 不显示cmd窗口
    time.sleep(5)
    if os.path.isfile("kill.bat"):
        os.remove("kill.bat")
    if check_port(server_port):
        return False
    return True


class tcp_mapping_qt(threading.Thread):
    """
    TCP 传输线程
    """

    def __init__(self, conn_receiver, conn_sender):
        super(tcp_mapping_qt, self).__init__()
        self.conn_receiver, self.conn_sender = conn_receiver, conn_sender

    def run(self):
        while True:
            try:
                # 接收数据缓存大小
                print("接收数据缓存大小")
                data = self.conn_receiver.recv(32768)
                if not data:
                    break
                print("接收数据缓存大小", len(data))
            except Exception as e:
                print("[-] 关闭: 映射请求已关闭.", e)
                break

            try:
                print("sendall", len(data))
                self.conn_sender.sendall(data)
                print("sendall")
            except Exception as e:
                print("[-] 错误: 发送数据时出错.", e)
                break
            if self.conn_receiver and self.conn_sender:
                print("[+] 映射请求: {} ---> 传输到: {} ---> {} bytes".format(self.conn_receiver.getpeername(),
                                                                              self.conn_sender.getpeername(),
                                                                              len(data)))
        self.conn_receiver.close()
        self.conn_sender.close()


class ip_bind(threading.Thread):
    """
        实现本地0.0.0.0：8000 <--> 远程端口 内网穿透
    """

    def __init__(self):
        super(ip_bind, self).__init__()
        self.remote_conn = None
        self.local_server = None
        self.remote_ip = "192.168.1.53"  # 对端地址
        self.remote_port = 8000  # 对端端口
        self.local_ip = "0.0.0.0"  # 本机地址
        self.local_port = 8000  # 本机端口
        self.stat = True  # 线程开关
        self.threads = []  # 线程列表

    def run(self):
        self.local_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 允许地址重复使用
        self.local_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.local_server.bind((self.local_ip, self.local_port))
        self.local_server.listen(5)

        print("[*] 本地端口监听 {}:{}".format(self.local_ip, self.local_port))
        while self.stat and self.local_server is not None:
            try:
                (local_conn, local_addr) = self.local_server.accept()
                # 远程端口
                self.remote_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.remote_conn.connect((self.remote_ip, self.remote_port))
                lr = tcp_mapping_qt(local_conn, self.remote_conn)
                rl = tcp_mapping_qt(self.remote_conn, local_conn)
                lr.start()
                rl.start()
                self.threads.append(lr)
                self.threads.append(rl)
            except Exception as e:
                print(e)
                break

        self.local_server.close()

    def on_mapping_thread_finished(self):
        # 如果线程完成，则移除
        self.threads.remove(threading.current_thread())

    def close(self):
        self.stat = False
        for t in self.threads:
            if t.isRuning():
                t.join(2)
        if self.remote_conn is not None:
            self.remote_conn.close()
        if self.local_server is not None:
            self.local_server.close()

    def set_ip(self, remote_ip: str, remote_port: str):
        """
            设置远程ip及端口

        Args:
            remote_ip: 远程ip
            remote_port: 远程端口

        Returns:

        """

        self.remote_ip = remote_ip  # 对端地址
        self.remote_port = int(remote_port)  # 对端端口
        print(f"[*] 端口映射 {self.local_ip}:{self.local_port}--->{self.remote_ip}:{self.remote_port}")


if __name__ == '__main__':
    # py2pyd(r"C:\Users\sindre\Downloads\55555")
    # exe2nsis(work_dir=r"C:\Users\sindre\Desktop\test",
    #          files_to_compress=[f"C:/Users/sindre/Desktop/test/t/{i}" for i in  ["app", "app.exe", "app.py"]],
    #          exe_name="app.exe")

    files = [f"C:/Users/sindre/Desktop/test/{i}" for i in ["AI",
                                                           "AI_Services.exe",
                                                           "kill.bat",
                                                           "python38.dll",
                                                           "7z.dll",
                                                           "7z.exe",
                                                           "curl.exe"]]
    exe2nsis(work_dir=r"C:\Users\sindre\Desktop\test",
             files_to_compress=files,
             exe_name="AI_Services.exe")
