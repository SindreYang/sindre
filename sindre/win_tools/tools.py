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
@History:
- 2024/6/17 :


(一)本代码的质量保证期（简称“质保期”）为上线内 1个月，质保期内乙方对所代码实行包修改服务。
(二)本代码提供三包服务（包阅读、包编译、包运行）不包熟
(三)本代码所有解释权归权归神兽所有，禁止未开光盲目上线
(四)请严格按照保养手册对代码进行保养，本代码特点：
      i. 运行在风电、水电的机器上
     ii. 机器机头朝东，比较喜欢太阳的照射
    iii. 集成此代码的人员，应拒绝黄赌毒，容易诱发本代码性能越来越弱
声明：未履行将视为自主放弃质保期，本人不承担对此产生的一切法律后果
如有问题，热线: 114

                                                    __----~~~~~~~~~~~------___
                                   .  .   ~~//====......          __--~ ~~         江城子 . 程序员之歌
                   -.            \_|//     |||\\  ~~~~~~::::... /~
                ___-==_       _-~o~  \/    |||  \\            _/~~-           十年生死两茫茫，写程序，到天亮。
        __---~~~.==~||\=_    -_--~/_-~|-   |\\   \\        _/~                    千行代码，Bug何处藏。
    _-~~     .=~    |  \\-_    '-~7  /-   /  ||    \      /                   纵使上线又怎样，朝令改，夕断肠。
  .~       .~       |   \\ -_    /  /-   /   ||      \   /
 /  ____  /         |     \\ ~-_/  /|- _/   .||       \ /                     领导每天新想法，天天改，日日忙。
 |~~    ~~|--~~~~--_ \     ~==-/   | \~--===~~        .\                          相顾无言，惟有泪千行。
          '         ~-|      /|    |-~\~~       __--~~                        每晚灯火阑珊处，夜难寐，加班狂。
                      |-~~-_/ |    |   ~\_   _-~            /\
                           /  \     \__   \/~                \__
                       _--~ _/ | .-~~____--~-/                  ~~==.
                      ((->/~   '.|||' -_|    ~~-/ ,              . _||
                                 -_     ~\      ~~---l__i__i__i--~~_/
                                 _-~-__   ~)  \--______________--~~
                               //.-~~~-~_--~- |-------~~~~~~~~
                                      //.-~~~--\

                              神兽保佑                                 永无BUG
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
__author__ = 'sindre'

import glob
import os
import shutil
import subprocess
import zipfile

from setuptools import Extension, setup
from Cython.Build import cythonize
from collections import Counter

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
        clear_py: 是否编译后清除py文件，注意备份。


    """
    tmp_path = os.path.join(source_path, "tmp")
    if not os.path.exists(tmp_path):
        print(f"创建临时目录:{tmp_path}")
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
                        print("编译：", extension)

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
                                        print(f"移动{old_path}-->{new_path}：")
                                        os.rename(old_path, new_path)
                                        if clear_py:
                                            print(f"清除{file_path}")
                                            os.remove(file_path)
                                    except Exception as e:
                                        print("未知错误：", e)

                        # 删除.c文件
                        c_file = file_path.replace(".py", ".c")
                        print("删除: ", c_file)
                        os.remove(c_file)

    if os.path.exists(tmp_path):
        print("删除临时目录：", tmp_path)
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
             exe_name:str,
             appname: str = "AI",
             version: str = "1.0.0.0",
             author: str = "SindreYang",
             license: str = "",
             icon_old: str=""):
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
    subprocess.run([f"{exe_7z_path}", "a", f"{work_dir}/app.7z"] + files_to_compress)
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
!define INSTALL_LICENCE_FILENAME    "{os.path.join(config_path, "license.txt") if license == "" else licence}"
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


if __name__ == '__main__':
    #py2pyd(r"C:\Users\sindre\Downloads\55555")
    exe2nsis(work_dir=r"C:\Users\sindre\Downloads\55555",
             files_to_compress=[f"C:/Users/sindre/Downloads/55555/{i}" for i in  ["app", "app.exe", "app.py"]],
             exe_name="app.exe")

