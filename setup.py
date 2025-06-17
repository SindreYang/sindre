# -*- coding: UTF-8 -*-
import os
import platform
import sys
from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
import time
import shutil

# python setup.py bdist_wheel
# pip install -e .
GFICLEE_VERSION  = os.getenv('PACKAGE_VERSION', time.strftime("%Y.%m.%d", time.localtime()))  # 默认为GFICLEE_VERSION，如果环境变量未设置
print(GFICLEE_VERSION)

    


def py2pyd(source_path: str, clear_py: bool = False):
    tmp_path = os.path.join(source_path, "tmp")
    if not os.path.exists(tmp_path):
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
                            if sys.platform.startswith('win32'):
                                if f_pyd.endswith('.pyd'):
                                    if f_pyd.split(".")[0] == module_name:
                                        # 保证只一次只处理一个文件
                                        pyd_name = f_pyd.split(".")[0] + ".pyd"
                                        old_path = os.path.join(tmp_path, f_pyd)
                                        new_path = os.path.join(root, pyd_name)
                                        try:
                                            print(f"move{old_path}-->{new_path}:")
                                            os.rename(old_path, new_path)
                                            if clear_py:
                                                print(f"clear:{file_path}")
                                                os.remove(file_path)
                                        except Exception as e:
                                            print("Exception:", e)
                            else:
                                 if f_pyd.endswith('.so'):
                                    if f_pyd.split(".")[0] == module_name:
                                        # 保证只一次只处理一个文件
                                        pyd_name = f_pyd.split(".")[0] + ".so"
                                        old_path = os.path.join(tmp_path, f_pyd)
                                        new_path = os.path.join(root, pyd_name)
                                        try:
                                            print(f"move{old_path}-->{new_path}:")
                                            os.rename(old_path, new_path)
                                            if clear_py:
                                                print(f"clear:{file_path}")
                                                os.remove(file_path)
                                        except Exception as e:
                                            print("Exception:", e)
    
                                

                        # 删除.c文件
                        c_file = file_path.replace(".py", ".c")
                        os.remove(c_file)

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)


# 将目录转换成加密目录
def sindre_py2pyd() -> list:
    """

    """
    dist_path= "dist/sindre"
    if os.path.exists(dist_path):
        shutil.rmtree(dist_path)
    shutil.copytree("sindre",dist_path)
    # 关闭cython转换
    #py2pyd(dist_path,True)
    

sindre_py2pyd()
#复制资源
shutil.copy("README.md","dist/README.md")
shutil.copy("MANIFEST.in","dist/MANIFEST.in")
#切换目录
os.chdir("dist")



""" 开启 -e 需要注释下面
#sindre_py2pyd()
# 复制资源
# shutil.copy("README.md","dist/README.md")
# shutil.copy("MANIFEST.in","dist/MANIFEST.in")
# 切换目录
#os.chdir("dist")


"""


# 获取系统版本信息
if sys.platform.startswith('win'):
    platform_tag = 'win_amd64'
elif sys.platform.startswith('linux'):
    platform_tag = 'manylinux1_x86_64'#f"linux_{platform.machine().lower()}"
# 获取 Python 版本
python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

setup(
    name='sindre',
    version=GFICLEE_VERSION,
    packages=find_packages(),
    install_requires=[
        "lmdb>=1.6.2",
        "msgpack>=1.1.0",
        'vedo>=2025.5.3',
        'loguru',
        'tqdm',
        'scipy',
        'numba',
    ],
    extras_require={"full": ["Cython", 'scikit-learn','Cython','trimesh',"open3d","pymeshlab"]},
    
    #data_files=find_files("sindre\Resources"),
    url='https://github.com/SindreYang/Sindre',
    license='GNU General Public License v3.0',
    author='SindreYang',
    description='自用脚本库',
    long_description=open('README.md', "r", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.8",
    include_package_data = True, #如果有符合MANIFEST.in的文件，会被打包
    # 工具脚本
    entry_points={
        'console_scripts': [
            'LmdbViewer = sindre.lmdb.Viewer.App:main',
        ],
    }
    
    # 开放源码，不再针对平台，版本单独编译。
    # options={
    #     'bdist_wheel': {
    #         'python_tag': python_version,
    #         'plat_name': platform_tag
    #     }
    # }
)
