import os

from setuptools import setup, find_packages
import time

# python setup.py bdist_wheel
GFICLEE_VERSION = time.strftime("%Y.%m.%d", time.localtime())

# 清理目录
time.sleep(3)
try:
    os.remove("./sindre.egg-info")
    os.remove("./build")
    os.remove("./dist")
except:
    pass


# 添加资源目录
def find_files(path: str = '.') -> list:
    """
    查找路径下所有文件
    Args:
        path: 路径

    Returns:
        所有文件组成的list

    """
    dir_files = []
    for root, dirs, files in os.walk(path):
        file_list = [os.path.join(root, file) for file in files]
        if file_list:  # 仅添加包含文件的目录
            dir_files.append((root, file_list))
    print(dir_files)
    return dir_files


setup(
    name='sindre',
    version=GFICLEE_VERSION,
    packages=find_packages(),
    install_requires=[

    ],
    #data_files=find_files("sindre\Resources"),
    url='https://github.com/SindreYang/sindre',
    license='GNU General Public License v3.0',
    author='SindreYang',
    description='自用脚本库',
    long_description=open('README.md', "r", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.4"
)
