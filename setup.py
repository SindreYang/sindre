import os

from setuptools import setup, find_packages
import time
import shutil
from sindre.win_tools.tools import py2pyd

# python setup.py bdist_wheel
GFICLEE_VERSION = time.strftime("%Y.%m.%d", time.localtime())


# 将目录转换成加密目录
def sindre_py2pyd() -> list:
    """

    """
    dist_path= "dist/sindre"
    if os.path.exists(dist_path):
        shutil.rmtree(dist_path)
    shutil.copytree("sindre",dist_path)
    py2pyd(dist_path,True)
    
sindre_py2pyd()

# 复制资源
shutil.copy("README.md","dist/README.md")
shutil.copy("MANIFEST.in","dist/MANIFEST.in")
# 切换目录
os.chdir("dist")



setup(
    name='sindre',
    version=GFICLEE_VERSION,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "lmdb",
        "msgpack",
        'pywin32',
        'Cython',
        'tqdm',
        'scikit-learn',
        'vedo',
    ],
    #data_files=find_files("sindre\Resources"),
    url='https://github.com/SindreYang/sindre',
    license='GNU General Public License v3.0',
    author='SindreYang',
    description='自用脚本库',
    long_description=open('README.md', "r", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.4",
    include_package_data = True #如果有符合MANIFEST.in的文件，会被打包
)
