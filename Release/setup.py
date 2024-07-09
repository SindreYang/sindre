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


setup(
    name='sindre',
    version=GFICLEE_VERSION,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    #data_files=find_files("sindre\Resources"),
    url='https://github.com/SindreYang/sindre',
    license='GNU General Public License v3.0',
    author='SindreYang',
    description='自用脚本库',
    long_description=open('README.md', "r", encoding="utf-8").read(),
    long_description_content_type='text/markdown',
    python_requires=">=3.4"
)
