# -*- coding: UTF-8 -*-
import os
from setuptools import Extension, setup, find_packages
import time

# python setup.py bdist_wheel
# pip install -e . --config-settings editable_mode=compat
GFICLEE_VERSION  = os.getenv('PACKAGE_VERSION', time.strftime("%Y.%m.%d", time.localtime()))  # 默认为GFICLEE_VERSION，如果环境变量未设置
print(GFICLEE_VERSION)


setup(
    name='sindre',
    version=GFICLEE_VERSION,
    packages=find_packages(),
    install_requires=[
        "lmdb>=1.6.2",
        "msgpack>=1.1.0",
        'vedo>=2025.5.3',
        'numpy<2.0.0',
        'loguru',
        'tqdm',
        'scipy',
        'numba',

    ],
    extras_require={"full": ["Cython",'nvitop','scikit-learn', # 常用
                             "pyqt5","qdarkstyle",# pyqt5
                             "fastapi",#网络
                             "opencv-contrib-python","opencv-python","scikit-image","imgaug","matplotlib","Pillow", # 2d
                             "pymeshlab","meshlib==3.0.6.229","libigl>=2.6.1","trimesh","open3d","pymeshlab" #3d
                             ],

                    "2d":["opencv-contrib-python","opencv-python","scikit-image","imgaug","matplotlib","Pillow"],
                    "3d":["pymeshlab","meshlib==3.0.6.229","libigl>=2.6.1","trimesh","open3d","pymeshlab"],

                    },
    
    #data_files=find_files("sindre\Resources"),
    url='https://github.com/SindreYang/Sindre',
    license="LGPL-3.0-or-later",
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
            'LmdbWebViewer = sindre.lmdb.WebViewer.App:main',
            'GpuViewer = nvitop.cli:main',
            'Annotator3D = sindre.utils3d.Annotator3D.App:main'
        ],
    }

)
