from setuptools import setup, find_packages

GFICLEE_VERSION = '2024.4.11'

setup(
    name='ProjectTemplate',
    version=GFICLEE_VERSION,
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    url='https://github.com/SindreYang/ProjectTemplate',
    license='GNU General Public License v3.0',
    author='SindreYang',
    description='ProjectTemplate'
)