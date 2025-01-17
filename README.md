<div align="center">

# Sindre-自用库

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![Conference](http://img.shields.io/badge/NeurIPS-2022-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2022-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
<p align="center">
    <br />
    <a href="https://sindreyang.github.io/sindre/"><strong><<  项目API文档 >></strong></a>
    <br />
    <br />
    <a href="https://github.com/SindreYang/sindre/releases">下载Releases</a>
    ·
    <a href="https://github.com/SindreYang/sindre/issues">报告Bug</a>
    ·
    <a href="https://github.com/SindreYang/sindre/issues">提出新特性</a>
  </p>
</div>

### 快速安装

```bash   
pip install  sindre
 ```   

### 推荐配置
1. python3.8
2. pytorch>=2.0


### 安装步骤

###### 安装依赖

```bash
# 克隆项目   
git clone https://github.com/SindreYang/sindre_page_private.git
# 安装依赖   
pip install -r requirements.txt
 ```   

###### 构建包


```bash
python setup.py bdist_wheel
```

###### 生成文档

* 设置文档范式 pycharm--->setting--->Tools--->python integrated Tools-->Google
  
```bash
mkdocs serve  -o  -c 
```

###### 构建文档

```bash
mkdocs  build
```


### 目录结构

```
├─.github                   # github的actions工作流脚本
├── docs                    # mkdocs文档目录
├── mkdocs.yml              # mkdocs 配置
├── README.md               # 项目文档
├── requirements.txt        # 项目依赖
├── setup.py                # 发布脚本
└── sindre                  # sindre库项目代码

```

### 引文

```
{
 @title={sindre},
  author={sindreyang},
  year={2024}
}
```   
