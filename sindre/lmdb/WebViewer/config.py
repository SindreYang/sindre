from pydantic import BaseModel
from typing import List, Dict, Optional

class Annotation(BaseModel):
    type: str  # keypoint, bbox, segmentation
    data: Dict


# 数据模型 - 用于接收数据库信息更新
class DBInfoUpdate(BaseModel):
    type: str
    tags: List[str]

class DBContent(BaseModel):
    info: "DBInfo"
    annotations: Optional[List[Annotation]] = None
    preview_data: Optional[Dict] = None
    raw_data: Optional[Dict] = None  # 原始数据，供用户选择键值

class DBInfo(BaseModel):
    id: str
    name: str
    ori_name:str #存最原始文件名,方便下载
    path: str
    size: int  # 数据库存储大小 KB
    type: str  # point_cloud, mesh, image
    length:int # 数据库大小
    modified_time: float  # 数据库的修改时间
    tags: List[str]  # 数据库的标签
    data_keys: List[str]  # 数据库包含的键
    data_shape: List[str]  # 对应键的形状
    data_dtype: List[str]  # 对应键的类型
    preview_available: bool

