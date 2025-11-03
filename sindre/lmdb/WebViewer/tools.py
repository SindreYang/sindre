import os
import json
import numpy as np
import cv2
from PIL import Image
import base64
import io
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from typing import List, Dict, Optional, Tuple
from sindre.utils3d.algorithm import labels2colors
from sindre.lmdb import Reader, Writer
from sindre.lmdb.WebViewer.config import DBInfo, DBContent


def get_file_size(path: str) -> int:
    """获取文件大小(KB)"""
    return int(os.path.getsize(path) / 1024)

def get_modified_time(path: str) -> float:
    """获取文件修改时间"""
    return os.path.getmtime(path)

def scan_directories(root: str) -> List[DBInfo]:
    """扫描目录下所有数据库文件"""
    db_list = []

    for i, file in enumerate(os.listdir(root)):
        path = os.path.join(root, file)
        if not os.path.isfile(path):
            continue

        ext = os.path.splitext(file)[1].lower()
        if ext not in ["db","lmdb","yx"]:
            continue

        file_size = get_file_size(path)
        ori_file_name =file
        file_name = os.path.splitext(file)[0]
        file_time = get_modified_time(path)

        try:
            with Reader(os.path.abspath(path)) as reader:
                # 尝试从元数据库读标签，数据类型
                data_tag = reader.get_meta("tag") or ["unknown"]
                if isinstance(data_tag, str):
                    data_tag = [data_tag]

                data_type = reader.get_meta("type") or "unknown"
                if data_type not in ["image", "point_cloud", "mesh"]:
                    data_type = "unknown"

                # 尝试获取数据信息
                if len(reader) > 0:
                    sample = reader[0]
                    key_info = []
                    shape_info = []
                    dtype_info = []
                    for key in sample.keys():
                        key_info.append(key)
                        dtype_info.append(str(sample[key].dtype))
                        shape_info.append(str(sample[key].shape))
                else:
                    key_info = []
                    shape_info = []
                    dtype_info = []

                db_info = DBInfo(
                    id=str(i),  # 使用索引作为ID
                    name=file_name,
                    ori_name=ori_file_name,
                    length =len(reader),
                    path=os.path.abspath(path).replace("\\",'/'),
                    type=data_type,
                    tags=data_tag,
                    size=file_size,
                    modified_time=file_time,
                    data_keys=key_info,
                    data_shape=shape_info,
                    data_dtype=dtype_info,
                    preview_available=True if len(reader) > 0 else False
                )
                db_list.append(db_info)
        except Exception as e:
            import traceback
            print(f"处理文件 {path} 时出错: {e}")
            print(traceback.format_exc())
            continue
    return db_list

def read_database_content(db_info: DBInfo, selected_keys: List[str]) -> DBContent:
    """读取数据库内容，根据类型调用不同的读取函数"""
    try:
        with Reader(db_info.path) as reader:
            if len(reader) == 0:
                return DBContent(info=db_info)

            # 读取第一个样本作为预览
            sample = reader[0]

            # 过滤选中的键
            filtered_data = {}
            if selected_keys:
                for key in selected_keys:
                    if key in sample:
                        filtered_data[key] = sample[key].tolist()
            else:
                for key in sample:
                    filtered_data[key] = sample[key].tolist()

            return DBContent(
                info=db_info,
                raw_data=filtered_data,
                preview_available=db_info.preview_available
            )
    except Exception as e:
        print(f"读取数据库内容失败: {e}")
        return DBContent(info=db_info)

def get_data_preview(db_path: str, db_type: str, specific_info: Dict,
                     preview_type: str, selected_key: str, data_index: int) -> Dict:
    """生成数据预览"""
    try:
        with Reader(db_path) as reader:
            if data_index < 0 or data_index >= len(reader):
                return {"error": "索引超出数据库范围"}

            data = reader[data_index]

            # 检查选中的键是否存在
            if selected_key:
                if selected_key not in data :
                    return {"error": f"数据库中不存在键: {selected_key}"}

            # 根据数据类型处理
            if db_type == "image":
                return process_image_preview(data, specific_info, preview_type, selected_key)
            elif db_type == "point_cloud":
                return process_point_cloud_preview(data, specific_info, preview_type, selected_key)
            elif db_type == "mesh":
                return process_mesh_preview(data, specific_info, preview_type, selected_key)
            else:
                return {"error": f"不支持的数据类型: {db_type}"}
    except Exception as e:
        return {"error": f"生成预览失败: {str(e)}"}

def process_image_preview(data: Dict, specific_info: Dict, preview_type: str, selected_key: str) -> Dict:
    """处理图像预览"""
    # 获取图像数据
    if "image" not in specific_info or specific_info["image"] not in data:
        return {"error": "未找到有效的图像数据键"}

    image = data[specific_info["image"]]
    if len(image.shape) != 3 or image.shape[-1] not in [3, 4]:
        return {"error": "图像数据格式无效"}

    # 确保图像是uint8类型
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

    # 根据预览类型绘制标注
    if preview_type == "keypoint":
        kps_list = []
        for kp in data[selected_key]:
            if len(kp) >= 2:
                kps_list.append(Keypoint(x=kp[0], y=kp[1]))

        kps = KeypointsOnImage(kps_list, shape=image.shape)
        image = kps.draw_on_image(image, color=(0, 255, 0))

    elif preview_type == "bbox":
        bbs_list = []
        for bbox in data[selected_key]:
            if len(bbox) >= 4:
                bbs_list.append(BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]))

        bbs = BoundingBoxesOnImage(bbs_list, shape=image.shape)
        image = bbs.draw_on_image(image, color=(255, 0, 0), alpha=0.5)

    elif preview_type == "segmentation":
        segmap = data[selected_key]
        H, W = image.shape[:2]
        if segmap.shape != (H, W):
            # 调整分割图大小以匹配图像
            segmap = cv2.resize(segmap, (W, H), interpolation=cv2.INTER_NEAREST)

        segmap = SegmentationMapsOnImage(segmap, shape=image.shape)
        image = segmap.draw_on_image(image, alpha=0.3)[0]
    else:
        pass

    # 转换为base64
    img_pil = Image.fromarray(image)
    buffer = io.BytesIO()
    img_pil.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return {"data": img_str, "format": "png"}

def process_point_cloud_preview(data: Dict, specific_info: Dict, preview_type: str, selected_key: str) -> Dict:
    """处理点云预览"""
    if "vertices" not in specific_info or specific_info["vertices"] not in data:
        return {"error": "未找到有效的顶点数据键"}

    vertices = data[specific_info["vertices"]]
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return {"error": "顶点数据格式无效"}

    # 处理颜色
    colors = None
    if preview_type == "vertex_labels":
        # 标签转颜色
        labels = data[selected_key]
        if len(labels) != len(vertices):
            return {"error": "标签数量与顶点数量不匹配"}
        colors = labels2colors(labels)
    elif preview_type == "vertex_colors":
        # 直接使用颜色数据
        colors = data[selected_key]
        if colors.ndim != 2 or colors.shape[1] not in [3, 4]:
            return {"error": "颜色数据格式无效"}


    # 确保颜色在0-1范围内
    if colors is not None and colors.max() > 1:
        colors = colors / 255.0

    return {
        "vertices": vertices.tolist(),
        "colors": colors[...,:3].tolist() if colors is not None else None,
        "type": "point_cloud"
    }

def process_mesh_preview(data: Dict, specific_info: Dict, preview_type: str, selected_key: str) -> Dict:
    """处理网格预览"""
    if "vertices" not in specific_info or specific_info["vertices"] not in data:
        return {"error": "未找到有效的顶点数据键"}
    if "faces" not in specific_info or specific_info["faces"] not in data:
        return {"error": "未找到有效的面数据键"}

    vertices = data[specific_info["vertices"]]
    faces = data[specific_info["faces"]]

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return {"error": "顶点数据格式无效"}
    if faces.ndim != 2 or faces.shape[1] not in [3, 4]:  # 支持三角形和四边形面
        return {"error": "面数据格式无效"}

    # 处理颜色
    colors = None
    color_type = None

    if preview_type in ["vertex_labels", "vertex_colors"]:
        color_type = "vertex"
        if preview_type == "vertex_labels":
            # 标签转颜色
            labels = data[selected_key]
            if len(labels) != len(vertices):
                return {"error": "标签数量与顶点数量不匹配"}
            colors = labels2colors(labels)
        else:
            # 直接使用颜色数据
            colors = data[selected_key]
            if colors.ndim != 2 or colors.shape[1] not in [3, 4]:
                return {"error": "顶点颜色数据格式无效"}


    elif preview_type in ["faces_labels", "faces_colors"]:
        color_type = "face"
        if preview_type == "faces_labels":
            # 标签转颜色
            labels = data[selected_key]
            if len(labels) != len(faces):
                return {"error": "标签数量与面数量不匹配"}
            colors = labels2colors(labels)
        else:
            # 直接使用颜色数据
            colors = data[selected_key]
            if colors.ndim != 2 or colors.shape[1] not in [3, 4]:
                return {"error": "面颜色数据格式无效"}


    # 确保颜色在0-1范围内
    if colors is not None and colors.max() > 1:
        colors = colors / 255.0
    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
        "colors": colors[...,:3].tolist() if colors is not None else None,
        "color_type": color_type,
        "type": "mesh"
    }


def set_db_info(db_info, db_type, db_tags):
    import math
    path = db_info.path
    size = math.ceil(db_info.size/1024)

    # 设置数据库元信息
    db = Writer(path,size)
    db.put_meta("tag", db_tags)
    db.put_meta("type", db_type)
    db.close()
