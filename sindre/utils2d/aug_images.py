# 由imgaug库提供支持
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox,BoundingBoxesOnImage
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
# 由torchvision库提供支持
from torchvision.transforms.v2 import Transform,CutMix,MixUp,RandomChoice
from torchvision import tv_tensors
import torchvision
import os
import torch
import numpy as np
from typing import Optional, Union, List, Tuple
from torch.utils.data import default_collate
import random



def resize_by_iaa(image, out_shape, bboxes=None, keypoints=None, seg=None, interpolation=None):
    """使用imgaug库调整图像大小及相关的标注信息。

    该函数可以同时调整图像大小以及对应的边界框、关键点和分割掩码，
    保持所有标注与调整后的图像对齐。

    Args:
        image (numpy.ndarray): 输入图像，形状为(H,W,C)或(H,W)
        out_shape (tuple): 输出图像的形状，格式为(高度, 宽度)
        bboxes (list, optional): 边界框列表，每个边界框格式为[x1,y1,x2,y2]或[x1,y1,x2,y2,label]
        keypoints (list, optional): 关键点列表格式为[x,y]
        seg (numpy.ndarray, optional): 分割掩码
        interpolation (str, optional): 插值方法，如'cubic','linear','nearest'等

    Returns:
        dict: 包含调整后的图像和标注信息的字典，包含以下可能的键:
            - 'image': 调整后的图像
            - 'bboxes': 调整后的边界框（如果提供了bboxes）
            - 'keypoints': 调整后的关键点（如果提供了keypoints）
            - 'seg': 调整后的分割掩码（如果提供了seg）

    Raises:
        ImportError: 当没有安装imgaug库时抛出
    """
    try:
        import imgaug as ia
        from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    except ImportError:
        raise ImportError("imgaug库未安装，请使用: pip install imgaug")

    # 调整图像大小
    resized_image = ia.imresize_single_image(image, out_shape, interpolation=interpolation)

    # 初始化返回字典
    result = {'image': resized_image}

    # 处理边界框
    if bboxes is not None:
        if len(bboxes) > 0 and len(bboxes[0]) == 5:
            iaa_boxes = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3], label=box[4])
                         for box in bboxes]
        else:
            iaa_boxes = [BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3])
                         for box in bboxes]

        bboxes_on_image = BoundingBoxesOnImage(iaa_boxes, shape=image.shape)
        result['bboxes'] = bboxes_on_image.on(resized_image)

    # 处理关键点
    if keypoints is not None:
        from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
        ia_keypoints = [Keypoint(x=kp[0], y=kp[1]) for kp in keypoints]
        keypoints_on_image = KeypointsOnImage(ia_keypoints, shape=image.shape)
        result['keypoints'] = keypoints_on_image.on(resized_image)

    # 处理分割掩码
    if seg is not None:
        # 分割掩码通常使用最近邻插值以保持类别标签的准确性
        seg_interpolation = 'nearest' if interpolation is None else interpolation
        result['seg'] = ia.imresize_single_image(seg, out_shape, interpolation=seg_interpolation)

    return result

def yolo_normalization(image,boxes):
    H,W = image.shape[:2]

    #image:  [3, H, W], 归一化到0-1;  res_anns: [nL, 6],归一化到0-1;  每行格式: [占位符, 类别ID, 中心x, 中心y, 宽, 高])
    image       = np.transpose(np.array(image, dtype=np.float32)/255, (2, 0, 1))
    boxes = np.array(boxes, dtype=np.float32)
    # 归一化边界框坐标
    boxes[:, [0, 2]] = boxes[:, [0, 2]] / W  # x坐标归一化
    boxes[:, [1, 3]] = boxes[:, [1, 3]] / H  # y坐标归一化

    # 从 [x1, y1, x2, y2] 转换为 [cx, cy, w, h]
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    centers_x = boxes[:, 0] + widths / 2
    centers_y = boxes[:, 1] + heights / 2

    # 更新boxes为 [index,class, cx, cy, w, h] 格式
    converted_boxes =  np.zeros((len(boxes), 6))
    converted_boxes[:, 0] = 0  # 后期用于找图片索引
    converted_boxes[:, 1] = boxes[:, 4]  # 类别ID
    converted_boxes[:, 2] = centers_x    # 中心x
    converted_boxes[:, 3] = centers_y    # 中心y
    converted_boxes[:, 4] = widths       # 宽
    converted_boxes[:, 5] = heights      # 高
    boxes = converted_boxes

    return image, boxes

def box_xyxy2cxcywh(x1, y1, x2, y2):
    """
    边界框：
    左上角(x1,y1) + 右下角(x2,y2) → 中心(cx,cy) + 宽高(w,h)

    Args:
        x1, y1: 左上角坐标（int/float 或 numpy数组）
        x2, y2: 右下角坐标（int/float 或 numpy数组）
    Returns:
        cx, cy: 中心坐标
        w, h: 宽和高
    """
    w = x2 - x1  # 宽 = 右下角x - 左上角x
    h = y2 - y1  # 高 = 右下角y - 左上角y
    cx = x1 + w / 2  # 中心x = 左上角x + 宽/2
    cy = y1 + h / 2  # 中心y = 左上角y + 高/2
    return cx, cy, w, h
def box_cxcywh2xyxy(cx, cy, w, h):
    """
    边界框：
    中心(cx,cy) + 宽高(w,h) → 左上角(x1,y1) + 右下角(x2,y2)

    Args:
        cx, cy: 中心坐标（int/float 或 numpy数组）
        w, h: 宽和高（int/float 或 numpy数组）
    Returns:
        x1, y1: 左上角坐标
        x2, y2: 右下角坐标
    """
    x1 = cx - w / 2  # 左上角x = 中心x - 宽/2
    y1 = cy - h / 2  # 左上角y = 中心y - 高/2
    x2 = cx + w / 2  # 右下角x = 中心x + 宽/2
    y2 = cy + h / 2  # 右下角y = 中心y + 高/2
    return x1, y1, x2, y2



def cutmix_augmentation(img1, img2, bboxes1=None, bboxes2=None, keypoints1=None, keypoints2=None, seg1=None, seg2=None):
    """
    执行CutMix数据增强，将两张相同大小的numpy图像混合成一张图像。

    Args:
        img1, img2: 输入图像，必须是相同大小的numpy数组，形状为[H, W, C]
        bboxes1, bboxes2: 边界框标注列表，格式为[[x1,y1,x2,y2,cls], ...]
        keypoints1, keypoints2: 关键点标注列表，格式为[[[x,y,v], ...], ...]
        seg1, seg2: 分割掩码，必须是相同大小的numpy数组，形状为[H, W]

    Returns:
        tuple: 混合后的图像和标注信息
    """
    # 检查图像尺寸是否相同
    if img1.shape != img2.shape:
        raise ValueError("输入的两张图像尺寸必须相同")

    # 获取图像尺寸
    out_h, out_w = img1.shape[0], img1.shape[1]

    # 随机生成裁剪区域的参数
    lam = np.random.beta(1.0, 1.0)  # CutMix混合比例
    cut_ratio = np.sqrt(1. - lam)  # 裁剪区域的比例

    cut_w = int(out_w * cut_ratio)
    cut_h = int(out_h * cut_ratio)

    # 随机生成裁剪中心点
    cx = np.random.randint(0, out_w)
    cy = np.random.randint(0, out_h)

    # 计算裁剪区域的边界
    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(out_w, cx + cut_w // 2)
    y2 = min(out_h, cy + cut_h // 2)

    # 创建混合图像
    mixed_img = img1.copy()
    mixed_img[y1:y2, x1:x2] = img2[y1:y2, x1:x2]

    # 处理标注信息
    mixed_bboxes = []
    mixed_keypoints = []
    mixed_seg = None

    # 处理第一张图像的标注（完整保留）
    if bboxes1 is not None:
        mixed_bboxes.extend(bboxes1)

    if keypoints1 is not None:
        mixed_keypoints.extend(keypoints1)

    # 处理分割掩码1
    if seg1 is not None:
        mixed_seg = seg1.copy()

    # 处理第二张图像的标注（只保留裁剪区域内的）
    if bboxes2 is not None:
        for bbox in bboxes2:
            x1_orig, y1_orig, x2_orig, y2_orig, cls = bbox

            # 只保留在裁剪区域内的边界框
            if (x1_orig < x2 and x2_orig > x1 and
                    y1_orig < y2 and y2_orig > y1):
                # 裁剪到混合区域内
                x1_clip = max(x1_orig, x1)
                y1_clip = max(y1_orig, y1)
                x2_clip = min(x2_orig, x2)
                y2_clip = min(y2_orig, y2)

                if x2_clip > x1_clip and y2_clip > y1_clip:
                    mixed_bboxes.append([x1_clip, y1_clip, x2_clip, y2_clip, cls])

    if keypoints2 is not None:
        for kp_instance in keypoints2:
            new_kp_instance = []
            for kp in kp_instance:
                if len(kp) != 3:
                    continue
                x, y, v = kp

                # 检查关键点是否在裁剪区域内
                if (x1 <= x < x2 and y1 <= y < y2):
                    new_kp_instance.append([x, y, v])

            if new_kp_instance:  # 只添加有关键点的实例
                mixed_keypoints.append(new_kp_instance)

    # 处理分割掩码2
    if seg2 is not None:
        if mixed_seg is not None:
            # 将第二张图像的分割掩码应用到裁剪区域
            mixed_seg[y1:y2, x1:x2] = seg2[y1:y2, x1:x2]
        else:
            mixed_seg = np.zeros((out_h, out_w), dtype=seg2.dtype)
            mixed_seg[y1:y2, x1:x2] = seg2[y1:y2, x1:x2]

    merged_anns = {
        "bboxes": mixed_bboxes,
        "keypoints": mixed_keypoints,
        "seg": mixed_seg
    }

    return mixed_img, merged_anns

def mosaic_augmentation(img_list, out_shape, bboxes=None, keypoints=None, seg=None):
    """
    执行Mosaic数据增强，将多张图像拼接成一张图像。

    Mosaic增强将4张图像按照随机切割点拼接成一张图像，同时对相应的标注信息进行转换。

    Args:
        img_list (list of numpy.ndarray): 输入图像列表，每个元素为H×W×C的numpy数组
        out_shape (tuple): 输出图像尺寸，格式为(out_h, out_w)
        bboxes (list of list, optional): 边界框标注，格式为[[x1,y1,x2,y2,cls], ...]，
                                        每个图像对应一个边界框列表
        keypoints (list of list, optional): 关键点标注，格式为[[[x,y,v], ...], ...]，
                                          每个图像对应一个关键点实例列表，每个实例包含多个关键点
        seg (list of numpy.ndarray, optional): 分割掩码列表，每个元素为H×W的numpy数组

    Returns:
        tuple: 包含以下元素的元组
            - merged_img (numpy.ndarray): 拼接后的图像，形状为out_h×out_w×3
            - merged_anns (dict): 拼接后的标注信息，包含以下键：
                - 'bboxes': 转换后的边界框列表
                - 'keypoints': 转换后的关键点列表
                - 'seg': 转换后的分割掩码

    Raises:
        ValueError: 当输入参数不符合要求时抛出异常

    Example:
        >>> img1 = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        >>> img2 = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
        >>> img_list = [img1, img2]
        >>>
        >>> bboxes = [
        ...     [[50, 60, 150, 180, 0], [200, 80, 280, 150, 1]],
        ...     [[30, 40, 100, 200, 0]]
        ... ]
        >>>
        >>> keypoints = [
        ...     [[60, 70, 1], [140, 170, 1]],
        ...     [[50, 50, 1], [80, 180, 1]]
        ... ]
        >>>
        >>> seg = [
        ...     np.random.randint(0, 3, (200, 300), dtype=np.int32),
        ...     np.random.randint(0, 3, (250, 250), dtype=np.int32)
        ... ]
        >>>
        >>> merged_img, merged_anns = mosaic_augmentation(
        ...     img_list=img_list,
        ...     out_shape=(400, 400),
        ...     bboxes=bboxes,
        ...     keypoints=keypoints,
        ...     seg=seg
        ... )
    """
    n = len(img_list)
    if n == 0:
        raise ValueError("img_list不能为空")
    out_h, out_w = out_shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_shape必须为正整数元组 (out_h, out_w)")

    # 检查标注长度匹配
    if bboxes is not None and len(bboxes) != n:
        raise ValueError("bboxes长度必须与img_list一致")
    if keypoints is not None and len(keypoints) != n:
        raise ValueError("keypoints长度必须与img_list一致")
    if seg is not None and len(seg) != n:
        raise ValueError("seg长度必须与img_list一致")

    # 选择4张图片（循环或随机）
    if n <= 4:
        selected_indices = [i % n for i in range(4)]
        random.shuffle(selected_indices)
    else:
        selected_indices = random.sample(range(n), 4)
    selected_imgs = [img_list[i] for i in selected_indices]
    selected_bboxes = [bboxes[i] if bboxes is not None else [] for i in selected_indices]
    selected_keypoints = [keypoints[i] if keypoints is not None else [] for i in selected_indices]
    selected_seg = [seg[i] if seg is not None else None for i in selected_indices]

    # 随机分割点
    cutx = random.randint(int(out_w * 0.2), int(out_w * 0.8))
    cuty = random.randint(int(out_h * 0.2), int(out_h * 0.8))

    # 4个拼接区域
    regions = [
        {"x0": 0, "y0": 0, "w": cutx, "h": cuty},          # 左上
        {"x0": 0, "y0": cuty, "w": cutx, "h": out_h - cuty},  # 左下
        {"x0": cutx, "y0": cuty, "w": out_w - cutx, "h": out_h - cuty},  # 右下
        {"x0": cutx, "y0": 0, "w": out_w - cutx, "h": cuty}   # 右上
    ]

    # 初始化输出
    merged_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    merged_bboxes = [] if bboxes is not None else None
    merged_keypoints = [] if keypoints is not None else None
    merged_seg = np.zeros((out_h, out_w), dtype=np.int32) if seg is not None else None

    # 遍历区域处理
    for region_idx, (region, img, bbox_list, kp_list, seg_mask) in enumerate(
            zip(regions, selected_imgs, selected_bboxes, selected_keypoints, selected_seg)
    ):
        x0, y0 = region["x0"], region["y0"]
        reg_w, reg_h = region["w"], region["h"]
        img_h, img_w = img.shape[0], img.shape[1]  # 原始图片的高和宽

        # 创建与区域尺寸一致的背景（黑色）
        region_canvas = np.zeros((reg_h, reg_w, 3), dtype=np.uint8)
        # 计算原始图片在画布中的有效区域（不超过图片和画布尺寸）
        paste_h = min(img_h, reg_h)
        paste_w = min(img_w, reg_w)
        # 将原始图片粘贴到画布左上角
        region_canvas[:paste_h, :paste_w, :] = img[:paste_h, :paste_w, :]
        # 赋值到输出图像
        merged_img[y0:y0 + reg_h, x0:x0 + reg_w, :] = region_canvas

        # --------------------------
        # 2. 修正：分割掩码拼接（确保尺寸匹配）
        # --------------------------
        if seg is not None and seg_mask is not None:
            if len(seg_mask.shape) != 2:
                raise ValueError(f"分割掩码格式错误，应为2D，实际shape={seg_mask.shape}")
            seg_canvas = np.zeros((reg_h, reg_w), dtype=np.int32)  # 背景为0
            paste_h_seg = min(seg_mask.shape[0], reg_h)
            paste_w_seg = min(seg_mask.shape[1], reg_w)
            seg_canvas[:paste_h_seg, :paste_w_seg] = seg_mask[:paste_h_seg, :paste_w_seg]
            merged_seg[y0:y0 + reg_h, x0:x0 + reg_w] = seg_canvas

        # --------------------------
        # 3. 包围框处理
        # --------------------------
        if bboxes is not None:
            for bbox in bbox_list:
                x1, y1, x2, y2, cls = bbox
                # 转换坐标
                x1_out = x0 + x1
                y1_out = y0 + y1
                x2_out = x0 + x2
                y2_out = y0 + y2
                # 裁剪到区域内
                x1_out = max(x0, min(x1_out, x0 + reg_w))
                y1_out = max(y0, min(y1_out, y0 + reg_h))
                x2_out = max(x0, min(x2_out, x0 + reg_w))
                y2_out = max(y0, min(y2_out, y0 + reg_h))
                # 过滤无效框
                if x2_out > x1_out and y2_out > y1_out:
                    merged_bboxes.append([x1_out, y1_out, x2_out, y2_out, cls])

        # --------------------------
        # 4. 修正：特征点处理 - 保持实例结构
        # --------------------------
        if keypoints is not None:
            for kp_instance in kp_list:  # 每个实例的关键点列表
                new_kp_instance = []
                for kp in kp_instance:  # 单个点：[x,y,v]
                    if len(kp) != 3:
                        continue  # 确保格式正确
                    x, y, v = kp
                    if x >= img_w or y >= img_h:
                        # 关键点超出原始图像范围，标记为不可见
                        new_kp_instance.append([x0 + min(x, img_w-1), y0 + min(y, img_h-1), 0])
                        continue
                    # 转换坐标
                    x_out = x0 + x
                    y_out = y0 + y

                    # 检查是否在粘贴区域内
                    if x < paste_w and y < paste_h:
                        # 关键点在粘贴区域内，保持可见性
                        new_kp_instance.append([x_out, y_out, v])
                    else:
                        # 关键点在粘贴区域外，标记为不可见
                        new_kp_instance.append([x_out, y_out, 0])
                merged_keypoints.append(new_kp_instance)

    merged_anns = {
        "bboxes": merged_bboxes,
        "keypoints": merged_keypoints,
        "seg": merged_seg
    }
    return merged_img, merged_anns


def collate_cutmix_mixup_(batch,prod=0.5,NUM_CLASSES=None):
    """
    Mixup:将随机的两张样本按比例混合，分类的结果按比例分配；
    CutMix:就是将一部分区域cut掉但不填充0像素而是随机填充训练集中的其他数据的区域像素值，分类结果按一定的比例分配;
    Args:
        num_classes：批次中类别数量，用于将标签进行 one-hot 编码，如果标签已经是 one-hot 编码形式，则可以为 None
    """
    if random.random()>prod:
        cutmix = CutMix(num_classes=NUM_CLASSES)
        mixup = MixUp(num_classes=NUM_CLASSES)
        cutmix_or_mixup = RandomChoice([cutmix, mixup])
        return  cutmix_or_mixup(*default_collate(batch))
    else:
        return batch



def iaa2tv(image,data):
    # 将imgaug类型转换为torchvision.transforms.v2类
    # 版本要求
    check_tv=int(torchvision.__version__.split(".")[1])>22 #0.23版本才支持特征点

    H,W = data.shape[:2]
    image =torch.tensor(image,dtype=torch.uint8).permute(2,0,1) #(H,W,3)-->(3,H,W)
    if type(data) == KeypointsOnImage and check_tv:
        kp_data = [[kp.x, kp.y] for kp in data]
        result =tv_tensors.KeyPoints(kp_data, canvas_size=(H, W))
    elif type(data) ==BoundingBoxesOnImage:
        aug_bboxes = [ [bbox.x1,bbox.y1,bbox.x2,bbox.y2] for  bbox in data.bounding_boxes]
        result = tv_tensors.BoundingBoxes(aug_bboxes, format="XYXY", canvas_size=(H, W))
    elif type(data) ==SegmentationMapsOnImage:
        data= torch.from_numpy(data.arr).permute(2,1,0) #(H,W,1)-->(1,H,W)
        result=tv_tensors.Mask(data)
    else:
        raise TypeError("Unsupported data type")
    return image,result


def tv2iaa(image,data):
    # 将为torchvision.transforms.v2类转换imgaug类型
    check_tv=int(torchvision.__version__.split(".")[1])>22 #0.23版本才支持特征点
    if "PIL" in str(type(image)):
        image =np.array(image)
    else:
        image = image.permute(1,2,0).cpu().numpy() #(3,H,W)-->(H,W,3)
    if check_tv:
        if type(data) == tv_tensors.KeyPoints:
            kp_data = [Keypoint(x=kp[0], y=kp[1]) for kp in data.data]
            result=KeypointsOnImage(kp_data)
    elif type(data) ==tv_tensors.BoundingBoxes:
        if data.format.name== "XYXY":
            aug_bboxes = [BoundingBox(x1=bbox[0],y1=bbox[1],x2=bbox[2],y2=bbox[3])  for bbox in data.data]
            result = BoundingBoxesOnImage(aug_bboxes, shape=image.shape)
        else:
            raise TypeError("Unsupported data type,Only support XYXY")
    elif type(data) ==tv_tensors.Mask:
        data = data.permute(1,2,0).cpu().numpy() #(1,H,W)-->(H,W,1)
        result=SegmentationMapsOnImage(data, shape=image.shape)
    else:
        raise TypeError("Unsupported data type")
    return image,result





def box_image_show(image,box_list):
    if type(image) is str:
        import imageio
        image = imageio.v2.imread(image, pilmode="RGB")
    # 边界框
    boxes=[]
    for box in box_list:
        if len(box) == 4:
            boxes.append(BoundingBox(box[0], box[1], box[2], box[3]))
        elif len(box) == 5:
            # 含标签
            boxes.append(BoundingBox(box[0], box[1], box[2], box[3],label=box[4]))
        else:
            print("请按照[x1,y1,x2,y2,label] 或 [x1,y1,x2,y2] 提供列表")

    bbs = BoundingBoxesOnImage(boxes, shape=image.shape)
    image_box = bbs.draw_on_image(image, size=2)
    ia.imshow(image_box)

if __name__ == '__main__':
    os.environ["DISPLAY"] = ":0"
    # 测试box_image_show
    #image = ia.quokka(size=(256, 256))
    #box_image_show(image,[[65, 100, 200, 150],[150,80, 200,130,"test"]])

    # 测试tv box
    # from torchvision.io import decode_image
    # import torchvision
    # print()
    # img = decode_image("tmp/astronaut.jpg")
    # print(f"{type(img) = }, {img.dtype = }, {img.shape = }")
    # boxes = tv_tensors.BoundingBoxes(
    #     [
    #         [15, 10, 370, 510],
    #         [275, 340, 510, 510],
    #         [130, 345, 210, 425]
    #     ],
    #     format="XYXY", canvas_size=img.shape[-2:])
    # iaa_img,iaa_res = tv2iaa(img,boxes)
    # #ia.imshow(iaa_res.draw_on_image(iaa_img))
    # # 还原
    # tv_img,tv_res = iaa2tv(iaa_img,iaa_res)
    # print(f"{type(tv_img) = }, {tv_img.dtype = }, {tv_img.shape = }")
    # print(tv_res)
    # assert torch.equal( tv_img, img)

    # # 测试tv分割
    # from torchvision import datasets
    # dataset = datasets.CocoDetection("tmp", "tmp/instances.json")
    # dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=("boxes", "labels", "masks"))
    # img, target =dataset[1]
    # print(f"{type(img) = }\n{type(target) = }\n{target.keys() = }")
    # print(f"{type(target['boxes']) = }\n{type(target['labels']) = }\n{type(target['masks']) = }")
    # print(target['masks'].shape)
    # iaa_img,iaa_res = tv2iaa(img,target['masks'])
    # ia.imshow(iaa_res.draw_on_image(iaa_img)[0])
    # # 还原
    # tv_img,tv_res = iaa2tv(iaa_img,iaa_res)
    # print(f"{type(tv_img) = }, {tv_img.dtype = }, {tv_img.shape = }")
    # print(tv_res.shape)
    # print(target['masks'], tv_res)

    # 测试拼图
    # 构造示例数据
    img1 =  ia.quokka(size=(200, 300))  # 图1：200x300
    img2 = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)  # 图2：250x250
    img_list = [img1, img2]

    # 构造标注（与img_list对应）
    bboxes = [
        [[50, 60, 150, 180, 0], [200, 80, 280, 150, 1]],  # 图1的包围框+类别
        [[30, 40, 100, 200, 0]]  # 图2的包围框+类别
    ]
    keypoints = [
        [ [[60, 70, 1]], [[140, 170, 1]] ],  # 图1的两个实例，每个实例一个关键点
        [ [[50, 50, 1]], [[80, 180, 1]] ]    # 图2的两个实例，每个实例一个关键点
    ]

    seg = [
        ia.quokka_segmentation_map( size=(200, 300)).arr.reshape((200, 300)),#np.random.randint(0, 3, (200, 300), dtype=np.int32),  # 图1的分割掩码
        np.random.randint(0, 3, (250, 250), dtype=np.int32)   # 图2的分割掩码
    ]
    seg_0 =  SegmentationMapsOnImage(seg[0],img1.shape)
    key_0 =  KeypointsOnImage([Keypoint(xy[0][0],xy[0][1]) for xy in keypoints[0]],img1.shape)
    box_0 = BoundingBoxesOnImage([BoundingBox(xy[0],xy[1],xy[2],xy[3],label=xy[4]) for xy in bboxes[0]],img1.shape)
    res = seg_0.draw_on_image(img1)[0]
    res  = key_0.draw_on_image(res)
    #res = box_0.draw_on_image(res)
    #ia.imshow(res)
    # 执行拼接（输出尺寸640x640）
    merged_img, merged_anns = mosaic_augmentation(
        img_list=img_list,
        out_shape=(200, 200),
        bboxes=bboxes,
        keypoints=keypoints,
        seg=seg
    )

    # 输出结果信息
    print(f"拼接后图像尺寸: {merged_img.shape}")
    print(f"合并后包围框数量: {len(merged_anns['bboxes'])}")
    print(f"合并后特征点数量: {len(merged_anns['keypoints'])}")
    print(f"合并后分割掩码尺寸: {merged_anns['seg'].shape}")
    print(f"{merged_anns['keypoints'] =}")
    seg_0 =  SegmentationMapsOnImage(merged_anns['seg'],merged_img.shape)
    keypoints_0 = []
    for xy in merged_anns['keypoints']:
        if xy[0][2]!=0:
            keypoints_0.append(Keypoint(xy[0][0],xy[0][1]))
    key_0 =  KeypointsOnImage(keypoints_0 ,merged_img.shape)
    box_0 = BoundingBoxesOnImage([BoundingBox(xy[0],xy[1],xy[2],xy[3],label=xy[4]) for xy in merged_anns['bboxes']],merged_img.shape)
    res = seg_0.draw_on_image(merged_img)[0]
    res  = key_0.draw_on_image(res)
    #res = box_0.draw_on_image(res)
    ia.imshow(res)




