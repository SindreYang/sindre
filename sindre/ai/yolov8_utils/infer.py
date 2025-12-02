from sindre.ai.yolov8.components.head import dist2bbox
import torch
from torchvision.ops import nms
import numpy as np
def decode_box( inputs,input_shape):
    # dbox  batch_size, 4, 8400
    # cls   batch_size, 20, 8400
    dbox, cls, origin_cls, anchors, strides = inputs
    # 获得中心宽高坐标
    dbox    = dist2bbox(dbox, anchors.unsqueeze(0), xywh=True, dim=1) * strides
    y       = torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1)
    # 进行归一化，到0~1之间
    y[:, :, :4] = y[:, :, :4] / torch.Tensor([input_shape[1], input_shape[0],input_shape[1],input_shape[0]]).to(y.device)
    return y
def non_max_suppression( prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
        #----------------------------------------------------------#
        #   将预测结果的格式转换成左上角右下角的格式。
        #   prediction  [batch_size, num_anchors, 85]
        #----------------------------------------------------------#
        box_corner          = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]
        for i, image_pred in enumerate(prediction):
            #----------------------------------------------------------#
            #   对种类预测部分取max。
            #   class_conf  [num_anchors, 1]    种类置信度
            #   class_pred  [num_anchors, 1]    种类
            #----------------------------------------------------------#
            class_conf, class_pred = torch.max(image_pred[:, 4:4 + num_classes], 1, keepdim=True)

            #----------------------------------------------------------#
            #   利用置信度进行第一轮筛选
            #----------------------------------------------------------#
            conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()

            #----------------------------------------------------------#
            #   根据置信度进行预测结果的筛选
            #----------------------------------------------------------#
            image_pred = image_pred[conf_mask]
            class_conf = class_conf[conf_mask]
            class_pred = class_pred[conf_mask]
            if not image_pred.size(0):
                continue
            #-------------------------------------------------------------------------#
            #   detections  [num_anchors, 6]
            #   6的内容为：x1, y1, x2, y2, class_conf, class_pred
            #-------------------------------------------------------------------------#
            detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)

            #------------------------------------------#
            #   获得预测结果中包含的所有种类
            #------------------------------------------#
            unique_labels = detections[:, -1].cpu().unique()

            if prediction.is_cuda:
                unique_labels = unique_labels.cuda()
                detections = detections.cuda()

            for c in unique_labels:
                #------------------------------------------#
                #   获得某一类得分筛选后全部的预测结果
                #------------------------------------------#
                detections_class = detections[detections[:, -1] == c]
                #------------------------------------------#
                #   使用官方自带的非极大抑制会速度更快一些！
                #   筛选出一定区域内，属于同一种类得分最大的框
                #------------------------------------------#
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]

                # # 按照存在物体的置信度排序
                # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
                # detections_class = detections_class[conf_sort_index]
                # # 进行非极大抑制
                # max_detections = []
                # while detections_class.size(0):
                #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
                #     max_detections.append(detections_class[0].unsqueeze(0))
                #     if len(detections_class) == 1:
                #         break
                #     ious = bbox_iou(max_detections[-1], detections_class[1:])
                #     detections_class = detections_class[1:][ious < nms_thres]
                # # 堆叠
                # max_detections = torch.cat(max_detections).data

                # Add max detections to outputs
                output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

            if output[i] is not None:
                output[i]           = output[i].cpu().numpy()
                box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
                output[i][:, :4]    = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
        return output


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        #-----------------------------------------------------------------#
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes
def yolo_infer(image_path ,model_path, shape=(640,640)):

    from PIL import Image
    from models import YOLOV8
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net    = YOLOV8(3,80,"n",task="Detect").to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    #net    = net.fuse().eval()

    # 加载图片并转成RGB
    image= Image.open(image_path).convert('RGB')
    # resize到指定大小
    image  = image.resize(shape, resample=Image.BICUBIC)

    with torch.no_grad():
        image = torch.from_numpy(image[None]).to(device=device,dtype=torch.float32)
        outputs = net(image)

    # 转换成box
    outputs = decode_box(outputs)
    # 进行非极大抑制
    outputs = non_max_suppression(outputs, num_classes, shape,
                                                 image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
    top_label   = np.array(outputs[0][:, 5], dtype = 'int32')
    top_conf    = outputs[0][:, 4]
    top_boxes   = outputs[0][:, :4]
    results    =[]
    for i, c in enumerate(top_label):
        result                      = {}
        top, left, bottom, right    = top_boxes[i]

        result["image_id"]      = int(image_id)
        result["category_id"]   = clsid2catid[c]
        result["bbox"]          = [float(left),float(top),float(right-left),float(bottom-top)]
        result["score"]         = float(top_conf[i])
        results.append(result)