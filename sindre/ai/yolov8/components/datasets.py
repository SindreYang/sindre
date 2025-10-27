
import  torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageDraw, ImageFont
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox,BoundingBoxesOnImage
from sindre.utils2d.aug_images import cutmix_augmentation,mosaic_augmentation,box_xyxy2cxcywh,box_cxcywh2xyxy
import random
class IMGDataset(Dataset):
    def __init__(self,
                 data,
                 out_shape,
                 batch_size: int = 2,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 prefetch_factor=None,
                 val = False,):
        super(Dataset, self).__init__()
        self.datasets   = data
        self.length = len(self.datasets)


        self.out_shape = out_shape
        self.val = val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor=prefetch_factor

        self.seq = iaa.Sequential([
                        iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
                        iaa.Affine(
                            translate_px={"x": 40, "y": 60},
                            scale=(0.5, 0.7)
                        ) # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
                        ])
        self.DEBUG=False
        # 缓存
        self.cache_images=[]
        self.cache_bboxes=[]
    def __len__(self):
        return  self.length

    def normalization(self,image,boxes):

        #image:  [3, H, W], 归一化到0-1;  res_anns: [nL, 6],归一化到0-1;  每行格式: [占位符, 类别ID, 中心x, 中心y, 宽, 高])
        image       = np.transpose(np.array(image, dtype=np.float32)/255, (2, 0, 1))
        boxes = np.array(boxes, dtype=np.float32)
        # 归一化边界框坐标
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.out_shape[1]  # x坐标归一化
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.out_shape[0]  # y坐标归一化

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


    def collate_fun(self,batch):
        images  = []
        bboxes  = []
        for i, (img, box) in enumerate(batch):
            images.append(img)
            box[:, 0] = i
            bboxes.append(box)

        images  = torch.from_numpy(np.array(images)).to(torch.float32)
        bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).to(torch.float32) # 合并成一个大的 numpy 数组,后期根据box[:, 0]来找对应图片
        return images, bboxes


    def __getitem__(self, idx):
        # 混合增强
        if len(self.cache_images)>2 and random.random()>0.5:
            # 有缓存且概率大于0.5，则进行拼图增强
            image,aug_anns = mosaic_augmentation(self.cache_images,out_shape=self.out_shape,bboxes=self.cache_bboxes)
            boxes  = aug_anns["bboxes"]
            # 用完这个缓存,则清空
            self.cache_images=[]
            self.cache_bboxes=[]
        else:
            # 获取数据
            data = self.datasets[idx].split()
            image   = np.array(Image.open(data[0]).convert('RGB'))
            boxes     = np.array([np.array(list(map(int,box.split(',')))) for box in data[1:]])
            # 添加到队列中，队列最多保存4个
            if len(self.cache_images)<4:
                print(image.shape)
                self.cache_images.append(image.copy())
                self.cache_bboxes.append(boxes.copy())
            else:
                # 如果缓存已满，移除最旧的，添加最新的
                self.cache_images.pop(0)
                self.cache_bboxes.pop(0)
                self.cache_images.append(image.copy())
                self.cache_bboxes.append(boxes.copy())

        iaa_boxes=[BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3],label=box[4]) for  box in boxes]
        bbs = BoundingBoxesOnImage(iaa_boxes, shape=image.shape)
        # 重新缩放
        image = ia.imresize_single_image(image, self.out_shape)
        bbs = bbs.on(image)
        if self.DEBUG:
            ia.imshow( bbs.draw_on_image(image))

         # 单张增强
        if not self.val:
            image, bbs = self.seq(image=image, bounding_boxes=bbs)

        if self.DEBUG:
            ia.imshow( bbs.draw_on_image(image))
        # 还原
        res_box = [[box.x1,box.y1,box.x2,box.y2,box.label] for  box in bbs.bounding_boxes]


        return self.normalization(image,res_box)



    def train_dataloader(self,multi_gpu=False):
        if multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self)
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,# 必须禁用，避免多卡冲突
                prefetch_factor=self.prefetch_factor,
                collate_fn=self.collate_fun,
                drop_last=True,
                sampler=sampler,
            )
        else:
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                collate_fn=self.collate_fun,
                prefetch_factor=self.prefetch_factor,
                drop_last=True,
            )


    def val_dataloader(self,multi_gpu=False):
        if multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self)
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,# 必须禁用，避免多卡冲突
                prefetch_factor=self.prefetch_factor,
                collate_fn=self.collate_fun,
                drop_last=True,
                sampler=sampler,
            )
        else:
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,# 必须禁用，避免多卡冲突
                prefetch_factor=self.prefetch_factor,
                collate_fn=self.collate_fun,
                drop_last=True,
            )






if __name__ == '__main__':
    import os
    os.environ["DISPLAY"] = ":0"
    # 读取txt
    with open(r"2007_train.txt", encoding='utf-8') as f:
        train_lines = f.readlines()
    class_names=["aeroplane",
                 "bicycle",
                 "bird",
                 "boat",
                 "bottle",
                 "bus",
                 "car",
                 "cat",
                 "chair",
                 "cow",
                 "diningtable",
                 "dog",
                 "horse",
                 "motorbike",
                 "person",
                 "pottedplant",
                 "sheep",
                 "sofa",
                 "train",
                 "tvmonitor",]
    print(len(class_names))


    dataset = IMGDataset(train_lines, [640, 640])  # [H, W]
    dataloader = dataset.train_dataloader()

    for batch in dataloader:
        images, boxes = batch

        # 可视化第一张图的第一个边界框
        if len(boxes) > 0:
            # 获取属于第一张图的所有边界框
            first_img_boxes = boxes[boxes[:, 0] == 0]  # 筛选出属于索引0的图片的边界框

            if len(first_img_boxes) > 0:
                # 获取第一个边界框
                box = first_img_boxes[0]
                label = int(box[1])
                cx, cy, w, h = box[2], box[3], box[4], box[5]

                # 转换为左上角和右下角坐标（归一化值）
                x1, y1, x2, y2 = box_cxcywh2xyxy(cx, cy, w, h)

                # 获取原图并还原到0-255范围
                img = images[0].permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)
                img = (img * 255).astype(np.uint8)

                # 将归一化坐标映射回图像尺寸
                H, W = img.shape[0], img.shape[1]
                x1, x2 = x1 * W, x2 * W
                y1, y2 = y1 * H, y2 * H

                # 创建边界框并显示
                bbox = BoundingBoxesOnImage(
                    [BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=class_names[label])],
                    shape=img.shape
                )
                ia.imshow(bbox.draw_on_image(img))
                #break
    else:
        print("未找到有效的边界框进行可视化")
