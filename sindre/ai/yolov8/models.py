"""
由于业界大都是用YOLOV5/V8,原因是YOLOV5在实际使用时泛化性精度并不比V6-v12低;
而且各个厂商对YOLOV5支持及优化很好;YOLOV8由于分割头多样性而适用用各个场景;

本YOLOV8模型及代码提取自:https://github.com/ultralytics/ultralytics
1. Backbone：采用改进的 CSPNet 结构，通过 C2f 模块提取多尺度特征（P3/P4/P5）。
2. Neck：使用上采样（Upsample）和下采样（Conv）结合特征拼接（Concat），实现多尺度特征融合。
3. Head：采用无锚框（Anchor-free）的分割头（Detect 层），直接预测目标坐标和类别，避免了锚框预设的局限性

支持:
1. 检测
2. 实例分割
3. 关键点
4. 定向框检测
5. 分类

"""
import torch
import torch.nn as nn
from sindre.ai.yolov8.components.backbone import Backbone
from sindre.ai.yolov8.components.head import Detect,Pose, Classify,OBB,Segment
from sindre.ai.yolov8.components.losses import v8PoseLoss,v8OBBLoss,v8DetectionLoss,v8SegmentationLoss,v8ClassificationLoss
from sindre.ai.yolov8.components.neck import Neck


class YOLOV8(nn.Module):
    # YOLOV8 共5种型号,分别对应模型深度(迭代次数)，宽度(隐藏通道)
    model_scale={
        'n': [1,  [16,32,64,128,256]],
        's': [1,  [32,64,128,256,512]],
        'm': [2,  [48,96,192,384,576]],
        'l': [3,  [64,128,256,512,512]],
        'x': [3,  [80,160,320,640,640]]
    }

    def __init__(self,input_channels=3,num_classes=19,model_name:str="n",task:str="Detect",kpt_shape=None):
        """
        YOLOV8统一模型类，支持检测、分割、姿态、OBB、分类5种任务;

        Args:
            input_channels: 输入图像通道数，默认3（RGB）;
            num_classes: 任务类别数，默认2类;
            model_name: 模型尺度，需为'n'/'s'/'m'/'l'/'x'，控制模型深度和宽度;
            task: 任务类型，需为"Detect"/"Segment"/"Pose"/"Classify"/"OBB";
            kpt_shape:如果为“Pose”任务,需要提供(关键点数量, 维度),维度:2维(x,y)/3维(x,y,可见性);
        """
        super(YOLOV8, self).__init__()
        assert task in ["Detect", "Segment", "Pose", "Classify", "OBB"]
        assert model_name in ["n","s","m","l","x"]
        if task == "Pose":
            assert kpt_shape is not None
            assert len(kpt_shape) == 2
            assert kpt_shape[1] == 2 or kpt_shape[1] == 3

        # 获取模型尺度
        model_args = self.model_scale[model_name]
        depth,hidden_channels= model_args[0], model_args[1]
        # 生成主干
        self.backbone   = Backbone(input_channels=input_channels,
                                   hidden_channels=hidden_channels,
                                   init_iter_n=depth)
        # 生成颈部
        self.neck = Neck(hidden_channels=hidden_channels,
                         init_iter_n=depth,
                         )
        # 颈部生成特征通道
        p3_ch,p4_ch,p5_ch =hidden_channels[2],hidden_channels[3],hidden_channels[4]
        # 生成头部，共五种
        if task == "Detect":
            # 检测Head：预测边界框（x,y,w,h）和类别
            self.head = Detect(nc=num_classes,
                               ch=(p3_ch,p4_ch,p5_ch)
                               )
        elif task == "Segment":
            # 分割Head：检测+掩码预测
            # 注： “为每个目标预测一个专属的掩码”（实例分割：区分同一类别的不同个体）
            # 每个目标只需预测 32 个系数，100 个目标仅需 3,200 个值。计算量随目标数量增长的速度大幅降低，这是 YOLO 兼顾速度和精度的关键设计。
            self.head = Segment(nc=num_classes,    # 类别数
                                nm=32,             # 掩码系数数量: 表示为每个目标预测 32 个 “系数”,系数的作用是 “加权组合原型”：每个目标的最终掩码 = 原型掩码 × 对应系数的加权和（类似 “用字典中的元素拼出目标”）。
                                npr=p3_ch,           # 原型数量:表示生成 256 个 “基础原型掩码”,原型是 “通用的掩码片段”，涵盖了图像中可能出现的各种边缘、纹理模式（类似 “字典” 中的基础元素）
                                ch=(p3_ch,p4_ch,p5_ch)# 输入特征通道
                                )
        elif task == "Pose":
            # 单类别姿态Head：检测+关键点预测
            # num_classes：表示 “目标的类别数”（如 80 类，包含人、猫、狗等）。
            # kpt_shape=(17,3)：表示 “当前类别目标的关键点配置”（17 个关键点，每个点用 3 个值描述）。
            #num_classes 是 “目标的类别总数”（如 80 类），而 kpt_shape 是 “某一类目标内部的关键点配置”。
            #例如：当 num_classes=80 时，可能只有 “人” 这 1 个类别需要预测关键点（17 个），其他 79 类（如椅子、杯子）不需要预测关键点。
            #YOLO 的姿态头通常针对 “单一主类别” 设计（如仅检测人体），因此 kpt_shape 固定为该类别的关键点数量，与总类别数无关。
            head = Pose(nc=num_classes,
                        kpt_shape=(17,3),# 关键点配置：(数量, 维度)
                        ch=(p3_ch,p4_ch,p5_ch)
                        )
        elif self.task == "Classify":
            # 分类Head:融合P3/P4/P5特征,输出类别数
            head = Classify(c1=p3_ch+p4_ch+p5_ch,
                            c2=num_classes
                            )
        elif self.task == "OBB":
            # 定向边界框Head：检测+旋转角度预测（ne=1，仅需1个参数表示角度θ）
            head = OBB(nc=num_classes,
                            ne=1,  # 旋转角度
                            ch=(p3_ch,p4_ch,p5_ch)
                       )

    def forward(self, x):
        p3_feat, p4_feat, p5_feat = self.backbone(x)
        print(p3_feat.shape, p4_feat.shape, p5_feat.shape)
        p3_p4_p5_list = self.neck(p3_feat, p4_feat, p5_feat)
        x = self.head(p3_p4_p5_list)
        return x





if __name__ == "__main__":
    model = YOLOV8(model_name="l")
    print(model)
    x = torch.randn(2,3,640,640)
    y = model(x)

    print(y)