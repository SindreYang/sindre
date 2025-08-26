import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred_output, gt):
        """
        Input:
            - pred_output: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map #这是原来的输入,最新输入为(N, C, H, W)
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred_output.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred_output, dim=1)

        # one-hot vector of ground truth
        #one_hot_gt = one_hot(gt.long(), c) # 这是原来的输入,最新输入为(N, C, H, W)
        one_hot_gt = gt



        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss





def dice_loss(input, target, eps=1e-7, if_sigmoid=True):
    if if_sigmoid:
        input = F.sigmoid(input)
    b = input.shape[0]
    iflat = input.contiguous().view(b, -1)
    tflat = target.float().contiguous().view(b, -1)
    intersection = (iflat * tflat).sum(dim=1)
    L = (1 - ((2. * intersection + eps) / (iflat.pow(2).sum(dim=1) + tflat.pow(2).sum(dim=1) + eps))).mean()
    return L

def smooth_truncated_loss(p, t, ths=0.06, if_reduction=True, if_balance=True):
    import math
    n_log_pt = F.binary_cross_entropy_with_logits(p, t, reduction='none')
    pt = (-n_log_pt).exp()
    L = torch.where(pt>=ths, n_log_pt, -math.log(ths)+0.5*(1-pt.pow(2)/(ths**2)))
    if if_reduction:
        if if_balance:
            return 0.5*((L*t).sum()/t.sum().clamp(1) + (L*(1-t)).sum()/(1-t).sum().clamp(1))
        else:
            return L.mean()
    else:
        return L

def balance_bce_loss(input, target):
    L0 = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    return 0.5*((L0*target).sum()/target.sum().clamp(1)+(L0*(1-target)).sum()/(1-target).sum().clamp(1))



import math


# combined with cross entropy loss, instance level
class LossVariance(nn.Module):
    """ The instances in target should be labeled
    """

    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):

        B = input.size(0)

        loss = 0
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]

            sum_var = 0
            for val in unique_vals:
                instance = input[k][:, target[k] == val]
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
        loss /= B
        return loss



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True, type="sigmoid"):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.type = type

    def forward(self, logit, target, class_weight=None):
        target = target.view(-1, 1).long()
        if self.type == 'sigmoid':
            if class_weight is None:
                class_weight = [1]*2

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif self.type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit,1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob*select).sum(1).view(-1,1)
        prob = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss

# Robust focal loss
class RobustFocalLoss2d(nn.Module):
    #assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True, type="sigmoid"):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.type = type

    def forward(self, logit, target, class_weight=None):
        target = target.view(-1, 1).long()
        if self.type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2

            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif self.type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C

            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob  = (prob*select).sum(1).view(-1,1)
        prob  = torch.clamp(prob,1e-8,1-1e-8)

        focus = torch.pow((1-prob), self.gamma)
        focus = torch.clamp(focus,0,2)

        batch_loss = - class_weight *focus*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss



class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss




class Weight_DiceLoss(nn.Module):
    def __init__(self):
        super(Weight_DiceLoss, self).__init__()

    def forward(self, input, target, weights):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        weights = weights.view(N, -1)

        intersection = input_flat * target_flat
        intersection = intersection * weights

        dice = 2 * (intersection.sum(1) + smooth) / ((input_flat * weights).sum(1) + (target_flat * weights).sum(1) + smooth)
        loss = 1 - dice.sum() / N

        return loss


class WeightMulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(WeightMulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # weights = torch.ones(C) #uniform weights for all classes
        # weights[0] = 3
        dice = DiceLoss()
        wdice = Weight_DiceLoss()
        totalLoss = 0

        for i in range(C):
            # diceLoss = dice(input[:, i], target[:, i])
            # diceLoss2 = 1 - wdice(input[:, i], target[:, i - 1])
            # diceLoss3 = 1 - wdice(input[:, i], target[:, i%(C-1) + 1])
            # diceLoss = diceLoss - diceLoss2 - diceLoss3

            # diceLoss = dice(input[:, i - 1] + input[:, i] + input[:, i%(C-1) + 1], target[:, i])
            ''''''
            if (i == 0):
                diceLoss = wdice(input[:, i], target[:, i], weights) * 2
            elif (i == 1):
                # diceLoss = dice(input[:, C - 1] + input[:, i] + input[:, i + 1], target[:, i])
                diceLoss = wdice(input[:, i], target[:, i], weights)
                diceLoss2 = 1 - wdice(input[:, i], target[:, C - 1], weights)
                diceLoss3 = 1 - wdice(input[:, i], target[:, i + 1], weights)
                diceLoss = diceLoss - diceLoss2 - diceLoss3

            elif (i == C - 1):
                # diceLoss = dice(input[:, i - 1] + input[:, i] + input[:, 1], target[:, i])
                diceLoss = wdice(input[:, i], target[:, i], weights)
                diceLoss2 = 1 - wdice(input[:, i], target[:, i - 1], weights)
                diceLoss3 = 1 - wdice(input[:, i], target[:, 1], weights)
                diceLoss = diceLoss - diceLoss2 - diceLoss3

            else:
                # diceLoss = dice(input[:, i - 1] + input[:, i] + input[:, i + 1], target[:, i])
                diceLoss = wdice(input[:, i], target[:, i], weights)
                diceLoss2 = 1 - wdice(input[:, i], target[:, i - 1], weights)
                diceLoss3 = 1 - wdice(input[:, i], target[:, i + 1], weights)
                diceLoss = diceLoss - diceLoss2 - diceLoss3

            #if weights is not None:
            #diceLoss *= weights[i]

            totalLoss += diceLoss
            avgLoss = totalLoss/C

        return avgLoss





class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=3, feat_dim=3, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            print(self.centers)
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, input_x, input_label):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        labels = input_label
        batch_size = input_x.size(0)
        channels = input_x.size(1)

        distmat = torch.pow(input_x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, input_x, self.centers.t()) # math:: out = beta * mat + alpha * (mat1_i @ mat2_i)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels2 = input_label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels2.cuda().eq(classes.expand(batch_size, self.num_classes))  # eq() 想等返回1, 不相等返回0

        dist = distmat * mask.float()


        # torch.clamp(input, min, max, out=None) 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss




class StyleTransfer:
    """
    风格迁移模型类，包含风格迁移所需的各种损失函数及前向计算方法
    https://zh.d2l.ai/chapter_computer-vision/neural-style.html#id8
    """

    def __init__(self, content_weight=1, style_weight=1e3, tv_weight=10):
        """
        初始化风格迁移模型参数

        参数:
            content_weight: 内容损失的权重
            style_weight: 风格损失的权重
            tv_weight: 总变差损失的权重
        """
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

    @staticmethod
    def gram_matrix(x):
        """
        计算输入特征图的Gram矩阵

        Gram矩阵用于衡量特征图之间的相关性，是风格损失计算的核心
        Gram矩阵的元素(i,j)表示第i个特征图与第j个特征图的内积

        参数:
            x: 输入特征图，形状为(batch_size, channels, height, width)

        返回:
            gram: 计算得到的Gram矩阵，形状为(batch_size, channels, channels)
        """
        # 获取输入特征图的维度信息
        batch_size, channels, height, width = x.size()

        # 将特征图展平为二维矩阵 (channels, height*width)
        features = x.view(batch_size, channels, height * width)

        # 计算Gram矩阵: (channels, height*width) 与 (height*width, channels) 的矩阵乘法
        gram = torch.bmm(features, features.transpose(1, 2))

        # 归一化处理，除以特征图元素总数
        return gram / (channels * height * width)

    @staticmethod
    def style_loss(Y_hat, gram_Y):
        """
        计算风格损失

        风格损失用于衡量生成图像与风格图像在风格上的差异
        通过比较两者特征图的Gram矩阵实现

        参数:
            Y_hat: 生成图像的特征图
            gram_Y: 风格图像特征图的Gram矩阵

        返回:
            风格损失值
        """
        # 计算生成图像特征图的Gram矩阵并与风格图像的Gram矩阵比较
        # 使用detach()从计算图中分离gram_Y，避免对其求梯度
        return torch.square(StyleTransfer.gram_matrix(Y_hat) - gram_Y.detach()).mean()

    @staticmethod
    def content_loss(Y_hat, Y):
        """
        计算内容损失

        内容损失用于衡量生成图像与内容图像在内容上的差异
        通过比较两者的特征图直接实现

        参数:
            Y_hat: 生成图像的特征图
            Y: 内容图像的特征图

        返回:
            内容损失值
        """
        # 比较生成图像与内容图像的特征图
        # 使用detach()从计算图中分离Y，避免对内容图像求梯度
        return torch.square(Y_hat - Y.detach()).mean()

    @staticmethod
    def tv_loss(Y_hat):
        """
        计算总变差损失(Total Variation Loss)

        总变差损失用于使生成图像更加平滑，减少高频噪声
        通过计算相邻像素之间的差异实现

        参数:
            Y_hat: 生成的图像

        返回:
            总变差损失值
        """
        # 计算水平方向相邻像素的差异
        horizontal_diff = torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean()
        # 计算垂直方向相邻像素的差异
        vertical_diff = torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean()

        # 返回水平和垂直方向差异的平均值的一半
        return 0.5 * (horizontal_diff + vertical_diff)

    def forward(self, X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
        """
        前向传播计算总损失

        参数:
            X: 生成的图像
            contents_Y_hat: 生成图像在内容层的特征图列表
            styles_Y_hat: 生成图像在风格层的特征图列表
            contents_Y: 内容图像在内容层的特征图列表
            styles_Y_gram: 风格图像在风格层的Gram矩阵列表

        返回:
            contents_l: 各内容层的损失列表
            styles_l: 各风格层的损失列表
            tv_l: 总变差损失
            total_loss: 总损失
        """
        # 分别计算内容损失、风格损失（带权重）
        contents_l = [self.content_loss(Y_hat, Y) * self.content_weight
                      for Y_hat, Y in zip(contents_Y_hat, contents_Y)]

        styles_l = [self.style_loss(Y_hat, Y) * self.style_weight
                    for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]

        # 计算总变分损失（带权重）
        tv_l = self.tv_loss(X) * self.tv_weight

        # 计算总损失（注意原代码中的10倍风格损失系数）
        total_loss = sum(10 * loss for loss in styles_l) + sum(contents_l) + tv_l

        return contents_l, styles_l, tv_l, total_loss



class BinaryFocalLoss(nn.Module):
    """
    二分类场景下的Focal Loss损失函数
    论文参考: <https://arxiv.org/abs/1708.02002>

    Focal Loss通过引入gamma参数降低易分类样本的权重，
    使模型更专注于难分类样本；同时通过alpha参数处理二分类中的类别不平衡问题。
    适用于二分类任务，如目标检测中的前景/背景分类、医学影像中的异常检测等。
    """
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """
        初始化二分类Focal Loss参数

        参数:
            gamma (float): 聚焦参数，控制难易样本的权重调节程度，gamma≥0。
                           值越大，模型越关注难分类样本，默认值为2.0
            alpha (float): 类别平衡参数，用于平衡正负样本比例，需满足0 < alpha < 1。
                           通常正样本比例低时，alpha可设大一些，默认值为0.5
            logits (bool): 指示预测值是否为logits（未经过sigmoid激活的值）。
                           - True: 输入为logits，内部会计算sigmoid
                           - False: 输入为已经过sigmoid的概率值，默认值为True
            reduce (bool): 指示是否对损失进行聚合。
                           - True: 返回聚合后的标量损失（平均值）
                           - False: 返回与输入同形状的损失张量，默认值为True
            loss_weight (float): 损失的整体权重系数，用于调整该损失在多损失函数中的占比，默认值为1.0
        """
        super(BinaryFocalLoss, self).__init__()
        # 验证alpha参数有效性，必须在(0,1)范围内
        assert 0 < alpha < 1, f"alpha必须在(0,1)范围内，但得到{alpha}"
        # 验证gamma参数有效性，必须为非负数
        assert gamma >= 0, f"gamma必须为非负数，但得到{gamma}"
        # 验证loss_weight参数有效性，必须为非负数
        assert loss_weight >= 0, f"loss_weight必须为非负数，但得到{loss_weight}"

        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """
        前向计算损失

        参数:
            pred (torch.Tensor): 模型预测输出，形状为(N, ...)，
                                其中N为批量大小，...为可选的其他维度（如空间维度）
            target (torch.Tensor): 目标标签，形状与pred完全一致，
                                 元素值为0或1（分别代表负样本和正样本）

        返回:
            torch.Tensor: 计算得到的损失值。
                         若reduce=True，返回标量损失；否则返回与输入同形状的损失张量
        """
        # 计算二元交叉熵损失（BCE）
        if self.logits:
            # 若输入为logits，使用带logits的BCE函数
            bce_loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction="none"  # 不进行聚合，保留每个元素的损失
            )
        else:
            # 若输入为概率值，直接使用BCE函数
            bce_loss = F.binary_cross_entropy(
                pred, target, reduction="none"  # 不进行聚合，保留每个元素的损失
            )

        # 计算pt值：pt = exp(-bce_loss)，即正确分类的概率
        pt = torch.exp(-bce_loss)

        # 计算alpha_t：根据样本类别动态选择alpha值
        # 正样本（target=1）使用alpha，负样本（target=0）使用1-alpha
        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        # 计算Focal Loss：alpha_t * (1 - pt)^gamma * bce_loss
        focal_loss = alpha_weight * torch.pow((1 - pt), self.gamma) * bce_loss

        # 根据reduce参数决定是否聚合损失
        if self.reduce:
            # 对所有元素取平均值
            focal_loss = torch.mean(focal_loss)

        # 应用整体损失权重
        return focal_loss * self.loss_weight


class FocalLoss(nn.Module):
    """
    Focal Loss 损失函数，用于解决类别不平衡问题
    论文参考: <https://arxiv.org/abs/1708.02002>

    Focal Loss通过引入gamma参数降低易分类样本的权重，
    使模型更专注于难分类样本；同时通过alpha参数处理类别不平衡。
    """
    def __init__(
            self,
            gamma=2.0,
            alpha=0.5,
            reduction="mean",
            loss_weight=1.0,
            ignore_index=-1
    ):
        """
        初始化Focal Loss参数

        参数:
            gamma (float): 聚焦参数，控制难易样本的权重调节程度，gamma≥0。
                           值越大，难分类样本的权重越高。
            alpha (float | list): 类别权重参数，用于处理类别不平衡。
                                 - 若为float，通常用于二分类场景
                                 - 若为list，长度需与类别数相同，分别指定每个类别的权重
            reduction (str): 损失聚合方式，可选"mean"(默认)或"sum"。
                            - "mean": 返回损失的平均值
                            - "sum": 返回损失的总和
            loss_weight (float): 损失的整体权重系数，默认为1.0
            ignore_index (int): 需要忽略的目标值索引，默认为-1。
                               常用于处理标签中的无效值或背景类
        """
        super(FocalLoss, self).__init__()
        # 验证参数合法性
        assert reduction in ("mean", "sum"), \
            f"参数错误: reduction必须为'mean'或'sum'，但得到{reduction}"
        assert isinstance(alpha, (float, list)), \
            f"参数错误: alpha必须为float或list类型， but got {type(alpha)}"
        assert isinstance(gamma, float) and gamma >= 0, \
            f"参数错误: gamma必须为非负float类型， but got {gamma}"
        assert isinstance(loss_weight, float) and loss_weight >= 0, \
            f"参数错误: loss_weight必须为非负float类型， but got {loss_weight}"
        assert isinstance(ignore_index, int), \
            f"参数错误: ignore_index必须为int类型， but got {type(ignore_index)}"

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """
        前向计算损失

        参数:
            pred (torch.Tensor): 模型预测输出，形状为(N, C, ...)，
                                其中N为批量大小，C为类别数，...为可选的空间维度
            target (torch.Tensor): 目标标签，形状与pred的非类别维度一致，即(N, ...)
                                 元素值为类别索引(0 ≤ target[i] ≤ C-1)

        返回:
            torch.Tensor: 计算得到的损失值
        """
        # 调整预测张量形状: [B, C, d1, d2, ...] -> [C, B, d1, d2, ...]
        pred = pred.transpose(0, 1)
        # 展平为二维张量: [C, N] 其中N = B * d1 * d2 * ...
        pred = pred.reshape(pred.size(0), -1)
        # 转置为 [N, C] 便于后续计算
        pred = pred.transpose(0, 1).contiguous()

        # 展平目标张量为一维: [B * d1 * d2 * ...]
        target = target.view(-1).contiguous()

        # 验证预测和目标的样本数量是否一致
        assert pred.size(0) == target.size(0), \
            f"预测与目标形状不匹配: 预测有{pred.size(0)}个样本，目标有{target.size(0)}个样本"

        # 生成有效样本掩码，过滤掉需要忽略的索引
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        # 如果没有有效样本，返回0损失
        if len(target) == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # 获取类别数量
        num_classes = pred.size(1)

        # 将目标标签转换为独热编码格式
        target = F.one_hot(target, num_classes=num_classes)

        # 处理alpha参数，如果是列表则转换为张量
        alpha = self.alpha
        if isinstance(alpha, list):
            # 确保alpha的长度与类别数一致
            assert len(alpha) == num_classes, \
                f"alpha长度与类别数不匹配: alpha有{len(alpha)}个元素，类别数为{num_classes}"
            alpha = torch.tensor(alpha, device=pred.device, dtype=pred.dtype)

        # 计算预测的sigmoid值（将logits转换为概率）
        pred_sigmoid = pred.sigmoid()

        # 确保目标张量与预测张量类型一致
        target = target.type_as(pred)

        # 计算(1 - p_t)，其中p_t是正确类别的预测概率
        # 对于正样本: p_t = pred_sigmoid，所以1 - p_t = 1 - pred_sigmoid
        # 对于负样本: p_t = 1 - pred_sigmoid，所以1 - p_t = pred_sigmoid
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)

        # 计算聚焦权重: alpha_t * (1 - p_t)^gamma
        # alpha_t根据类别取alpha或1-alpha
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(self.gamma)

        # 计算带权重的二元交叉熵损失
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction="none"
        ) * focal_weight

        # 根据指定方式聚合损失
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        # 应用整体损失权重
        return self.loss_weight * loss


class LovaszLoss(nn.Module):
    """
    Lovasz损失函数，用于语义分割任务。
    支持二值分割、多类别分割和多标签分割三种场景。

    论文参考: https://arxiv.org/abs/1705.08790
    """
    def __init__(
            self,
            mode: str = "binary",
            class_seen: Optional[List[int]] = None,
            per_image: bool = False,
            ignore_index: Optional[int] = None,
            loss_weight: float = 1.0
    ):
        """
        初始化Lovasz损失函数

        参数:
            mode: 损失计算模式，可选值为"binary"(二值)、"multiclass"(多类别)、"multilabel"(多标签)
            class_seen: 需要纳入损失计算的类别列表
            per_image: 若为True，将按图像计算损失后取平均；否则按整个批次计算
            ignore_index: 需要忽略的标签值（不参与损失计算）
            loss_weight: 损失的权重因子
        """
        super().__init__()
        # 验证模式的有效性
        assert mode in ["binary", "multiclass", "multilabel"], \
            f"无效的模式: {mode}。必须是'binary', 'multiclass', 'multilabel'中的一种"

        self.mode = mode
        self.class_seen = class_seen
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算Lovasz损失

        参数:
            y_pred: 模型预测结果。形状根据模式有所不同:
                - 二值/多标签模式: (B, H, W) 或 (B, C, H, W)
                - 多类别模式: (B, C, H, W)
            y_true: 真实标签。形状:
                - 二值/多类别模式: (B, H, W)
                - 多标签模式: (B, C, H, W)

        返回:
            计算得到的损失值
        """
        if self.mode in ["binary", "multilabel"]:
            # 二值和多标签模式使用hinge损失
            loss = self._lovasz_hinge(y_pred, y_true)
        elif self.mode == "multiclass":
            # 多类别模式先应用softmax，再计算损失
            y_pred_softmax = F.softmax(y_pred, dim=1)
            loss = self._lovasz_softmax(y_pred_softmax, y_true)

        return loss * self.loss_weight

    def _lovasz_grad(self, gt_sorted: torch.Tensor) -> torch.Tensor:
        """计算Lovasz扩展关于排序误差的梯度"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()  # 真实标签中前景像素的总数
        # 计算累积交集和并集
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        # 计算Jaccard指数
        jaccard = 1.0 - intersection / union

        # 处理多像素情况的梯度计算
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[:-1]
        return jaccard

    def _flatten_binary_scores(self, scores: torch.Tensor, labels: torch.Tensor) -> tuple:
        """展平二值预测和标签，并移除被忽略的像素"""
        scores = scores.view(-1)  # 展平为一维
        labels = labels.view(-1)  # 展平为一维

        # 过滤掉被忽略的像素
        if self.ignore_index is not None:
            valid = labels != self.ignore_index
            return scores[valid], labels[valid]
        return scores, labels

    def _lovasz_hinge_flat(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """针对展平后的预测计算二值Lovasz hinge损失"""
        if len(labels) == 0:
            # 没有有效像素时，返回0（梯度也为0）
            return logits.sum() * 0.0

        # 计算标签的符号（1或-1）
        signs = 2.0 * labels.float() - 1.0
        # 计算误差
        errors = 1.0 - logits * signs
        # 按误差降序排序
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        # 按相同顺序排序真实标签
        gt_sorted = labels[perm.data]

        # 计算损失：排序误差与Lovasz梯度的点积
        return torch.dot(F.relu(errors_sorted), self._lovasz_grad(gt_sorted))

    def _lovasz_hinge(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算二值Lovasz hinge损失"""
        if self.per_image:
            # 按图像计算损失后取平均
            return torch.mean(torch.stack([
                self._lovasz_hinge_flat(*self._flatten_binary_scores(
                    log.unsqueeze(0), lab.unsqueeze(0)
                )) for log, lab in zip(logits, labels)
            ]))
        else:
            # 对整个批次计算损失
            return self._lovasz_hinge_flat(*self._flatten_binary_scores(logits, labels))

    def _flatten_probas(self, probas: torch.Tensor, labels: torch.Tensor) -> tuple:
        """展平多类别情况下的概率和标签"""
        if probas.dim() == 3:
            # 如果缺少通道维度，则添加（适用于二值情况）
            probas = probas.unsqueeze(1)

        # 将通道维度移到最后并展平
        C = probas.size(1)
        probas = torch.movedim(probas, 1, -1).contiguous().view(-1, C)
        labels = labels.view(-1)

        # 过滤掉被忽略的像素
        if self.ignore_index is not None:
            valid = labels != self.ignore_index
            return probas[valid], labels[valid]
        return probas, labels

    def _lovasz_softmax_flat(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """针对展平后的预测计算多类别Lovasz-Softmax损失"""
        if probas.numel() == 0:
            # 没有有效像素时，返回0（梯度也为0）
            return probas * 0.0

        C = probas.size(1)  # 类别数量
        losses = []
        unique_labels = labels.unique()  # 获取所有出现过的标签

        # 对每个出现过的类别计算损失
        for c in unique_labels:
            # 如果指定了需要关注的类别且当前类别不在其中，则跳过
            if self.class_seen is not None and c not in self.class_seen:
                continue

            # 当前类别的前景掩码
            fg = (labels == c).type_as(probas)

            # 如果当前类别没有前景像素，则跳过
            if fg.sum() == 0:
                continue

            # 获取当前类别的预测概率
            class_pred = probas[:, c] if C > 1 else probas[:, 0]

            # 计算误差和梯度
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
            fg_sorted = fg[perm.data]

            # 计算当前类别的损失并添加到列表
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))

        # 平均所有类别的损失
        if losses:
            return torch.mean(torch.stack(losses))
        else:
            return torch.tensor(0.0, device=probas.device)

    def _lovasz_softmax(self, probas: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算多类别Lovasz-Softmax损失"""
        if self.per_image:
            # 按图像计算损失后取平均
            return torch.mean(torch.stack([
                self._lovasz_softmax_flat(*self._flatten_probas(
                    prob.unsqueeze(0), lab.unsqueeze(0)
                )) for prob, lab in zip(probas, labels)
            ]))
        else:
            # 对整个批次计算损失
            return self._lovasz_softmax_flat(*self._flatten_probas(probas, labels))

    def __repr__(self) -> str:
        """返回类的字符串表示，便于调试"""
        return (f"LovaszLoss(mode={self.mode}, per_image={self.per_image}, "
                f"ignore_index={self.ignore_index}, loss_weight={self.loss_weight})")




class SDFLoss(nn.Module):
    """
    基于符号距离函数(SDF)特性的复合损失函数。

    该损失函数通过多个约束项，确保模型预测的SDF满足其数学性质和几何特性，
    包括SDF值准确性、内外区域区分度、法向量一致性和梯度单位模长约束。

    属性:
        sdf_weight: SDF值约束项的权重系数
        inter_weight: 区域区分约束项的权重系数
        normal_weight: 法向量一致性约束项的权重系数
        grad_weight: 梯度模长约束项的权重系数
    """

    def __init__(self,
                 sdf_weight: float = 3e3,
                 inter_weight: float = 1e2,
                 normal_weight: float = 1e2,
                 grad_weight: float = 5e1):
        """初始化SDF损失函数的各项权重。

        参数:
            sdf_weight: SDF值约束项的权重，默认3e3
            inter_weight: 区域区分约束项的权重，默认1e2
            normal_weight: 法向量一致性约束项的权重，默认1e2
            grad_weight: 梯度模长约束项的权重，默认5e1
        """
        super().__init__()
        self.sdf_weight = sdf_weight
        self.inter_weight = inter_weight
        self.normal_weight = normal_weight
        self.grad_weight = grad_weight

    def gradient(self,y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)
        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad

    def forward(self, model_output: dict, gt: dict) -> dict:
        """计算SDF重建的复合损失。

        参数:
            model_output: 模型输出字典，应包含:
                - 'model_in': 输入的空间坐标张量，形状为[B, N, 3]
                - 'model_out': 预测的SDF值，形状为[B, N, 1]
            gt: 真实标签字典，应包含:
                - 'sdf': 真实SDF值，形状为[B, N, 1]，-1表示无标签
                - 'normals': 真实法向量，形状为[B, N, 3]

        返回:
            包含各项损失的字典，键包括:
                - 'sdf_loss': SDF值约束损失
                - 'inter_loss': 区域区分约束损失
                - 'normal_loss': 法向量一致性损失
                - 'grad_loss': 梯度模长约束损失
                - 'total_loss': 总损失（各项加权和）
        """
        # 提取输入和预测值
        gt_sdf = gt['sdf']
        gt_normals = gt['normals']
        coords = model_output['model_in']
        pred_sdf = model_output['model_out']

        # 计算SDF梯度（用于法向量和梯度约束）
        gradient = self.gradient(pred_sdf, coords)

        # 1. SDF值约束：仅对有标签的点(gt_sdf != -1)进行约束
        sdf_constraint = torch.where(
            gt_sdf != -1,
            pred_sdf,
            torch.zeros_like(pred_sdf)
        )
        sdf_loss = torch.abs(sdf_constraint).mean() * self.sdf_weight

        # 2. 区域区分约束：对无标签的点(gt_sdf == -1)，鼓励SDF值远离0
        inter_constraint = torch.where(
            gt_sdf != -1,
            torch.zeros_like(pred_sdf),
            torch.exp(-1e2 * torch.abs(pred_sdf))
        )
        inter_loss = inter_constraint.mean() * self.inter_weight

        # 3. 法向量一致性约束：仅对有标签的点，约束梯度与真实法向量的一致性
        normal_constraint = torch.where(
            gt_sdf != -1,
            1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
            torch.zeros_like(gradient[..., :1])
        )
        normal_loss = normal_constraint.mean() * self.normal_weight

        # 4. 梯度模长约束：所有点的梯度模长应接近1
        grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
        grad_loss = grad_constraint.mean() * self.grad_weight

        # 总损失
        total_loss = sdf_loss + inter_loss + normal_loss + grad_loss

        return {
            'sdf_loss': sdf_loss,
            'inter_loss': inter_loss,
            'normal_loss': normal_loss,
            'grad_loss': grad_loss,
            'total_loss': total_loss
        }


import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """二分类焦点损失函数，适用于类别不平衡场景

    通过降低易分类样本的权重，使得模型在训练时更关注难分类样本
    论文: <https://arxiv.org/abs/1708.02002>

    Attributes:
        gamma (float): 调节因子，用于调整困难样本的权重
        alpha (float): 类别权重平衡参数，范围(0,1)
        logits (bool): 输入是否为logits（启用sigmoid）
        reduce (bool): 是否对损失进行均值降维
        loss_weight (float): 损失权重系数
    """

    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """前向计算函数

        Args:
            pred (torch.Tensor): 模型预测值，形状为(N)
            target (torch.Tensor): 真实标签，形状为(N)。若为类别索引，每个值应为0或1；
                若为概率值，形状需与pred一致

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)  # 计算概率p_t
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)  # 动态alpha平衡
        focal_loss = alpha * (1 - pt) ** self.gamma * bce  # 焦点损失公式

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


class FocalLoss(nn.Module):
    """多分类焦点损失函数，支持类别忽略

    适用于多分类任务的焦点损失实现，支持指定忽略类别
    论文: <https://arxiv.org/abs/1708.02002>

    Attributes:
        gamma (float): 困难样本调节因子
        alpha (float/list): 类别权重，可设置为列表指定各类别权重
        reduction (str): 损失降维方式，可选'mean'或'sum'
        loss_weight (float): 损失项的缩放权重
        ignore_index (int): 需要忽略的类别索引
    """

    def __init__(
            self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "reduction应为'mean'或'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "alpha应为float或list类型"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """前向计算函数

        Args:
            pred (torch.Tensor): 模型预测值，形状为(N, C)，C为类别数
            target (torch.Tensor): 真实标签。若为类别索引，形状为(N)，值范围为0~C-1；
                若为概率，形状需与pred一致

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        # 调整张量形状
        pred = pred.transpose(0, 1).reshape(pred.size(0), -1).transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()

        # 验证形状一致性
        assert pred.size(0) == target.size(0), "预测与标签形状不匹配"

        # 过滤忽略索引
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)  # 转换为one-hot编码

        # 处理alpha参数
        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)

        # 计算焦点权重
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(self.gamma)

        # 计算加权损失
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * focal_weight

        # 降维处理
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return self.loss_weight * loss


class DiceLoss(nn.Module):
    """Dice系数损失函数，适用于分割任务

    通过测量预测与标签的相似度进行优化，尤其适用于类别不平衡场景
    论文: <https://arxiv.org/abs/1606.04797>

    Attributes:
        smooth (float): 平滑系数，防止除零
        exponent (int): 指数参数，控制计算方式
        loss_weight (float): 损失项的缩放权重
        ignore_index (int): 需要忽略的类别索引
    """

    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """前向计算函数

        Args:
            pred (torch.Tensor): 模型预测值，形状为(B, C, d1, d2, ...)
            target (torch.Tensor): 真实标签，形状为(B, d1, d2, ...)

        Returns:
            torch.Tensor: 计算得到的损失值
        """
        # 调整张量形状
        pred = pred.transpose(0, 1).reshape(pred.size(0), -1).transpose(0, 1).contiguous()
        target = target.view(-1).contiguous()

        # 验证形状一致性
        assert pred.size(0) == target.size(0), "预测与标签形状不匹配"

        # 过滤忽略索引
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        # 计算softmax概率
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        # 逐类别计算Dice损失
        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                numerator = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                denominator = torch.sum(pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)) + self.smooth
                dice_loss = 1 - numerator / denominator
                total_loss += dice_loss

        # 平均损失并加权
        loss = total_loss / num_classes
        return self.loss_weight * loss



def dice_loss_multi_classes(input, target, epsilon=1e-5, weight=None):
    r"""
    多类别Dice损失函数，用于语义分割任务，计算每个类别的Dice系数并转化为损失。
    修改自：https://github.com/wolny/pytorch-3dunet/blob/.../losses.py 的compute_per_channel_dice

    参数：
        input (Tensor): 模型预测的输出，形状为(batch_size, num_classes, [depth, height, width])
        target (Tensor): 真实标签的one-hot编码，形状需与input相同
        epsilon (float): 平滑系数，防止分母为零，默认为1e-5
        weight (Tensor, optional): 各类别的权重，形状应为(num_classes, )

    返回：
        Tensor: 每个类别的Dice损失，形状为(num_classes, )

    注意：
        - 输入和目标的维度必须完全一致
        - 本实现暂未使用weight参数，如需加权需后续手动处理
    """
    # 校验输入形状一致性
    assert input.size() == target.size(), "'input'和'target'的维度必须完全相同"

    # 调整维度顺序，将类别通道移至第0维，便于逐类别计算
    # 原维度假设为(batch, num_classes, ...)，调整后为(num_classes, batch, ...)
    axis_order = (1, 0) + tuple(range(2, input.dim()))
    input = input.permute(axis_order)
    target = target.permute(axis_order)

    # 转换目标类型为float，确保与预测值类型匹配
    target = target.float()

    # 计算逐类别的Dice系数
    # 分子：2 * 预测与目标的逐元素乘积之和（按批次和空间维度求和）
    # 分母：预测值的平方和 + 目标值的平方和 + 平滑项
    per_channel_dice = (2 * torch.sum(input * target, dim=1) + epsilon) / \
                       (torch.sum(input * input, dim=1) + torch.sum(target * target, dim=1) + 1e-4 + epsilon)

    # 将Dice系数转换为损失（1 - Dice）
    loss = 1. - per_channel_dice

    # 若需类别加权，可在此处添加权重计算（当前实现未使用weight参数）
    # if weight is not None:
    #     loss = loss * weight

    return loss






class GramLoss(nn.Module):
    """Gram矩阵损失的实现

    该损失函数通过最小化学生模型与教师模型输出特征的Gram矩阵差异，
    使学生模型学习到特征间的相关性结构，实现知识蒸馏。
    Gram矩阵反映了特征之间的关联强度，是一种结构化知识的表示。
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/gram_loss.py
    """

    def __init__(
            self,
            apply_norm=True,
            img_level=True,
            remove_neg=True,
            remove_only_teacher_neg=False,
    ):
        """初始化GramLoss模块

        Args:
            apply_norm: 是否对特征进行L2归一化。若为True，内积等价于余弦相似度
            img_level: 是否在单张图像级别计算Gram矩阵
                       - True: 输入特征形状为(B, N, D)，每张图像单独计算Gram矩阵
                       - False: 输入特征展平为(B*N, D)，计算全局Gram矩阵
            remove_neg: 是否将Gram矩阵中的负值置为0，仅保留正相关性
            remove_only_teacher_neg: 仅将教师Gram矩阵的负值置为0，并同步清零学生矩阵对应位置
                                     注意：与remove_neg互斥，只能有一个为True

        Raises:
            AssertionError: 当remove_neg和remove_only_teacher_neg同时为True或同时为False时触发
        """
        super().__init__()

        # 采用MSE损失计算两个Gram矩阵的差异
        self.mse_loss = torch.nn.MSELoss()

        # 保存超参数
        self.apply_norm = apply_norm
        self.remove_neg = remove_neg
        self.remove_only_teacher_neg = remove_only_teacher_neg

        # 确保两个去负值参数不同时生效
        if self.remove_neg or self.remove_only_teacher_neg:
            assert self.remove_neg != self.remove_only_teacher_neg, \
                "remove_neg和remove_only_teacher_neg不能同时为True或同时为False"

    def forward(self, output_feats, target_feats, img_level=True):
        """计算输入特征与目标特征的Gram矩阵之间的MSE损失

        Args:
            output_feats: 学生模型输出的特征张量
                          形状: (B, N, dim) 若img_level=True；(B*N, dim) 若img_level=False
            target_feats: 教师模型输出的目标特征张量，形状同output_feats
            img_level: 是否在图像级别计算Gram矩阵（覆盖类初始化时的同名参数）

        Returns:
            loss: 标量损失值，学生与教师Gram矩阵的MSE差异
        """

        # 当在图像级别计算时，确保输入是3维张量 (B, N, dim)
        if img_level:
            assert len(target_feats.shape) == 3 and len(output_feats.shape) == 3, \
                "img_level=True时，输入特征必须为3维张量 (B, N, dim)"

        # 转换为float类型以保证计算精度
        output_feats = output_feats.float()
        target_feats = target_feats.float()

        # 对教师特征进行归一化（若启用）
        if self.apply_norm:
            target_feats = F.normalize(target_feats, dim=-1)  # 按特征维度归一化

        # 若不在图像级别计算且特征是3维，则展平为2维 (B*N, dim)
        if not img_level and len(target_feats.shape) == 3:
            target_feats = target_feats.flatten(0, 1)  # 合并批次和特征数维度

        # 计算教师特征的Gram矩阵（特征相关性矩阵）
        # 形状: (B, N, N) 若img_level=True；(B*N, B*N) 若img_level=False
        target_sim = torch.matmul(target_feats, target_feats.transpose(-1, -2))

        # 对学生特征进行归一化（若启用）
        if self.apply_norm:
            output_feats = F.normalize(output_feats, dim=-1)  # 按特征维度归一化

        # 若不在图像级别计算且特征是3维，则展平为2维 (B*N, dim)
        if not img_level and len(output_feats.shape) == 3:
            output_feats = output_feats.flatten(0, 1)  # 合并批次和特征数维度

        # 计算学生特征的Gram矩阵（特征相关性矩阵）
        student_sim = torch.matmul(output_feats, output_feats.transpose(-1, -2))

        # 处理Gram矩阵中的负值
        if self.remove_neg:
            # 将学生和教师Gram矩阵中的所有负值置为0
            target_sim[target_sim < 0] = 0.0
            student_sim[student_sim < 0] = 0.0

        elif self.remove_only_teacher_neg:
            # 仅将教师矩阵的负值置为0，并同步清零学生矩阵中对应位置的负值
            target_sim[target_sim < 0] = 0.0
            student_sim[(student_sim < 0) & (target_sim < 0)] = 0.0

        # 计算两个Gram矩阵的MSE损失
        return self.mse_loss(student_sim, target_sim)




class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko熵损失正则化器
    源自论文：Sablayrolles et al. - 2018 - Spreading vectors for similarity search
    作用：通过最大化特征分布的熵，使向量在特征空间中更分散，提升相似度搜索性能
    https://github.com/facebookresearch/dinov3/blob/main/dinov3/loss/koleo_loss.py
    """

    def __init__(self):
        super().__init__()
        # 初始化 pairwise 距离计算器，计算L2距离（欧氏距离）
        # 参数2表示L2范数，eps为数值稳定性的微小值
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """为L2归一化的向量找到成对的最近邻
        使用PyTorch实现而非Faiss，以保持在GPU上运行

        Args:
            x: 形状为 (N, D) 的张量，N为样本数，D为特征维度（已L2归一化）

        Returns:
            indices: 形状为 (N,) 的张量，每个元素表示对应样本的最近邻索引
        """
        # 计算向量间的内积（对于L2归一化向量，内积等价于余弦相似度，与距离成反比）
        dots = torch.mm(x, x.t())  # 结果形状为 (N, N)

        n = x.shape[0]  # 样本数量
        # 技巧：将内积矩阵的对角线元素（自身与自身的内积）设为-1
        # 避免将样本自身识别为最近邻（自身内积为1，是最大的）
        dots.view(-1)[:: (n + 1)].fill_(-1)

        # 找到每个样本的最大内积对应的索引（即最近邻）
        # 最大内积对应最小距离（因为内积与距离成反比）
        _, indices = torch.max(dots, dim=1)
        return indices

    def forward(self, student_output, eps=1e-8):
        """前向传播计算损失

        Args:
            student_output (BxD): 学生模型的backbone输出特征，B为批次大小，D为特征维度
            eps: 数值稳定性参数，防止log(0)或除以零

        Returns:
            loss: 标量损失值，越小表示特征分布越分散
        """
        # 禁用自动混合精度（确保数值稳定性，尤其是在计算log时）
        with torch.autocast("cuda", enabled=False):
            # 对学生输出特征进行L2归一化，使后续内积计算等价于余弦相似度
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)

            # 找到每个特征向量的最近邻索引
            indices = self.pairwise_NNs_inner(student_output)

            # 计算每个向量与其最近邻的欧氏距离
            # student_output[indices] 取出每个样本的最近邻特征，形状为 (B, D)
            # pdist计算对应位置的距离，结果形状为 (B,)
            distances = self.pdist(student_output, student_output[indices])

            # 损失定义为：负的距离对数的平均值
            # 最小化此损失会迫使距离增大，从而使特征分布更分散（熵更大）
            loss = -torch.log(distances + eps).mean()

        return loss





class SSIMUtil(nn.Module):
    """
    结构相似性指数(SSIM)工具类，包含SSIM计算、损失函数和结构一致性度量

    支持两种计算模式：
    - 高斯滤波模式：使用高斯窗口计算，精度高
    - 平均池化模式：使用平均池化替代高斯滤波，计算更快
    """
    def __init__(self, window_size=11, size_average=True, mode='gaussian'):
        """
        初始化SSIM工具类

        参数:
            window_size: 窗口大小
            size_average: 是否对结果进行平均
            mode: 计算模式，'gaussian'或'average'
        """
        super(SSIMUtil, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.mode = mode
        self.channel = 1
        self.window = self._create_gaussian_window(window_size) if mode == 'gaussian' else None

        # 定义平均池化层（用于average模式）
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 稳定常数
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def _create_gaussian_window(self, window_size, sigma=1.5):
        """创建高斯窗口"""
        gauss = torch.Tensor([torch.exp(-(x - window_size // 2) **2 / float(2 * sigma** 2))
                              for x in range(window_size)])
        gauss = gauss / gauss.sum()

        # 扩展为2D窗口
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window

    def _adjust_window(self, img):
        """根据输入图像调整窗口以匹配通道数和设备"""
        _, channel, _, _ = img.size()

        if channel != self.channel or self.window.device != img.device:
            # 扩展窗口以匹配通道数
            window = self.window.expand(channel, 1, self.window_size, self.window_size).contiguous()
            # 确保窗口在正确的设备上
            window = window.to(img.device)
            self.window = window
            self.channel = channel

        return self.window

    def ssim_gaussian(self, img1, img2):
        """使用高斯滤波计算SSIM"""
        window = self._adjust_window(img1)
        channel = self.channel

        # 计算均值
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        # 计算均值的平方和乘积
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2) /
                    ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)))

        # 平均处理
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def ssim_average(self, img1, img2):
        """使用平均池化计算SSIM（快速版）"""
        # 计算均值
        mu_x = self.avg_pool(img1)
        mu_y = self.avg_pool(img2)

        # 计算均值的平方和乘积
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        # 计算方差和协方差
        sigma_x = self.avg_pool(img1 * img1) - mu_x_sq
        sigma_y = self.avg_pool(img2 * img2) - mu_y_sq
        sigma_xy = self.avg_pool(img1 * img2) - mu_x_mu_y

        # 计算SSIM
        ssim_n = (2 * mu_x_mu_y + self.C1) * (2 * sigma_xy + self.C2)
        ssim_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)
        ssim_map = ssim_n / ssim_d

        return torch.clamp(ssim_map, 0, 1).mean()

    def forward(self, img1, img2):
        """
        计算SSIM值

        参数:
            img1: 输入图像1
            img2: 输入图像2

        返回:
            ssim值
        """
        if self.mode == 'gaussian':
            return self.ssim_gaussian(img1, img2)
        else:
            return self.ssim_average(img1, img2)

    def ssim_loss(self, img1, img2):
        """计算SSIM损失 (1 - SSIM)"""
        if self.mode == 'gaussian':
            return 1 - (1 + self.ssim_gaussian(img1, img2)) / 2
        else:
            return torch.clamp((1 - self.ssim_average(img1, img2)) / 2, 0, 1)

    def saliency_structure_consistency(self, img1, img2):
        """计算显著性结构一致性（SSIM均值）"""
        return self.forward(img1, img2).mean()






class ContourLoss(torch.nn.Module):
    """
    ContourLoss（轮廓损失）
        核心思想：同时优化区域一致性和轮廓平滑性，特别适合需要精确边界的分割任务（如医学图像分割、目标边缘分割）。
        组成部分：
            长度项（length term）：通过计算预测图的水平和垂直梯度，惩罚剧烈变化的区域，鼓励生成平滑连续的轮廓。
            区域项（region term）：惩罚前景区域（target=1）和背景区域（target=0）的预测误差，确保区域内部的一致性
    """
    def __init__(self):
        super(ContourLoss, self).__init__()

    def forward(self, pred, target, weight=10):
        '''
        target, pred: tensor of shape (B, C, H, W), where target[:,:,region_in_contour] == 1,
                        target[:,:,region_out_contour] == 0.
        weight: scalar, length term weight.
        '''
        # length term
        delta_r = pred[:,:,1:,:] - pred[:,:,:-1,:] # horizontal gradient (B, C, H-1, W)
        delta_c = pred[:,:,:,1:] - pred[:,:,:,:-1] # vertical gradient   (B, C, H,   W-1)

        delta_r    = delta_r[:,:,1:,:-2]**2  # (B, C, H-2, W-2)
        delta_c    = delta_c[:,:,:-2,1:]**2  # (B, C, H-2, W-2)
        delta_pred = torch.abs(delta_r + delta_c)

        epsilon = 1e-8 # where is a parameter to avoid square root is zero in practice.
        length = torch.mean(torch.sqrt(delta_pred + epsilon)) # eq.(11) in the paper, mean is used instead of sum.

        c_in  = torch.ones_like(pred)
        c_out = torch.zeros_like(pred)

        region_in  = torch.mean( pred     * (target - c_in )**2 ) # equ.(12) in the paper, mean is used instead of sum.
        region_out = torch.mean( (1-pred) * (target - c_out)**2 )
        region = region_in + region_out

        loss =  weight * length + region

        return loss




class StructureLoss(torch.nn.Module):
    """
     StructureLoss（结构损失）
        核心思想：结合加权二元交叉交叉熵（WBCE）和加权 IoU（WIoU），平衡前景背景不平衡问题，同时关注结构一致性。
        组成部分：
            加权 BCE（wbce）：通过权重矩阵weit（基于目标区域的局部均值）赋予前景区域更高的权重，缓解类别不平衡。
            加权 IoU（wiou）：在 IoU 计算中引入相同权重矩阵，更关注目标结构区域的重叠度。
    """
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, target):
        weit  = 1+5*torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15)-target)
        wbce  = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

        pred  = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou  = 1-(inter+1)/(union-inter+1)

        return (wbce+wiou).mean()

class IoULoss(torch.nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, pred, target):
        b = pred.shape[0]
        IoU = 0.0
        for i in range(0, b):
            # compute the IoU of the foreground
            Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
            Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
            IoU1 = Iand1 / Ior1
            # IoU loss is (1-IoU1)
            IoU = IoU + (1-IoU1)
        # return IoU/b
        return IoU
class PatchIoULoss(torch.nn.Module):
    """
    PatchIoULoss（补丁 IoULoss）
        核心思想：将图像分割为多个子 patch 计算 IoU 损失，增强对局部细节的关注。
        实现方式：
            将图像按固定大小（64×64）划分为多个子区域。
            对每个子区域计算 IoU 损失并累加。
        适用场景：
            大尺寸图像分割（如遥感图像、高分辨率医学影像）。
            需要关注局部细节的任务（如病灶分割中的小肿瘤区域）。

    """
    def __init__(self):
        super(PatchIoULoss, self).__init__()
        self.iou_loss = IoULoss()

    def forward(self, pred, target):
        win_y, win_x = 64, 64
        iou_loss = 0.
        for anchor_y in range(0, target.shape[0], win_y):
            for anchor_x in range(0, target.shape[1], win_y):
                patch_pred = pred[:, :, anchor_y:anchor_y+win_y, anchor_x:anchor_x+win_x]
                patch_target = target[:, :, anchor_y:anchor_y+win_y, anchor_x:anchor_x+win_x]
                patch_iou_loss = self.iou_loss(patch_pred, patch_target)
                iou_loss += patch_iou_loss
        return iou_loss