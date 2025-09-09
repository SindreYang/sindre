from functools import partial

import numpy as np
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from sindre.ai.pointcloud_utils.point_transformerV3 import PointTransformerV3, Point, PointSequential, \
    SerializedPooling, Block
from addict import Dict
def map_features(info: str):
    """
    将编码转换为特定条件

    例如：
    "32normal" 代表找32号正常牙

    Args:
        info: 字符串，如"11inlay"

    Returns:
        numpy数组: 形状为(32, 3)的矩阵，每行代表一颗牙齿的条件
                  [象限, 牙位, 状态]，未指定的牙齿为-1
    """
    # ISO标准牙齿编号到矩阵索引的映射
    # 牙齿编号: 11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,
    #           31,32,33,34,35,36,37,38,41,42,43,44,45,46,47,48
    iso_tooth_mapping = {
        '11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 5, '17': 6, '18': 7,
        '21': 8, '22': 9, '23': 10, '24': 11, '25': 12, '26': 13, '27': 14, '28': 15,
        '31': 16, '32': 17, '33': 18, '34': 19, '35': 20, '36': 21, '37': 22, '38': 23,
        '41': 24, '42': 25, '43': 26, '44': 27, '45': 28, '46': 29, '47': 30, '48': 31
    }
    # 牙位映射（1-8号牙）
    tooth_map = {
        '1': 0,  # 中切牙
        '2': 1,  # 侧切牙
        '3': 2,  # 尖牙
        '4': 3,  # 第一前磨牙
        '5': 4,  # 第二前磨牙
        '6': 5,  # 第一磨牙
        '7': 6,  # 第二磨牙
        '8': 7,  # 第三磨牙
    }

    # 象限映射
    quadrant_map = {
        '1': 0,  # 右上象限
        '2': 1,  # 左上象限
        '3': 2,  # 左下象限
        '4': 3,  # 右下象限
    }

    # 牙齿状态映射
    status_map = {
        'normal': 0,    # 正常牙
        'inlay': 1,     # 嵌体
        'abutment': 2,  # 基座（全冠，内冠)
        'veneer': 3,    # 贴面
    }


    # 只做二分类
    quadrant_idx = quadrant_map[info[0]]
    tooth_idx = tooth_map[info[1]]
    status_idx = status_map[info[2:]]
    return np.array([quadrant_idx, tooth_idx, status_idx])


    # # 初始化结果矩阵，全部为-1
    # code_all = np.full((32, 3), -1, dtype=int)
    #
    # # 如果没有输入信息，返回全部为-1的矩阵
    # if not info or info.strip() == "":
    #     return code_all

    # # 分割输入字符串
    # code_list = info.split(",")
    #
    # for code in code_list:
    #     code = code.strip()
    #     if len(code) < 3:
    #         continue
    #     # 提取象限、牙位和状态
    #     quadrant_char = code[0]
    #     tooth_char = code[1]
    #     status_str = code[2:]
    #     # 设置编码
    #     row_idx =iso_tooth_mapping[code[:2]]
    #     quadrant_idx = quadrant_map[quadrant_char]
    #     tooth_idx = tooth_map[tooth_char]
    #     status_idx = status_map[status_str]
    #     code_all[row_idx] = [quadrant_idx, tooth_idx, status_idx]

    # return code_all


class embed_condition(nn.Module):
    def __init__(self):
        super(embed_condition, self).__init__()
        embed = nn.Embedding(3,512,)

    def forward(self, input):
        pass


from sindre.ai.pointcloud_utils.point_transformerV3 import Embedding
class PTV3(nn.Module):
    def __init__(self,
                 in_channels=6,
                 stride=(2, 2, 2, 2),
                 enc_depths=(2, 2, 2, 6, 2),
                 enc_channels=(32, 64, 128, 256, 512),
                 enc_num_head=(2, 4, 8, 16, 32),
                 enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                 ):
        super(PTV3, self).__init__()

        # 构建初始函数
        bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer = nn.LayerNorm
        act_layer = nn.GELU
        self.order =("z", "z-trans", "hilbert", "hilbert-trans")

        # 构建特征嵌入器
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # 根据路径长度分割等长
        enc_drop_path = []
        for x in torch.linspace(0, 0.3, sum(enc_depths)):
            enc_drop_path.append(x.item())

        # 编码容器
        self.enc = PointSequential()
        for s in range(len(enc_depths)):
            enc_drop_path_ = enc_drop_path[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            enc = PointSequential()
            # 从第二层添加池化层
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        traceable=True,
                    ),
                    name="down",
                )
            # 每层按照约定添加卷积模块
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        attn_drop=0.0,
                        proj_drop=0.0,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=True,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=False,
                        enable_flash=True,
                        upcast_attention=False,
                        upcast_softmax=False,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")








class mv_pcd(nn.Module):
    def __init__(self,):
        super(mv_pcd, self).__init__()
        self.image_model =UNet(
            spatial_dims=2,
            in_channels=5,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.pcd_model = PointTransformerV3(in_channels=6 ,
                                            enable_flash=True,
                                            shuffle_orders=False,
                                            ).cuda()

        # 分离编码器和解码器
        self.image_encoder = torch.nn.Sequential(*list(self.image_model.model.children())[:2])
        self.image_decoder = torch.nn.Sequential(list(self.image_model.model.children())[2])
        self.pcd_encoder =self.pcd_model.enc
        self.pcd_decoder =self.pcd_model.dec

    def create_pointdict(self, x, p):
        # 构造符合Pointcept要求的data_dict字典
        data_dict = Dict()  # 使用自定义字典对象
        # 展平特征矩阵: (B, N, C) -> (B*N, C)
        data_dict.feat = torch.flatten(x, start_dim=0, end_dim=1)
        # 生成批次索引: 每个点对应所属的样本批次
        data_dict.batch = torch.arange(x.shape[0]).repeat_interleave(x.shape[1]).to(x.device)
        # 构造三维坐标:  (B, N, 3) -> (B*N, 3)
        data_dict.coord = torch.flatten(p, start_dim=0, end_dim=1)
        # 设置体素化网格尺寸
        data_dict.grid_size = 0.01
        # 点云处理流程 ----------------------------------------------------------
        # 初始化点云处理对象
        point = Point(data_dict)
        # 序列化处理：对点云进行有序化排列
        # - order: 排序依据的维度顺序
        # - shuffle_orders: 是否启用随机维度顺序（数据增强）
        point.serialization(order=self.pcd_model.order, shuffle_orders=self.pcd_model.shuffle_orders)
        # 稀疏化处理：将点云转换为稀疏张量格式
        point.sparsify()
        # 特征变换流程 ----------------------------------------------------------
        # 嵌入层：特征空间映射
        point = self.pcd_model.embedding(point)
        return point

    def forward(self, x,p):
        # unet_encoder = self.encoder(x)
        # unet_decoder = self.decoder(unet_encoder)
        return self.pcd_model(x,p)




# 测试函数
if __name__ == "__main__":
    m= PTV3()
    # # 测试前向传播
    # x = torch.randn(2, 10000, 6).cuda()  # 批次x点数x特征
    # p = torch.randn(2, 10000, 3).cuda()  # 坐标
    # output = m(x, p)
    # print(f"output shape: {output.feat.shape}")
    #
    # # 编码器前向传播
    # features = m.create_pointdict(x,p)
    # features = m.pcd_encoder(features)
    # output = m.pcd_decoder(features)
    # print(f"\nEncoder output shape: {output.feat.shape,features.feat.shape}")



