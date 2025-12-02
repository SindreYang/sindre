from sindre.ai.pointcloud_utils.layers import PointNetFeaturePropagation, PointNetSetAbstraction, PointNetSetAbstractionMsg
import torch.nn as nn
import torch


class pointnet2_ssg(nn.Module):
    def __init__(self,  in_channel=9,part_seg_class=None):
        super(pointnet2_ssg, self).__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, 
                                          radius=0.2, 
                                          nsample=32, 
                                          in_channel=3+in_channel, 
                                          mlp=[64, 64, 128], 
                                          group_all=False
                                          )
        self.sa2 = PointNetSetAbstraction(npoint=128,
                                          radius=0.4, 
                                          nsample=64, 
                                          in_channel=128 + 3, 
                                          mlp=[128, 128, 256], 
                                          group_all=False
                                          )
        self.sa3 = PointNetSetAbstraction(npoint=None,
                                          radius=None, 
                                          nsample=None, 
                                          in_channel=256 + 3, 
                                          mlp=[256, 512, 1024], 
                                          group_all=True
                                          )
        self.fp3 = PointNetFeaturePropagation(in_channel=1280, 
                                              mlp=[256, 256]
                                              )
        self.fp2 = PointNetFeaturePropagation(in_channel=384,
                                              mlp=[256, 128]
                                              )
        self.fp1 = PointNetFeaturePropagation(in_channel=128,
                                                mlp=[128, 128, 128]
                                                )
        if part_seg_class is not None:
            self.part_seg_class=part_seg_class
            self.register_buffer('hot_labels', torch.eye(part_seg_class))
            self.fp1 = PointNetFeaturePropagation(in_channel=128+part_seg_class+3+in_channel,
                                                mlp=[128, 128, 128]
                                                )

    def forward(self, xyz, cls_label=None,cls_label_one_hot=None):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        cls_features=l3_points
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if cls_label is not None and self.part_seg_class is not None:
            if cls_label_one_hot is None:
                cls_label_one_hot = self.hot_labels[cls_label,].unsqueeze(-1).repeat(1,1,N)
            l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        else:
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
            
        seg_features = l0_points
        return {
            "cls_features":cls_features,
            "seg_features":seg_features
            }
        
    
    


class pointnet2_msg(nn.Module):
    def __init__(self, in_channel=9,part_seg_class=None):
        super(pointnet2_msg, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(npoint=512, 
                                             radius_list=[0.1, 0.2, 0.4], 
                                             nsample_list=[32, 64, 128],
                                             in_channel=in_channel, 
                                             mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
                                             )
        self.sa2 = PointNetSetAbstractionMsg(npoint=128,
                                             radius_list=[0.4,0.8],
                                             nsample_list=[64, 128],
                                             in_channel=128+128+64,
                                             mlp_list=[[128, 128, 256], [128, 196, 256]]
                                             )
        self.sa3 = PointNetSetAbstraction(npoint=None, 
                                          radius=None,
                                          nsample=None,
                                          in_channel=512 + 3,
                                          mlp=[256, 512, 1024],
                                          group_all=True
                                          )
        self.fp3 = PointNetFeaturePropagation(in_channel=1536,
                                              mlp=[256, 256]
                                              )
        self.fp2 = PointNetFeaturePropagation(in_channel=576,
                                              mlp=[256, 128]
                                              )
        
        self.fp1 = PointNetFeaturePropagation(in_channel=128,
                                              mlp=[128, 128]
                                              )
        if part_seg_class is not None:
            self.part_seg_class=part_seg_class
            self.register_buffer('hot_labels', torch.eye(part_seg_class))
            self.fp1 = PointNetFeaturePropagation(in_channel=128+in_channel+part_seg_class+3,
                                              mlp=[128, 128]
                                              )
            


    def forward(self, xyz, cls_label=None):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        cls_features=l3_points
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if cls_label is not None and self.part_seg_class is not None:
            cls_label_one_hot = self.hot_labels[cls_label,].unsqueeze(-1).repeat(1,1,N)
            l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
           
        else:
            l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)
        seg_features = l0_points
        return {
                "cls_features":cls_features,
                "seg_features":seg_features
                }

    



# class pointnet2_ssg_wtih_R(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = pointnet2_ssg(9) 
#         self.mlp_R = nn.Sequential(
#             nn.Conv1d(128, 256, 1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Conv1d(256, 128*128, 1)
#         )

#         self.mlp_seg= nn.Sequential(
#             nn.Conv1d(128+9, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, 1, 1),# 16_tooth 0_gum
#             nn.Sigmoid()
#         )
        
#     def forward(self, pcd,num_class):
#         # 处理输入
#         B,N,C = pcd.shape
#         device = pcd.device
#         pcd = pcd[...,:9].transpose(-1, 1)
#         multi_hot_labels = num_class.repeat(1,1,N)

#         # 获取特征
#         feat = self.model(pcd,multi_hot_labels)["seg_features"]
#         feat_max = torch.max(feat,dim=-1)[0]
#         feat_R = self.mlp_R(feat_max.unsqueeze(-1)).reshape(-1,128,128)+torch.eye(128, device=device)
    

#         # 旋转特征
#         feat = torch.matmul( feat.transpose(-1,1) ,feat_R.transpose(-1,1) ).transpose(-1,1) 

#         # 汇总特征
#         feat_seg = torch.cat([feat,pcd],dim=1)

#         # 预测16个牙位的分割
#         pre_labels = self.mlp_seg(feat_seg).transpose(-1, 1)
        
#         return {"pre_labels": pre_labels}


if __name__ == '__main__':

    xyz = torch.randn(6, 9, 6666)
    # 假设有3个类别，每个类别有n个部件，共计8个部件，3个类别---3个类别作为输入，net预测8个类别
    part_classes = {'Airplane': [0, 1, 2, 3],'Bag': [4, 5],  'Cap': [6, 7]}
    num_classes = len(part_classes.keys())
    select_classes = torch.randint(0,num_classes,(6,)) #每个样本一个类别
    print(select_classes,num_classes,select_classes.shape)
    
    # 测试msg
    model = pointnet2_msg(9)
    out = model(xyz)
    print("Msg Classification prediction shape:", out["cls_features"].shape)
    print("Msg Segmentation prediction shape:", out["seg_features"].shape)
    
    # 测试ssg
    model = pointnet2_ssg(9)
    out = model(xyz)
    print("ssg Classification prediction shape:", out["cls_features"].shape)
    print("ssg Segmentation prediction shape:", out["seg_features"].shape)
    
    # 测试ssg part
    model = pointnet2_ssg(9,part_seg_class=num_classes)
    out = model(xyz,select_classes)
    print("ssg part Classification prediction shape:", out["cls_features"].shape)
    print("ssg part Segmentation prediction shape:", out["seg_features"].shape)
    
    
    # 测试msg part
    model = pointnet2_msg(9,part_seg_class=num_classes)
    out = model(xyz,select_classes)
    print("msg part Classification prediction shape:", out["cls_features"].shape)
    print("msg  part Segmentation prediction shape:", out["seg_features"].shape)
    
    
    
    

    
    
    

