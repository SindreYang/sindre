import math
from vedo import *
from sindre.utils3d import *
from sindre.utils3d.dental_tools import *

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sindre.utils3d.networks.pointnet2 import PointNetSetAbstraction,PointNetFeaturePropagation


class pointnet2(nn.Module):
    def __init__(self,  in_channel=9):
        super(pointnet2, self).__init__()
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
        class_num = 14
        self.fp1 = PointNetFeaturePropagation(in_channel=128+3+class_num+in_channel,
                                            mlp=[128, 128, 128]
                                            )

    def forward(self, xyz,cls_label_one_hot=None):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot,l0_xyz,l0_points],1), l1_points)
        seg_features = l0_points
        return {
            "seg_features":seg_features
            }
        
    


def load_mesh(sm_path: str) :
    mesh= SindreMesh()
    mesh.load(sm_path)
    # 转换成1-18,并保持原有颜色
    mesh.vertex_labels=convert_fdi2idx(mesh.vertex_labels)
    # mesh.set_vertex_labels(convert_fdi2idx(mesh.vertex_labels))
    # print(np.unique( mesh.vertex_labels))
    # mesh.show()
    # return
    
    # 定义变换
    Normalize_std = Normalize_np()


    # 标准化
    vertices =Normalize_std(mesh.vertices)
    
    # 组合特征
    feat = np.hstack([vertices,mesh.vertex_normals,mesh.vertex_colors[...,:3]/255])


    # 选取任意标签
    class_id = [1,6,3,12]
    vertex_labels = mesh.vertex_labels.copy()
    labels = np.zeros_like(vertex_labels)
    for i in class_id:
        idx = np.where(vertex_labels==i)[0]
        labels[idx] = 1


    # 转换为multi_hot_labels
    num_class = np.zeros(14)
    num_class[class_id,]=1


    

    return {
            "feat":feat ,
            "face":mesh.faces,
            "labels":labels,
            "num_class":num_class.reshape(1,-1)
        }




class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def dice_loss(self,input, target):
     
        # 集合损失： 输入，目标要求是0，1
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1. - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


        
    def cd_loss(self,pcd,pre_labels,tgt_labels):
        # 期望分割下来的点云与目标点云一致
        from pytorch3d.loss import chamfer_distance
        sum_loss =0
        for B in range(pcd.shape[0]):
            # 每个牙点云
            tgt_idx = (tgt_labels[B]>0).squeeze(-1)
            pre_idx = (pre_labels[B]>0.5).squeeze(-1)
            cd_loss =  chamfer_distance( pcd[B][pre_idx].unsqueeze(0),pcd[B][tgt_idx].unsqueeze(0))[0]
            sum_loss=cd_loss+sum_loss
        return  sum_loss
        

    def forward(self,pcd,pre_labels,tgt_labels):
        #print(torch.unique(tgt_labels),pre_labels.shape)
        pcd =pcd.repeat(2,1,1)
        tgt_labels = tgt_labels.repeat(2,1,1)
        loss_seg = torch.nn.functional.binary_cross_entropy(pre_labels,tgt_labels)
        dice_loss = self.dice_loss(pre_labels,tgt_labels)
        print(loss_seg.item(),dice_loss.item())
        
        if loss_seg<0.1:
            # 初分割后，加入约束项（可选，其实不加也差不多）
            loss_seg = 0.5*(loss_seg+dice_loss)
        return loss_seg
    
class PoseOptimizer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.model = pointnet2(9) 

        self.mlp_tooth = nn.Sequential(
            nn.Conv1d(1, 128, 1,bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1,bias=False),
            nn.Sigmoid()
        )

        self.mlp_tooth2 = nn.Sequential(
            nn.Conv1d(128+9, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid()
        )




        
    def forward(self, pcd,num_class):
        B,N,C = pcd.shape
        B=2
        # 认为B=2
        multi_hot_labels = num_class.unsqueeze(-1).repeat(B,1,N) #(B,14,N)
        pcd =pcd.repeat(B,1,1)
        pcd = pcd[...,:9].transpose(-1, 1)
        feat_seg = self.model(pcd,multi_hot_labels)["seg_features"]
        feat_seg_max = torch.max(feat_seg,dim=1,keepdim=True)[0]
        feat_seg_weight= self.mlp_tooth(feat_seg_max)
        print(feat_seg_max.shape,pcd.shape)
        feat_seg = torch.cat([feat_seg*feat_seg_weight,pcd],dim=1)
        print(feat_seg.shape)
        
        sig_tooth = self.mlp_tooth2(feat_seg).transpose(-1, 1)
        return {"labels":sig_tooth}

def save_models(net,optimizer):
    torch.save(
        {
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        "best.pt",
    )



def main():
    """Main execution pipeline."""
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=r"C:\Users\yx\Downloads\data_part_1\sm_dir\01346914_upper.sm")
    parser.add_argument("--output_gif", default="optimization.gif")
    parser.add_argument("--num_iters", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize components
    inputs_dict= load_mesh(args.model_path)

    # Initialize optimization
    model = PoseOptimizer().to(device)
    seg_loss = SegLoss().to(device)
    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 渲染
    original_verts = inputs_dict["feat"][...,:3] 
    v_mesh = Mesh([original_verts.copy(),inputs_dict["face"] ])
    tgt_mesh = Mesh([original_verts, inputs_dict["face"]])
    pv =Plotter(N=2,axes=3,interactive=False,title="render")
    pv.at(0).show(v_mesh)
    pv.at(1).show(tgt_mesh)
    
    # Training loop
    progress_bar = tqdm(range(args.num_iters))
    for iteration in progress_bar:
        optimizer.zero_grad()
        feat = torch.from_numpy(inputs_dict["feat"][None]).to(device,torch.float32)
        tgt_labels =torch.from_numpy(inputs_dict["labels"][None]).to(device,torch.float32)
        num_class =torch.from_numpy(inputs_dict["num_class"]).to(device,torch.float32)
        out = model(feat,num_class)
        loss = seg_loss(feat[:,:,:3],pre_labels=out["labels"],tgt_labels=tgt_labels)
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

        if iteration % 5 == 0:
            with torch.no_grad():
                c_ = out["labels"].detach().cpu().numpy()[0]>0.5
                v_mesh.pointcolors = c_*np.array([255,0,0])
                pv.render()
    save_models(net=model,optimizer=optimizer)
    pv.close()

if __name__ == "__main__":
    main()