import math
from vedo import *
from sindre.utils3d import *

import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sindre.utils3d.networks.pointnet2 import pointnet2_ssg
from pytorch3d.loss import chamfer_distance

def load_mesh(sm_path: str) :
    mesh= SindreMesh()
    mesh.load(sm_path)
    # print(np.unique( mesh.vertex_labels))
    #  mesh.show()
    
    # 定义变换
    Normalize_std = Normalize_np()
    Transform=RotateXYZ_np()

    # 标准化
    vertices =Normalize_std(mesh.vertices)
    
    # 变换后
    feat = np.hstack([vertices,mesh.vertex_normals])
    feat = Transform(feat)
    feat = np.hstack([feat,mesh.vertex_colors[...,:3]/255])
    R = Transform.get_R()
    print(feat.max(),feat.min(),feat.shape,R)
    
    
 
    return {
        "feat":feat ,
        "R":R ,
        "face":mesh.faces,
        "labels":mesh.vertex_labels,
        }




class PoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def feature_transform_regularizer(self,trans):
        d = trans.size()[1]
        I = torch.eye(d)[None, :, :].to(trans.device)
        product = torch.bmm(trans, trans.transpose(2, 1))
        # 检查是否接近正交矩阵
        loss = torch.mean(torch.norm(product - I, dim=(1, 2)))
        return loss

    def forward(self,pre_R,tgt_R,pre_labels,tgt_labels):
        # R正交矩阵，正交矩阵的逆等于转置
        # verts_transformed = torch.matmul(v, R)
        # transformed_mesh = Meshes(
        #     verts=verts_transformed,
        #     faces=f,
        # )
        # transformed_mesh._compute_vertex_normals(True)
        # x,xn=sample_points_from_meshes(transformed_mesh, 2000,return_normals=True)
        # y,yn=sample_points_from_meshes(self.meshes_tgt, 2000,return_normals=True)
        # # x,xn=verts_transformed,transformed_mesh.verts_normals_padded()
        # # y,yn=v.unsqueeze(0) ,self.meshes_tgt.verts_normals_padded()
        # loss,normals_loss = chamfer_distance(x,y=y,x_normals=xn,y_normals=yn)
        loss_I =self.feature_transform_regularizer(pre_R)
        loss_R= torch.nn.functional.mse_loss(pre_R,tgt_R)
        loss_seg = torch.nn.functional.binary_cross_entropy(pre_labels,tgt_labels.repeat(2,1,1))


        print(loss_R.item(),loss_I.item(),loss_seg.item())
        return loss_seg+loss_R+loss_I*0.1
    
class PoseOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = pointnet2_ssg(9)

        self.mlp_tooth = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 9, 1),
        )
   
    #@torch.no_grad()
    def pca_with_svd(self,data, eps=1e-6):
        """PCA实现 """
        identity = torch.eye(data.size(-1), device=data.device) * eps 
        cov = torch.matmul(data.transpose(-2, -1), data) / (data.size(-2) - 1)
        cov_reg = cov + identity
        _, _, v = torch.linalg.svd(cov_reg, full_matrices=False)
        rotation = v.transpose(1,2)  
        # det = torch.det(rotation)    # 确保右手坐标系
        # new_last_column = rotation[:, :, -1] * det.unsqueeze(-1)
        # rotation = torch.cat([rotation[:, :, :-1], new_last_column.unsqueeze(-1)], dim=-1)
        return rotation

        
    def forward(self, pcd):
        
        pcd =pcd.repeat(2,1,1)
        pcd = pcd[...,:9].transpose(-1, 1)
        feat_seg = self.model(pcd)["seg_features"]
        sig_tooth = self.mlp_tooth(feat_seg).transpose(-1, 1)
        feat_mat = torch.max(feat_seg,dim=-1)[0]

        out = self.mlp(feat_mat.unsqueeze(-1))# (B, 3，N)
        R = out.reshape(-1,3,3)+  torch.eye(3, device=out.device)# self.pca_with_svd(out.transpose(-1, 1))
        return {"R": R,"labels":sig_tooth}

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
    pose_loss = PoseLoss().to(device)
    optimizer =torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 渲染
    original_verts = inputs_dict["feat"][...,:3] 
    v_mesh = Mesh([original_verts.copy(),inputs_dict["face"] ])
    tgt_mesh = Mesh([np.matmul(original_verts,inputs_dict["R"]), inputs_dict["face"]])
    pv =Plotter(N=2,axes=3,interactive=False,title="render")
    pv.at(0).show(v_mesh)
    pv.at(1).show(tgt_mesh)
    
    # Training loop
    progress_bar = tqdm(range(args.num_iters))
    for iteration in progress_bar:
        optimizer.zero_grad()
        feat = torch.from_numpy(inputs_dict["feat"][None]).to(device,torch.float32)
        tgt_R =torch.from_numpy(inputs_dict["R"][None]).to(device,torch.float32)
        tgt_labels =torch.from_numpy(inputs_dict["labels"][None]>0).to(device,torch.float32)
        
        out = model(feat)
        loss = pose_loss(pre_R=out["R"],tgt_R=tgt_R,pre_labels=out["labels"],tgt_labels=tgt_labels)
        loss.backward()
        optimizer.step()
        progress_bar.set_description(f"Loss: {loss.item():.4f}")

        if iteration % 5 == 0:
            with torch.no_grad():
                R_ = out["R"].detach().cpu().numpy()[0]  
                c_ = out["labels"].detach().cpu().numpy()[0]>0.5
                transformed_verts = np.matmul(original_verts, R_)
                v_mesh.vertices = transformed_verts
                v_mesh.pointcolors = c_*255
                pv.render()
    save_models(net=model,optimizer=optimizer)
    pv.close()

if __name__ == "__main__":
    main()