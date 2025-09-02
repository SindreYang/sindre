
import argparse
import os
import random

import torch.nn as nn
import vedo
from pytorch3d.renderer import (
    FoVOrthographicCameras, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    HardPhongShader, AmbientLights
)
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sindre.lmdb import Reader
from sindre.utils3d import *
from sindre.utils3d.pointcloud_augment import *

# 屏蔽TensorFlow的INFO/WARNING级日志（包括oneDNN提示
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# 屏蔽Protobuf版本不匹配的警告（精确匹配日志文本）
import warnings
warnings.filterwarnings("ignore", message="Protobuf gencode version.*")
import monai

class multi_view_model(nn.Module):
    """
    基于多视角的3D网格分割模型
    使用多视角渲染将3D网格投影到2D平面，然后用2D UNet进行分割
    loss = monai.losses.DiceCELoss(to_onehot_y=True,softmax=True)
    # 基于球归一化
    selected_vertices = out_mesh.vertices[out_labels == 1]
    center = np.mean(selected_vertices,axis=0)
    r=13
    out_mesh.cut_with_sphere(center,r )
    out_mesh.vertices = (out_mesh.vertices-center)/r


    """

    def __init__(self,num_classes=1,image_size=480,num_views=64,radius=0.8, device='cuda'):
        super().__init__()
        self.num_views= num_views
        self.num_classes = num_classes
        self.radius = radius
        self.device = device
        # 相机
        #self.cameras = FoVPerspectiveCameras(device=self.device)
        self.cameras = FoVOrthographicCameras(device=self.device)
        # 光栅化设置
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0,
            faces_per_pixel=1,
        )
        # 光源
        self.lights = AmbientLights(device=self.device)
        # 光栅化器
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings
        )
        # 渲染器
        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        )

        # 网络
        self.UNet = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=5,   # images: torch.cuda.FloatTensor[batch_size,224,224,4]
            out_channels=num_classes,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(self.device)

        # 相机视角：
        # 生成均匀分布的球面坐标（Fibonacci采样法）
        indices = torch.arange(0, self.num_views, dtype=torch.float32, device=self.device)
        phi = torch.acos(1 - 2 * indices / self.num_views)  # 俯仰角 [0, pi]
        theta = torch.pi * (1 + 5**0.5) * indices       # 方位角 [0, 2pi]
        # 转换为笛卡尔坐标 (x, y, z)
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        camera_positions = torch.stack((x, y, z), dim=1)*radius
        # 生成视角
        self.R = look_at_rotation(camera_positions,
                                  device=self.device
                                  )
        # 计算平移向量 T = -R @ camera_position
        self.T = -torch.bmm(self.R.transpose(1, 2), camera_positions[:, :, None])[:, :, 0] # (num_views, 3)
        # self.register_buffer("R", R)
        # self.register_buffer("T", T)

    def get_2d(self,meshes):
        # 渲染图像
        images = self.renderer(meshes_world=meshes, R=self.R, T=self.T)
        # 获取光栅化信息
        map_info = self.rasterizer(meshes_world=meshes, R=self.R, T=self.T)
        # 获取深度
        depth = map_info.zbuf
        min_depth = depth.min()
        max_depth = depth.max()
        depth_normalized = (depth - min_depth) / (max_depth - min_depth)
        # 像素-面映射
        pix2face=map_info.pix_to_face
        # 合并特征
        combined_features = torch.cat([images, depth_normalized], dim=-1)
        return  combined_features,pix2face



    @torch.no_grad()
    def gen_mesh_labels(self, vertices, faces,center,r=13):
        DEBUG =True
        cut_mesh = vedo.Mesh([vertices, faces])
        cut_mesh.cut_with_sphere(center,r )
        cut_mesh.compute_normals()
        cut_mesh.decimate(n=10000)
        cut_mesh.write(r"debug/gen_cut_mesh.ply")
        vertices = (cut_mesh.vertices-center)/r
        vertex_normals = np.array(cut_mesh.vertex_normals)
        faces=np.array(cut_mesh.cells)
        # 构建
        vertices=torch.from_numpy(vertices).to(device=self.device, dtype=torch.float32)
        faces=torch.from_numpy(faces).to(device=self.device,dtype=torch.float32)
        vertex_normals=torch.from_numpy(vertex_normals*0.5 + 0.5).to(device=self.device,dtype=torch.float32)
        textures = TexturesVertex(verts_features=vertex_normals.unsqueeze(0))
        mesh = Meshes(
            verts=vertices.unsqueeze(0),
            faces=faces.unsqueeze(0),
            textures=textures,
        )
        meshes =mesh.extend(self.num_views).to(self.device)
        images, pix2face = self.get_2d(meshes)
        if DEBUG:
            for j in range(0,64,10):
                vedo.Image(images[j].cpu().numpy()*255).write(f"debug/debug_inputs_{j}.png")
        images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        pred_logits = self.UNet(images)
        if DEBUG:
            for j in range(0,64,10):
                vedo.Image(pred_logits[j].permute(1,2,0).cpu().numpy()*255).write(f"debug/debug_pred_{j}.png")
            print("开始投票...")
        # 获取预测概率并确定类别
        if self.num_classes == 1:
            # 二分类情况
            pred_probs = torch.sigmoid(pred_logits)
            pred_labels = (pred_probs > 0.5).long().squeeze(1)  # (N, H, W)
        else:
            # 多分类情况
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_labels = torch.argmax(pred_probs, dim=1)  # (N, H, W)

        num_views, H, W = pix2face.shape[:3]
        num_faces = mesh.num_faces_per_mesh().item()

        # 初始化面标签投票器
        vote_num_classes = 2 if self.num_classes == 1 else self.num_classes
        face_votes = torch.zeros((num_faces, vote_num_classes),
                                 device=self.device, dtype=torch.long)
        for i in range(num_views):
            view_pix2face = pix2face[i].view(-1)  # 展平 (H*W,)
            view_pred = pred_labels[i].view(-1)  # 展平 (H*W,)
            valid_mask = view_pix2face >= 0
            valid_faces = view_pix2face[valid_mask] % num_faces
            valid_preds = view_pred[valid_mask]


            if len(valid_faces) > 0:
                face_pred_pairs = valid_faces * vote_num_classes + valid_preds
                minlength = num_faces * vote_num_classes
                if face_pred_pairs.max() >= minlength:
                    minlength = face_pred_pairs.max().item() + 1
                votes = torch.bincount(face_pred_pairs, minlength=minlength)
                votes = votes[:num_faces * vote_num_classes]
                votes = votes.view(num_faces, vote_num_classes)
                face_votes += votes

        mesh_labels = torch.argmax(face_votes, dim=1).cpu().numpy()
        # from sindre.utils3d import UnifiedLabelRefiner
        # print(mesh_labels.shape, face_votes.cpu().numpy().shape)
        # refine_mesh_labels =UnifiedLabelRefiner(cut_mesh.vertices,
        #                                         np.array(cut_mesh.cells),
        #                                         face_votes.cpu().numpy(),
        #                                         2,
        #                                         100).refine()

        new_mesh=vedo.Mesh([cut_mesh.vertices,np.array(cut_mesh.cells)[mesh_labels>0]])
        if DEBUG:
            new_mesh.write("debug/debug_ouput.ply")


        return new_mesh.clean().fill_holes().boundaries().extract_largest_region().join(reset=True).vertices



    def forward(self, meshes,face_labels):
        B = len(meshes)
        pre_labels =[]
        target_labels = []
        for i  in range(B):
            mesh = meshes[i].extend(self.num_views).to(self.device)
            images,pix2face = self.get_2d(mesh)
            # debug
            #vedo.show([vedo.Image(im.cpu().numpy()*255) for im in images],N=64).close()
            # for j in range(64):
            #     vedo.Image(images[j].cpu().numpy()*255).write(f"debug/{j}.png")


            images = images.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            # 映射面标签到像素,过滤背景像素(面索引为-1的像素)
            # 由于网格被扩展，面索引会超出原始范围，需要取模回到原始范围
            valid_mask = pix2face >= 0
            # 将扩展后的面索引映射回原始网格的面索引
            face_indices = pix2face.clone()
            face_indices[valid_mask] = face_indices[valid_mask] % len(face_labels[i])
            # 映射面标签到像素，过滤背景像素(面索引为-1的像素)
            target_img_labels = torch.take(face_labels[i].to(self.device), face_indices) * valid_mask

            # debug
            # for j in range(64):
            #     vedo.Image(target_img_labels[j].cpu().numpy()*255).write(f"debug/label_{j}.png")
            # 将图片进行分割
            pre_images_labels = self.UNet(images)
            pre_labels.append(pre_images_labels)
            target_labels.append(target_img_labels.permute(0, 3,1, 2))

        return {
            "pre_labels":torch.hstack(pre_labels),
            "target_labels":torch.hstack(target_labels),
        }




class MyDataset(Dataset):
    def __init__(
            self,
            data_dir ,
            batch_size: int = 2,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 1024,
            prefetch_factor=None,
            sample_size: int = 6666,
            test = False,
            val = False,
    ):
        super().__init__()
        self.db = Reader(data_dir,True)
        self.datasets = [ i for i in range(len(self.db))]

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.sample_size = sample_size
        self.prefetch_factor=prefetch_factor


        self.DEBUG=False
        self.test = test
        split_idx = batch_size*10
        random.shuffle(self.datasets) # 打乱list
        if val:
            self.datasets =self.datasets[:split_idx]
        else:
            self.datasets =self.datasets[split_idx:]

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx_):
        data =self.db[self.datasets[idx_]]
        tf = RotateXYZ_np()
        pcd_normal = tf(np.hstack([data['vertices'],data["vertex_normals"]]))
        vertices,normal = pcd_normal[:, 0:3],pcd_normal[:, 3:6]

        vertices=torch.from_numpy(vertices).to(dtype=torch.float32)
        colors=torch.from_numpy(normal*0.5 + 0.5).to(dtype=torch.float32)

        faces=torch.from_numpy(data['faces'].copy()).to(dtype=torch.int32)
        faces_labels =torch.from_numpy(data["faces_labels"].copy()).to(dtype=torch.int32)
        return vertices, faces,colors,faces_labels

    def batch_fn(self,batch):
        # verts=[]
        # faces=[]
        # textures=[]
        # meshes=[]
        # labels = []
        # max_len = 500000
        # for  item  in batch:
        #     mesh_pt3d, face_labels = item
        #     meshes.append(mesh_pt3d)
        #     len_labels = len(face_labels)
        #     if len_labels > max_len:
        #         RuntimeError(f"{len_labels} > {max_len}")
        #     verts.append(mesh_pt3d.verts_packed())
        #     faces.append(mesh_pt3d.faces_packed())
        #     textures.append(mesh_pt3d.textures)
        #     labels.append(torch.cat([face_labels, torch.full((max_len-len_labels,), -1, dtype=face_labels.dtype)]))
        # batched_meshes=join_meshes_as_batch(meshes)
        # batched_labels = torch.stack(labels, dim=0)
        # return batched_meshes, batched_labels
        #return  [item[0] for  item  in batch], [item[1] for  item  in batch]
        vertices_list = [item[0] for item in batch]
        faces_list = [item[1] for item in batch]
        colors_list = [item[2] for item in batch]
        labels_list = [item[3] for item in batch]

        # 创建纹理对象
        textures = TexturesVertex(verts_features=colors_list)
        # 创建网格对象
        meshes = Meshes(
            verts=vertices_list,
            faces=faces_list,
            textures=textures

        )
        return meshes, labels_list








    def _init_fn(self, worker_id):
        # 固定随机数
        np.random.seed(self.seed + worker_id)


    def train_dataloader(self,multi_gpu=False):
        if multi_gpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self)
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,# 必须禁用，避免多卡冲突
                worker_init_fn=self._init_fn,
                collate_fn=self.batch_fn,
                prefetch_factor=self.prefetch_factor,
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
                worker_init_fn=self._init_fn,
                collate_fn=self.batch_fn,

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
                worker_init_fn=self._init_fn,
                prefetch_factor=self.prefetch_factor,
                drop_last=True,
                collate_fn=self.batch_fn,
                sampler=sampler,
            )
        else:
            return DataLoader(
                dataset=self,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,# 必须禁用，避免多卡冲突
                worker_init_fn=self._init_fn,
                prefetch_factor=self.prefetch_factor,
                collate_fn=self.batch_fn,
                drop_last=True,
            )




def save_models(args,net,optimizer,loss,curr_iter,multi_gpu):
    if multi_gpu:
        net_dict =net.module.state_dict()
    else:
        net_dict =net.state_dict()
    torch.save(
        {
            "state_dict": net_dict,
            "optimizer": optimizer.state_dict(),
            "curr_iter": curr_iter,
            "loss":loss,
        },
        args.model_name,
    )




def main():
    """Main execution pipeline. """
    os.environ["DISPLAY"]=":0"
    os.chdir(os.path.dirname(__file__))
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_path", default="origin_1067_with_faces.db")
    parser.add_argument("--num_iters", type=int, default=5000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--model_name",  default=r"best_mv_depth.pt")
    args = parser.parse_args()
    # 设置随机数
    set_seeds(1024)

    # Device setup
    if args.multi_gpu:
        import torch.distributed as dist
        # 每个进程的线程数
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device="cuda" if torch.cuda.is_available() else "cpu")


    # Initialize components
    datasets=MyDataset(data_dir=args.datasets_path,batch_size=args.batch_size,num_workers=args.num_workers)
    print("datasets:",len(datasets))
    train_dataloader=datasets.train_dataloader(multi_gpu=args.multi_gpu)
    datasets_val=MyDataset(data_dir=args.datasets_path,batch_size=args.batch_size,num_workers=args.num_workers,val=True)
    print("datasets_val:",len(datasets_val))
    val_dataloader=datasets_val.val_dataloader(multi_gpu=args.multi_gpu)



    # Initialize optimization
    model = multi_view_model()
    if args.multi_gpu:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
    my_loss =  monai.losses.DiceLoss(sigmoid=True).to(device)
    optimizer =torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.005)
    best_loss = float('inf')
    curr_epoch = 0


    if os.path.exists(args.model_name):
        checkpoint = torch.load(args.model_name, map_location=device)
        model.load_state_dict(checkpoint["state_dict"],strict=False)
        try:
            #optimizer.load_state_dict(checkpoint["optimizer"])
            curr_epoch =checkpoint["curr_iter"]
            #best_loss =checkpoint["loss"]
            print(f"epoch {curr_epoch} ,loss {best_loss}")
        except Exception as e:
            print(f"加载错误：{e}")


    # Training loop
    for epoch in range(args.num_iters)[curr_epoch:]:
        model.train()
        train_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch_idx, data in enumerate(progress_bar):
            meshes,labels = data
            optimizer.zero_grad()
            outputs = model(meshes,labels)
            loss = my_loss(outputs["pre_labels"],outputs["target_labels"])
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            train_loss += loss.item()




        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for  data in val_dataloader:
                meshes,labels = data
                outputs = model(meshes,labels)
                loss_val = my_loss(outputs["pre_labels"],outputs["target_labels"])
                val_loss += loss_val.item()

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)


        # 打印训练信息
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}| LR: {optimizer.param_groups[0]['lr']}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f"保存模型:{avg_val_loss}")
            save_models(args=args,net=model,optimizer=optimizer,loss=avg_val_loss,curr_iter=epoch,multi_gpu=args.multi_gpu)



if __name__ == "__main__":
    main()
