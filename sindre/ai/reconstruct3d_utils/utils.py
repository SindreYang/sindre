import numpy as np
import torch


def sdf2mesh_by_diso(sdf,diffdmc=None ,deform=None,return_quads=False, normalize=True,isovalue=0 ,invert=True):
    """用pytorch方式给，sdf 转换成 mesh"""
    device = sdf.device
    try:
        from diso import DiffDMC
    except ImportError:
        print("请安装 pip install diso")
    if diffdmc is None:
        diffdmc =DiffDMC(dtype=torch.float32).to(device)
    if invert:
        sdf*=-1
    v, f = diffdmc(sdf, deform, return_quads=return_quads, normalize=normalize, isovalue=isovalue)
    return v,f



def occ2mesh_by_pytorch3d(occ,isovalue=0 ):
    """用pytorch3d方式给，sdf 转换成 mesh"""
    from pytorch3d.ops import cubify
    from pytorch3d.structures import Meshes
    meshes = cubify(occ, isovalue)
    return meshes





def feat_to_voxel(feat_data, grid_size=None, fill_mode='feature'):
    """
    将稀疏特征还原为体素特征网格
    # 查看特征数据结构（确认关键字段）
    print("特征包含的键:", feat.keys())
    print("稀疏形状:", feat.sparse_shape)
    print("特征形状:", feat.sparse_conv_feat.features.shape)

    voxel_feat = feat_to_voxel(feat,grid_size=[289,289,289], fill_mode='feature')
    voxel_feat = F.max_pool3d(torch.from_numpy(voxel_feat).unsqueeze(0).permute(0, 4, 1, 2, 3), kernel_size=(3,3,3), stride=(3,3,3)).permute(0, 2, 3, 4, 1).squeeze(0).cpu().numpy()
    print("体素特征网格形状:", voxel_feat.shape,voxel_feat[...,0].shape)
    # verts, faces, normals, values = measure.marching_cubes(
    #     voxel_feat[...,30],
    #     level=0,
    #     spacing=(0.01, 0.01, 0.01),
    # )
    # reconstructed_mesh = vedo.Mesh([verts, faces])
    # vedo.show([reconstructed_mesh]).show().close()

    Args:
        feat_data: 包含稀疏特征的数据结构，需包含:
                  - sparse_conv_feat: spconv.SparseConvTensor
                  - sparse_shape: 稀疏网格形状
                  - grid_size: 体素尺寸（可选）
        grid_size: 自定义体素网格尺寸，默认使用sparse_shape
        fill_mode: 填充模式:
                  - 'feature': 使用原始特征（取第一个特征值）
                  - 'count': 使用体素内点数量
                  - 'mean': 使用特征平均值
    Returns:
        dense_voxel: 密集体素特征网格，形状 [D, H, W] 或 [D, H, W, C]
    """
    # 1. 提取关键数据
    sparse_feat = feat_data.sparse_conv_feat
    sparse_shape = feat_data.sparse_shape if grid_size is None else grid_size
    indices = sparse_feat.indices.cpu().numpy()  # [N, 4]：[batch_idx, z, y, x]（spconv坐标格式）
    features = sparse_feat.features.cpu().numpy()  # [N, C]：体素特征
    batch_size = sparse_feat.batch_size

    # 2. 初始化体素网格（多批次支持）
    if isinstance(sparse_shape, (list, tuple)) and len(sparse_shape) == 3:
        z_size, y_size, x_size = sparse_shape
    else:
        z_size = y_size = x_size = sparse_shape  # 若为单值则使用立方体网格
    # 根据填充模式定义网格形状
    if fill_mode == 'feature' and features.shape[1] > 1:
        dense_voxel = np.zeros((batch_size, z_size, y_size, x_size, features.shape[1]), dtype=np.float32)
    else:
        dense_voxel = np.zeros((batch_size, z_size, y_size, x_size), dtype=np.float32)

    # 3. 填充体素特征
    for i in range(indices.shape[0]):
        batch_idx, z, y, x = indices[i].astype(int)
        # 检查坐标是否在有效范围内
        if 0 <= z < z_size and 0 <= y < y_size and 0 <= x < x_size and batch_idx < batch_size:
            # 使用原始特征（支持多通道）
            if features.shape[1] == 1:
                dense_voxel[batch_idx, z, y, x] = features[i, 0]
            else:
                dense_voxel[batch_idx, z, y, x] = features[i]
    # 4. 单批次数据可去除批次维度
    if batch_size == 1:
        dense_voxel = dense_voxel[0]

    return dense_voxel