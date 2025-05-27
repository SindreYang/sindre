import torch
import torch.nn.functional as F

def grid_sample_3d(image, grid, align_corners=False):
    """
    3D版本的grid_sample，功能等同于F.grid_sample(image, grid_3d, align_corners=True)
    支持align_corners参数，控制坐标归一化方式
    
    Args:
        image (torch.Tensor): 输入图像，形状为 (N, C, D, H, W)
        grid (torch.Tensor): 采样网格，形状为 (N, D_out, H_out, W_out, 3)，最后一维是(x,y,z)坐标
        align_corners (bool): 坐标归一化方式，与F.grid_sample的align_corners参数含义一致
        
    Returns:
        torch.Tensor: 采样结果，形状为 (N, C, D_out, H_out, W_out)
        
    Notes:
        等效于PyTorch的F.grid_sample，padding_mode='zeros'
    """
    N, C, ID, IH, IW = image.shape
    _, D, H, W, _ = grid.shape

    ix = grid[..., 0]
    iy = grid[..., 1]
    iz = grid[..., 2]

    # 根据align_corners调整坐标转换方式
    if align_corners:
        # 对应align_corners=True的转换方式
        ix = ((ix + 1) / 2) * (IW - 1)
        iy = ((iy + 1) / 2) * (IH - 1)
        iz = ((iz + 1) / 2) * (ID - 1)
    else:
        # 对应align_corners=False的转换方式
        ix = ((ix + 1) * IW - 1) / 2
        iy = ((iy + 1) * IH - 1) / 2
        iz = ((iz + 1) * ID - 1) / 2

    # 计算8个最近邻点的坐标
    with torch.no_grad():
        ix_tnw = torch.floor(ix)
        iy_tnw = torch.floor(iy)
        iz_tnw = torch.floor(iz)

        ix_tne = ix_tnw + 1
        iy_tne = iy_tnw
        iz_tne = iz_tnw

        ix_tsw = ix_tnw
        iy_tsw = iy_tnw + 1
        iz_tsw = iz_tnw

        ix_tse = ix_tnw + 1
        iy_tse = iy_tnw + 1
        iz_tse = iz_tnw

        ix_bnw = ix_tnw
        iy_bnw = iy_tnw
        iz_bnw = iz_tnw + 1

        ix_bne = ix_tnw + 1
        iy_bne = iy_tnw
        iz_bne = iz_tnw + 1

        ix_bsw = ix_tnw
        iy_bsw = iy_tnw + 1
        iz_bsw = iz_tnw + 1

        ix_bse = ix_tnw + 1
        iy_bse = iy_tnw + 1
        iz_bse = iz_tnw + 1

    # 计算三线性插值权重
    tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz)
    tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz)
    tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz)
    tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz)
    bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse)
    bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw)
    bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne)
    bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw)

    # 创建掩码以标识超出边界的坐标点 (padding_mode='zero')
    with torch.no_grad():
        # 为每个邻居点创建是否在有效范围内的掩码
        mask_tnw = (ix_tnw >= 0) & (ix_tnw < IW) & (iy_tnw >= 0) & (iy_tnw < IH) & (iz_tnw >= 0) & (iz_tnw < ID)
        mask_tne = (ix_tne >= 0) & (ix_tne < IW) & (iy_tne >= 0) & (iy_tne < IH) & (iz_tne >= 0) & (iz_tne < ID)
        mask_tsw = (ix_tsw >= 0) & (ix_tsw < IW) & (iy_tsw >= 0) & (iy_tsw < IH) & (iz_tsw >= 0) & (iz_tsw < ID)
        mask_tse = (ix_tse >= 0) & (ix_tse < IW) & (iy_tse >= 0) & (iy_tse < IH) & (iz_tse >= 0) & (iz_tse < ID)
        mask_bnw = (ix_bnw >= 0) & (ix_bnw < IW) & (iy_bnw >= 0) & (iy_bnw < IH) & (iz_bnw >= 0) & (iz_bnw < ID)
        mask_bne = (ix_bne >= 0) & (ix_bne < IW) & (iy_bne >= 0) & (iy_bne < IH) & (iz_bne >= 0) & (iz_bne < ID)
        mask_bsw = (ix_bsw >= 0) & (ix_bsw < IW) & (iy_bsw >= 0) & (iy_bsw < IH) & (iz_bsw >= 0) & (iz_bsw < ID)
        mask_bse = (ix_bse >= 0) & (ix_bse < IW) & (iy_bse >= 0) & (iy_bse < IH) & (iz_bse >= 0) & (iz_bse < ID)

        # 将超出边界的坐标clamp到有效范围，以便后续gather操作不会报错
        # 但会在取值后应用掩码，将超出边界的值设为0
        ix_tnw = torch.clamp(ix_tnw, 0, IW - 1)
        iy_tnw = torch.clamp(iy_tnw, 0, IH - 1)
        iz_tnw = torch.clamp(iz_tnw, 0, ID - 1)

        ix_tne = torch.clamp(ix_tne, 0, IW - 1)
        iy_tne = torch.clamp(iy_tne, 0, IH - 1)
        iz_tne = torch.clamp(iz_tne, 0, ID - 1)

        ix_tsw = torch.clamp(ix_tsw, 0, IW - 1)
        iy_tsw = torch.clamp(iy_tsw, 0, IH - 1)
        iz_tsw = torch.clamp(iz_tsw, 0, ID - 1)

        ix_tse = torch.clamp(ix_tse, 0, IW - 1)
        iy_tse = torch.clamp(iy_tse, 0, IH - 1)
        iz_tse = torch.clamp(iz_tse, 0, ID - 1)

        ix_bnw = torch.clamp(ix_bnw, 0, IW - 1)
        iy_bnw = torch.clamp(iy_bnw, 0, IH - 1)
        iz_bnw = torch.clamp(iz_bnw, 0, ID - 1)

        ix_bne = torch.clamp(ix_bne, 0, IW - 1)
        iy_bne = torch.clamp(iy_bne, 0, IH - 1)
        iz_bne = torch.clamp(iz_bne, 0, ID - 1)

        ix_bsw = torch.clamp(ix_bsw, 0, IW - 1)
        iy_bsw = torch.clamp(iy_bsw, 0, IH - 1)
        iz_bsw = torch.clamp(iz_bsw, 0, ID - 1)

        ix_bse = torch.clamp(ix_bse, 0, IW - 1)
        iy_bse = torch.clamp(iy_bse, 0, IH - 1)
        iz_bse = torch.clamp(iz_bse, 0, ID - 1)

    # 重塑图像为一维数组以便使用torch.gather
    image = image.view(N, C, ID * IH * IW)

    # 计算每个最近邻点的一维索引并获取对应的值
    tnw_val = torch.gather(image, 2, (iz_tnw * IW * IH + iy_tnw * IW + ix_tnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tne_val = torch.gather(image, 2, (iz_tne * IW * IH + iy_tne * IW + ix_tne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tsw_val = torch.gather(image, 2, (iz_tsw * IW * IH + iy_tsw * IW + ix_tsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    tse_val = torch.gather(image, 2, (iz_tse * IW * IH + iy_tse * IW + ix_tse).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bnw_val = torch.gather(image, 2, (iz_bnw * IW * IH + iy_bnw * IW + ix_bnw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bne_val = torch.gather(image, 2, (iz_bne * IW * IH + iy_bne * IW + ix_bne).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bsw_val = torch.gather(image, 2, (iz_bsw * IW * IH + iy_bsw * IW + ix_bsw).long().view(N, 1, D * H * W).repeat(1, C, 1))
    bse_val = torch.gather(image, 2, (iz_bse * IW * IH + iy_bse * IW + ix_bse).long().view(N, 1, D * H * W).repeat(1, C, 1))

    # 应用掩码，将超出边界的点的值设为0
    tnw_val = tnw_val * mask_tnw.view(N, 1, D * H * W).repeat(1, C, 1)
    tne_val = tne_val * mask_tne.view(N, 1, D * H * W).repeat(1, C, 1)
    tsw_val = tsw_val * mask_tsw.view(N, 1, D * H * W).repeat(1, C, 1)
    tse_val = tse_val * mask_tse.view(N, 1, D * H * W).repeat(1, C, 1)
    bnw_val = bnw_val * mask_bnw.view(N, 1, D * H * W).repeat(1, C, 1)
    bne_val = bne_val * mask_bne.view(N, 1, D * H * W).repeat(1, C, 1)
    bsw_val = bsw_val * mask_bsw.view(N, 1, D * H * W).repeat(1, C, 1)
    bse_val = bse_val * mask_bse.view(N, 1, D * H * W).repeat(1, C, 1)

    # 应用权重并组合结果
    out_val = (tnw_val.view(N, C, D, H, W) * tnw.view(N, 1, D, H, W) +
               tne_val.view(N, C, D, H, W) * tne.view(N, 1, D, H, W) +
               tsw_val.view(N, C, D, H, W) * tsw.view(N, 1, D, H, W) +
               tse_val.view(N, C, D, H, W) * tse.view(N, 1, D, H, W) +
               bnw_val.view(N, C, D, H, W) * bnw.view(N, 1, D, H, W) +
               bne_val.view(N, C, D, H, W) * bne.view(N, 1, D, H, W) +
               bsw_val.view(N, C, D, H, W) * bsw.view(N, 1, D, H, W) +
               bse_val.view(N, C, D, H, W) * bse.view(N, 1, D, H, W))

    return out_val    


def grid_sample_2d(im, grid, align_corners=False):
 
    """
    2D版本的grid_sample，使用双线性插值对输入像素进行采样
    
    Args:
        im (torch.Tensor): 输入特征图，形状 (N, C, H, W)
        grid (torch.Tensor): 点坐标，形状 (N, Hg, Wg, 2)，最后一维是(x,y)坐标
        align_corners (bool): 坐标归一化方式，与F.grid_sample的align_corners参数含义一致
        
    Returns:
        torch.Tensor: 采样结果，形状为 (N, C, Hg, Wg)
        
    Notes:
        等效于PyTorch的F.grid_sample，padding_mode='zeros'
    """
 
    n, c, h, w = im.shape
 
    gn, gh, gw, _ = grid.shape
 
    assert n == gn
 
 
    x = grid[:, :, :, 0]
 
    y = grid[:, :, :, 1]
 
 
    if align_corners:
 
        x = ((x + 1) / 2) * (w - 1)
 
        y = ((y + 1) / 2) * (h - 1)
 
    else:
 
        x = ((x + 1) * w - 1) / 2
 
        y = ((y + 1) * h - 1) / 2
 
 
    x = x.contiguous().view(n, -1)
 
    y = y.contiguous().view(n, -1)
 
 
    x0 = torch.floor(x).long()
 
    y0 = torch.floor(y).long()
 
    x1 = x0 + 1
 
    y1 = y0 + 1
 
 
    wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
 
    wb = ((x1 - x) * (y - y0)).unsqueeze(1)
 
    wc = ((x - x0) * (y1 - y)).unsqueeze(1)
 
    wd = ((x - x0) * (y - y0)).unsqueeze(1)
 
 
    # Apply default for grid_sample function zero padding
 
    im_padded = F.pad(im, pad=[1, 1, 1, 1], mode='constant', value=0)
 
    padded_h = h + 2
 
    padded_w = w + 2
 
    # save points positions after padding
 
    x0, x1, y0, y1 = x0 + 1, x1 + 1, y0 + 1, y1 + 1
 
 
    # Clip coordinates to padded image size
 
    x0 = torch.where(x0 < 0, torch.tensor(0), x0)
 
    x0 = torch.where(x0 > padded_w - 1, torch.tensor(padded_w - 1), x0)
 
    x1 = torch.where(x1 < 0, torch.tensor(0), x1)
 
    x1 = torch.where(x1 > padded_w - 1, torch.tensor(padded_w - 1), x1)
 
    y0 = torch.where(y0 < 0, torch.tensor(0), y0)
 
    y0 = torch.where(y0 > padded_h - 1, torch.tensor(padded_h - 1), y0)
 
    y1 = torch.where(y1 < 0, torch.tensor(0), y1)
 
    y1 = torch.where(y1 > padded_h - 1, torch.tensor(padded_h - 1), y1)
 

 
 
    im_padded = im_padded.view(n, c, -1)
 
 
    x0_y0 = (x0 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
 
    x0_y1 = (x0 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
 
    x1_y0 = (x1 + y0 * padded_w).unsqueeze(1).expand(-1, c, -1)
 
    x1_y1 = (x1 + y1 * padded_w).unsqueeze(1).expand(-1, c, -1)
 
 
    Ia = torch.gather(im_padded, 2, x0_y0)
 
    Ib = torch.gather(im_padded, 2, x0_y1)
 
    Ic = torch.gather(im_padded, 2, x1_y0)
 
    Id = torch.gather(im_padded, 2, x1_y1)
 
 
    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).reshape(n, c, gh, gw)


def grid_sample(input, grid, align_corners=True):
    """
    自适应版本的grid_sample，根据输入维度自动选择2D或3D实现
    
    Args:
        input (torch.Tensor): 输入图像
            - 对于2D采样: 形状为 (N, C, H, W)
            - 对于3D采样: 形状为 (N, C, D, H, W)
        grid (torch.Tensor): 采样网格
            - 对于2D采样: 形状为 (N, Hg, Wg, 2)
            - 对于3D采样: 形状为 (N, D_out, H_out, W_out, 3)
        align_corners (bool): 坐标归一化方式，与F.grid_sample的align_corners参数含义一致
        
    Returns:
        torch.Tensor: 采样结果
            - 对于2D采样: 形状为 (N, C, Hg, Wg)
            - 对于3D采样: 形状为 (N, C, D_out, H_out, W_out)
            
    Raises:
        ValueError: 如果输入维度不是4D(2D)或5D(3D)
        ValueError: 如果grid的最后一维不是2(2D)或3(3D)
        
    Notes:
        该函数根据输入维度自动调用grid_sample_2d或grid_sample_3d
        等效于PyTorch的F.grid_sample，padding_mode='zeros'
    """
    # 检查输入维度
    if input.dim() == 4:
        # 2D情况
        if grid.dim() != 4 or grid.shape[-1] != 2:
            raise ValueError(f"对于2D grid_sample，grid应是4D张量且最后一维大小为2，但得到grid形状为{grid.shape}")
        return grid_sample_2d(input, grid, align_corners=align_corners)
    elif input.dim() == 5:
        # 3D情况
        if grid.dim() != 5 or grid.shape[-1] != 3:
            raise ValueError(f"对于3D grid_sample，grid应是5D张量且最后一维大小为3，但得到grid形状为{grid.shape}")
        return grid_sample_3d(input, grid, align_corners=align_corners)
    else:
        raise ValueError(f"input应为4D(2D)或5D(3D)张量，但得到{input.dim()}D张量")   








if __name__ == '__main__':
    # 用户函数计时
    start = torch.cuda.Event(enable_timing=True)
    end =  torch.cuda.Event(enable_timing=True)
    start.record()
    #########################2D 测试##################################################    
    # 生成随机输入
    torch.manual_seed(42)
    n, c, h, w = 2, 3, 50, 50
    grid = torch.randn(n, h, w, 2)  # 归一化坐标
    im = torch.randn(n, c, h, w)

    # 测试align_corners=True
    out_user = grid_sample_2d(im, grid, align_corners=True)
    out_pytorch = F.grid_sample(im, grid, align_corners=True)
    error_true = torch.mean(torch.abs(out_user - out_pytorch)).item()

    # 测试align_corners=False
    out_user_false = grid_sample_2d(im, grid, align_corners=False)
    out_pytorch_false = F.grid_sample(im, grid, align_corners=False)
    error_false = torch.mean(torch.abs(out_user_false - out_pytorch_false)).item()

    print(f"2D align_corners=True 误差: {error_true:.6f}")
    print(f"2D align_corners=False 误差: {error_false:.6f}")

    # 测速
    start.record()
    for _ in range(100):
        grid_sample(im, grid, align_corners=True)
    end.record()
    torch.cuda.synchronize()
    time_user = start.elapsed_time(end) / 100


    start.record()
    for _ in range(100):
        F.grid_sample(im, grid, align_corners=True)
    end.record()
    torch.cuda.synchronize()
    time_pytorch = start.elapsed_time(end) / 100

    print(f"2D 自我实现时间: {time_user:.2f} ms, PyTorch时间: {time_pytorch:.2f} ms\n")
    ###################################################################################


    #########################3D 测试##################################################
    n, c,d, h, w = 2, 3,50, 50, 50
    grid_3d = torch.randn(n,d, h, w, 3)  # 归一化坐标
    voxel = torch.randn(n, c, d,h, w)

    # 测试align_corners=True
    out_user = grid_sample_3d(voxel, grid_3d,align_corners=True)
    out_pytorch = F.grid_sample(voxel, grid_3d,align_corners=True)
    error_true = torch.mean(torch.abs(out_user - out_pytorch)).item()

    # 测试align_corners=False
    out_user_false = grid_sample_3d(voxel, grid_3d, align_corners=False)
    out_pytorch_false = F.grid_sample(voxel, grid_3d, align_corners=False)
    error_false = torch.mean(torch.abs(out_user_false - out_pytorch_false)).item()
    print(f"3D align_corners=True 误差: {error_true:.6f}")
    print(f"3D align_corners=False 误差: {error_false:.6f}")

    # 测速
    start.record()
    for _ in range(100):
        grid_sample_3d(voxel, grid_3d, align_corners=True)
    end.record()
    torch.cuda.synchronize()
    time_user = start.elapsed_time(end) / 100


    start.record()
    for _ in range(100):
        F.grid_sample(voxel, grid_3d, align_corners=True)
    end.record()
    torch.cuda.synchronize()
    time_pytorch = start.elapsed_time(end) / 100

    print(f"3D 自我实现时间: {time_user:.2f} ms, PyTorch时间: {time_pytorch:.2f} ms")
    
        
    #################################################################################################    



