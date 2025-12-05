from typing import Union, Tuple, List, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat,rearrange
from skimage import measure
from tqdm import tqdm
import numpy as np
from sindre import CustomLogger
from sindre.ai.layers import CrossAttentionDecoder,FourierEmbedder
log=CustomLogger("sindre_generate").get_logger()

class SurfaceExtractorV2:
    """ mc效果最好"""
    def __init__(self,device:torch.device,algo="dmc"):
        super().__init__()
        self.bounds=1.01
        self.mc_level=0.0
        self.num_chunks=20000
        self.octree_resolution=256
        self.enable_pbar=True
        self.device = device
        self.algo = algo
        if self.algo == "dmc":
            from diso import DiffDMC
            self.dmc = DiffDMC(dtype=torch.float32).to(device)

    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size


    def mc_extractor(self,grid_logit):
        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logit.cpu().numpy(),
            self.mc_level,
            method="lewiner"
        )
        grid_size, bbox_min, bbox_size = self._compute_box_stat(self.bounds, self.octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces
    def dmc_extractor(self,grid_logit):
        sdf = -grid_logit / self.octree_resolution
        sdf = sdf.to(torch.float32).contiguous()
        verts, faces = self.dmc(-sdf, deform=None, return_quads=False, normalize=True)
        verts = self.center_vertices(verts)
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]
        return vertices, faces

    def center_vertices(self,vertices):
        """Translate the vertices so that bounding box is centered at zero."""
        vert_min = vertices.min(dim=0)[0]
        vert_max = vertices.max(dim=0)[0]
        vert_center = 0.5 * (vert_min + vert_max)
        return vertices - vert_center

    def __call__(self, grid_logits):
        outputs = []
        B =grid_logits.shape[0]
        for i in range(B):
            try:
                if self.algo != "dmc":
                    vertices, faces = self.mc_extractor(grid_logits[i])
                    vertices = vertices.astype(np.float32)
                    faces = np.ascontiguousarray(faces)
                    outputs.append([vertices, faces])
                else:
                    vertices, faces = self.dmc_extractor(grid_logits[i])
                    outputs.append([vertices, faces])

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs



class FlashVDMCrossAttentionProcessor:
    def __init__(self, topk=None):
        self.topk = topk
        try:
            from sageattention import sageattn
            self.scaled_dot_product_attention = sageattn
        except ImportError:
            log.warning('Please install the package "sageattention" to use this USE_SAGEATTN.')
            self.scaled_dot_product_attention = F.scaled_dot_product_attention


    def __call__(self, attn, q, k, v):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = self.scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = self.scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            outs = []
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                q1 = q_chunk[:, :, ::50, :]
                sim = q1 @ k.transpose(-1, -2)
                sim = torch.mean(sim, -2)
                topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
                topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
                v0 = torch.gather(v, dim=-2, index=topk_ind)
                k0 = torch.gather(k, dim=-2, index=topk_ind)
                out = self.scaled_dot_product_attention(q_chunk, k0, v0)
                outs.append(out)
                start += count
            out = torch.cat(outs, dim=-2)
        self.topk = False
        return out
def generate_dense_grid_points(
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        octree_resolution: int,
        indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    """
    修复维度问题的PyTorch实现
    作用是识别 3D 体素中 “靠近物体表面” 的体素（通过判断体素与其 6 个邻居的符号一致性），
    为分层采样提供 “高价值区域”（表面区域比背景区域更重要）。
    即：
    输入一个 3D 体素张量（input_tensor），输出一个掩码（0/1），其中1 表示该体素靠近表面，0 表示远离表面或无效。


    Args:
        input_tensor: shape [D, D, D], torch.float16
        alpha: 标量偏移值
    Returns:
        mask: shape [D, D, D], torch.int32 表面掩码
    """
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # 添加偏移并处理无效值
    val = input_tensor + alpha
    valid_mask = val > -9000  # 假设-9000是无效值

    # 改进的邻居获取函数（保持维度一致）
    def get_neighbor(t, shift, axis):
        """根据指定轴进行位移并保持维度一致"""
        if shift == 0:
            return t.clone()

        # 确定填充轴（输入为[D, D, D]对应z,y,x轴）
        pad_dims = [0, 0, 0, 0, 0, 0]  # 格式：[x前，x后，y前，y后，z前，z后]

        # 根据轴类型设置填充
        if axis == 0:  # x轴（最后一个维度）
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y轴（中间维度）
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z轴（第一个维度）
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # 执行填充（添加batch和channel维度适配F.pad）
        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')  # 反转顺序适配F.pad

        # 构建动态切片索引
        slice_dims = [slice(None)] * 3  # 初始化为全切片
        if axis == 0:  # x轴（dim=2）
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y轴（dim=1）
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z轴（dim=0）
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # 应用切片并恢复维度
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # 获取各方向邻居（确保维度一致）
    left = get_neighbor(val, 1, axis=0)  # x方向
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y方向
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z方向
    up = get_neighbor(val, -1, axis=2)

    # 处理边界无效值（使用where保持维度一致）
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # 计算符号一致性（转换为float32确保精度）
    sign = torch.sign(val.to(torch.float32))
    neighbors_sign = torch.stack([
        torch.sign(left.to(torch.float32)),
        torch.sign(right.to(torch.float32)),
        torch.sign(back.to(torch.float32)),
        torch.sign(front.to(torch.float32)),
        torch.sign(down.to(torch.float32)),
        torch.sign(up.to(torch.float32))
    ], dim=0)

    # 检查所有符号是否一致
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # 生成最终掩码
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


class FlashVDMVolumeDecoding:
    """
    https://github.com/Tencent-Hunyuan/FlashVDM/blob/main/flashvdm_decoder/volume_decoders.py
    用于将潜在向量（latents）通过几何解码器（geo_decoder）转换为 3D 体素数据
    引入 “迷你网格分组” 和 “查询点聚类”，适合需要注意力机制的复杂几何解码器。
    """
    def __init__(self):
        self.processor = FlashVDMCrossAttentionProcessor()  # 只保留速度快的， FlashVDMTopMCrossAttentionProcessor()去除

    @torch.no_grad()
    def __call__(
            self,
            latents: torch.FloatTensor,
            geo_decoder: Callable,
            bounds: Union[Tuple[float], List[float], float] = 1.01,
            num_chunks: int = 10000,
            mc_level: float = 0.0,
            octree_resolution: int = None,
            min_resolution: int = 63,
            mini_grid_num: int = 4,
            enable_pbar: bool = True,
            **kwargs,
    ):
        processor = self.processor

        # 将普通点积计算替换成FlashVDMCrossAttentionProcessor
        geo_decoder.cross_attn_decoder.attn.attention.processor = processor

        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i, resolution in enumerate(resolutions[1:]):
            resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)

        log.info(f"Resolution: {resolutions}")

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

        grid_size = np.array(grid_size)

        # 2. latents to 3d volume
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
        batch_size = latents.shape[0]
        mini_grid_size = xyz_samples.shape[0] // mini_grid_num
        xyz_samples = xyz_samples.view(
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size, 3
        ).permute(
            0, 2, 4, 1, 3, 5, 6
        ).reshape(
            -1, mini_grid_size * mini_grid_size * mini_grid_size, 3
        )
        batch_logits = []
        num_batchs = max(num_chunks // xyz_samples.shape[1], 1)
        for start in tqdm(range(0, xyz_samples.shape[0], num_batchs),
                          desc=f"Decoding", disable=not enable_pbar):
            queries = xyz_samples[start: start + num_batchs, :]
            batch = queries.shape[0]
            batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch)
            processor.topk = True
            logits = geo_decoder(queries=queries, latents=batch_latents)
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=0).reshape(
            mini_grid_num, mini_grid_num, mini_grid_num,
            mini_grid_size, mini_grid_size,
            mini_grid_size
        ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2])
        )

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=device)
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)

            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                           torch.tensor(bbox_min, dtype=torch.float32, device=device))

            query_grid_num = 6
            min_val = next_points.min(axis=0).values
            max_val = next_points.max(axis=0).values
            vol_queries_index = (next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)
            index = torch.floor(vol_queries_index).long()
            index = index[..., 0] * (query_grid_num ** 2) + index[..., 1] * query_grid_num + index[..., 2]
            index = index.sort()
            next_points = next_points[index.indices].unsqueeze(0).contiguous()
            unique_values = torch.unique(index.values, return_counts=True)
            grid_logits = torch.zeros((next_points.shape[1]), dtype=latents.dtype, device=latents.device)
            input_grid = [[], []]
            logits_grid_list = []
            start_num = 0
            sum_num = 0
            for grid_index, count in zip(unique_values[0].cpu().tolist(), unique_values[1].cpu().tolist()):
                if sum_num + count < num_chunks or sum_num == 0:
                    sum_num += count
                    input_grid[0].append(grid_index)
                    input_grid[1].append(count)
                else:
                    processor.topk = input_grid
                    logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                    start_num = start_num + sum_num
                    logits_grid_list.append(logits_grid)
                    input_grid = [[grid_index], [count]]
                    sum_num = count
            if sum_num > 0:
                processor.topk = input_grid
                logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                logits_grid_list.append(logits_grid)
            logits_grid = torch.cat(logits_grid_list, dim=1)
            grid_logits[index.indices] = logits_grid.squeeze(0).squeeze(-1)
            next_logits[nidx] = grid_logits
            grid_logits = next_logits.unsqueeze(0)

        grid_logits[grid_logits == -10000.] = float('nan')

        return grid_logits

class SurfaceExtractor:
    """ mc效果最好"""
    def __init__(self):
        super().__init__()

    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size
    def mc_extractor(self,grid_logit, mc_level, bounds, octree_resolution):
        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logit.cpu().numpy(),
            mc_level,
            method="lewiner"
            )
        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces
    def __call__(self, grid_logits, **kwargs):
        outputs = []
        B =grid_logits.shape[0]
        for i in range(B):
            try:
                vertices, faces = self.mc_extractor(grid_logits[i], **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append([vertices, faces])

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)

        return outputs


class Decode(nn.Module):
    def __init__(self,num_latents:int,width,heads):
        super(Decode, self).__init__()
        self.volume_decoder = FlashVDMVolumeDecoding()
        self.surface_extractor = SurfaceExtractor()
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=True)
        self.geo_decoder = CrossAttentionDecoder(
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width ,
            heads=heads,
            qkv_bias=True,
            qk_norm=False,
        )

        self.bounds=1.01
        self.mc_level=0.0
        self.num_chunks=20000
        self.octree_resolution=256
        self.enable_pbar=True

    def forward(self,latents):
        """
        将特征向量输出为mesh
        Args:
            latents: 潜在特征向量

        Returns:
            mesh:按照batch_size依次返回[v,f]

        """

        grid_logits = self.volume_decoder(latents,
                                              self.geo_decoder,
                                              bounds=self.bounds,
                                              octree_resolution=self.octree_resolution,
                                              num_chunks=self.num_chunks,
                                              enable_pbar=self.enable_pbar,
                                              mc_level=self.mc_level)
        outputs = self.surface_extractor(grid_logits,
                                         mc_level=self.mc_level,
                                         bounds=self.bounds,
                                         octree_resolution=self.octree_resolution)
        return outputs




if __name__ == "__main__":
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device} ",torch.cuda.get_device_name(device))

    # 初始化模型
    num_latents = 256  # 潜在向量数量
    width = 256        # 解码器宽度
    heads = 8          # 注意力头数
    model = Decode(num_latents=num_latents, width=width, heads=heads).to(device)

    # 创建随机输入数据
    batch_size = 1
    latents = torch.randn(batch_size, num_latents, width).to(device)
    from sindre.deploy.check_tools import timeit
    # 测试模型
    print("开始生成网格...")
    try:
        with timeit("运行时间",True):
            with torch.no_grad():
                outputs = model(latents)
        # 检查输出
        if outputs and outputs[0] is not None:
            vertices, faces = outputs[0]
            print(f"生成成功！顶点数: {len(vertices)}, 面数: {len(faces)}")
        else:
            print("生成失败，输出为空")

    except Exception as e:
        print(f"生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()