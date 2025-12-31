from typing import Union, List, Tuple
import numpy as np
import torch
from einops import repeat
from tqdm import tqdm



class Latents2Meshes:
    def __init__(self,
                 bounds: Union[Tuple[float], List[float], float] = 1.1,
                 octree_depth: int = 8,
                 num_chunks: int = 20000,
                 mc_level: float = -1 / 512,
                 method="mc",
                 enable_half=True):
        self.mc_algo = method
        self.num_chunks = num_chunks #分割单位
        self.mc_level = mc_level
        self.mc=MarchingCubes()
        self.octree_depth=octree_depth
        self.enable_half = enable_half
        if self.mc_level ==0:
            print(f'使用软标签进行训练，使用sigmoid进行推理，使用0级行进立方体进行推理.')


        # 1. 生成查询点
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        self.bbox_min = np.array(bounds[0:3])
        self.bbox_max = np.array(bounds[3:6])
        self.bbox_size = self.bbox_max - self.bbox_min
        xyz_samples, self.grid_size, length = self.generate_dense_grid_points(
            bbox_min=self.bbox_min,
            bbox_max=self.bbox_max,
            octree_depth=self.octree_depth,
        )

        self.xyz_samples = torch.Tensor(xyz_samples).to(dtype=torch.float32)
        if self.enable_half:
            self.xyz_samples=self.xyz_samples.half()


    @staticmethod
    def generate_dense_grid_points(bbox_min: np.ndarray,
                                   bbox_max: np.ndarray,
                                   octree_depth: int,
                                   ):
        length = bbox_max - bbox_min
        num_cells = np.exp2(octree_depth)
        x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
        y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
        z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
        [xs, ys, zs] = np.meshgrid(x, y, z, indexing="ij")
        xyz = np.stack((xs, ys, zs), axis=-1)
        xyz = xyz.reshape(-1, 3)
        grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

        return xyz, grid_size, length



    @staticmethod
    def center_vertices(vertices):
        """Translate the vertices so that bounding box is centered at zero."""
        vert_min = vertices.min(dim=0)[0]
        vert_max = vertices.max(dim=0)[0]
        vert_center = 0.5 * (vert_min + vert_max)
        return vertices - vert_center
    
    def run(self,latents,fun_callback,**kwargs):
        batch_logits = []
        batch_size = latents.shape[0]
        device = latents.device
        for start in tqdm(range(0, self.xyz_samples.shape[0], self.num_chunks),desc=f"{self.mc_algo} Level { self.mc_level}:"):
            queries = self.xyz_samples[start: start + self.num_chunks, :].to(device)
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = fun_callback(batch_queries.to(latents.dtype), latents=latents,**kwargs)
            if self.mc_level == 0:
                logits = torch.sigmoid(logits) * 2 - 1
            batch_logits.append(logits)
        grid_logits = torch.cat(batch_logits, dim=1)
        grid_logits = grid_logits.view((batch_size, self.grid_size[0], self.grid_size[1], self.grid_size[2])).float()

        # 重建mesh
        meshes=[]
        for i in range(batch_size):
            if self.mc_algo == 'mc':
                vertices,faces=self.mc.apply_skimage( grid_logits[i].detach().cpu().numpy(),self.mc_level)
                vertices = vertices / (np.array(self.grid_size)-1) * self.bbox_size +self.bbox_min
            elif self.mc_algo == 'dmc':
                octree_resolution = 2 ** self.octree_depth
                sdf = -grid_logits[i] / octree_resolution
                vertices,faces=self.mc.apply_diso(sdf)
                vertices = self.center_vertices(vertices).detach().cpu().numpy()
                faces = faces.detach().cpu().numpy()[:, ::-1]
            else:
                raise ValueError(f"mc_algo {self.mc_algo} not supported.")
            meshes.append([vertices,faces])
        return meshes



class GetSDF:
    """封装多种 SDF（有符号距离函数）计算实现的工具类。

    核心功能：从三角网格（顶点+面）计算一组查询点的有符号距离（SDF），
    支持不同后端（cubvh/pcu/open3d/pysdf），适配 NumPy 数组/PyTorch 张量、CPU/GPU 等不同场景。

    关键说明：
    - SDF 定义：查询点到网格表面的最短距离，点在网格内部为负值，外部为正值，在表面上为 0；
    - 不同后端适配性：
      - cubvh：CUDA 加速，支持 NumPy/PyTorch 输入，计算最快；
      - pcu：CPU/GPU 兼容，仅支持 NumPy 输入，返回额外面索引/重心坐标；
      - open3d：张量版接口，支持 PyTorch 输入，可指定设备（CPU/GPU）；
      - pysdf：轻量级实现，仅支持 NumPy 输入，适合小规模数据。
    """

    def __init__(self, method="cubvh"):
        """初始化 SDF 计算工具类，指定默认计算后端。

        Args:
            method (str, optional): 默认使用的 SDF 计算方法，可选值：
                - "cubvh"（默认）：CUDA 加速的 BVH 方法，效率最高；
                - "pcu"：point_cloud_utils 实现；
                - "o3d"：Open3D 张量版实现；
                - "pysdf"：pysdf 轻量级实现。
        """
        pass

    def apply_cubvh(self, vertices, faces, query_pts):
        """基于 cubvh 库（CUDA 加速 BVH）计算 SDF，支持 NumPy 数组/PyTorch 张量输入。

        该方法是性能最优的实现，适合大规模查询点的 SDF 计算（如百万级点云）。

        Args:
            vertices (np.ndarray | torch.Tensor): 网格顶点数组/张量，形状为 (N, 3)，N 为顶点数；
            faces (np.ndarray | torch.Tensor): 网格面索引数组/张量，形状为 (M, 3)，M 为面数；
            query_pts (np.ndarray | torch.Tensor): 查询点数组/张量，形状为 (K, 3)，K 为查询点数。

        Returns:
            np.ndarray | torch.Tensor: 每个查询点的 SDF 值，形状为 (K,)，
                数据类型/设备与输入 query_pts 一致（输入 Torch 则返回 Torch，输入 NumPy 则返回 NumPy）。

        Note:
            - cubvh 库需提前安装（需 CUDA 环境，支持 PyTorch 张量直接计算）；
            - return_uvw=False 表示不返回重心坐标，仅返回 SDF，若需额外信息可改为 True；
            - 输入张量需在 CUDA 设备上以获得加速效果（CPU 输入会自动降级为 CPU 计算）。
        """
        import cubvh
        fbvh = cubvh.cuBVH(vertices, faces)
        sdf, face_id, uvw = fbvh.signed_distance(query_pts, return_uvw=False)
        return sdf

    def apply_pcu(self, vertices: np.ndarray, faces: np.ndarray, query_pts: np.ndarray):
        """基于 point_cloud_utils（pcu）库计算 SDF，仅支持 NumPy 数组输入。

        除 SDF 外可返回网格最近面索引和重心坐标（注释中已保留计算逻辑，仅返回 SDF），适合中小规模计算。

        Args:
            vertices (np.ndarray): 网格顶点数组，形状为 (N, 3)，N 为顶点数；
            faces (np.ndarray): 网格面索引数组，形状为 (M, 3)，M 为面数；
            query_pts (np.ndarray): 查询点数组，形状为 (K, 3)，K 为查询点数。

        Returns:
            np.ndarray: 每个查询点的 SDF 值，形状为 (K,)，数据类型为 float32/float64（与输入一致）。

        Note:
            - point_cloud_utils 库需提前安装（pip install point_cloud_utils）；
            - 若需面索引/重心坐标，可修改返回值为 `return sdf, face_ids, barycentric_coords`；
            - 支持 CPU/GPU 计算（需编译 pcu 时开启 CUDA 支持）。
        """
        import point_cloud_utils as pcu
        # 计算 sdf、网格中最近面的索引以及重心坐标
        sdf, face_ids, barycentric_coords = pcu.signed_distance_to_mesh(query_pts, vertices, faces)
        return sdf

    def apply_o3d(self, vertices: torch.Tensor, faces: torch.Tensor, query_pts: torch.Tensor, device="CPU:0"):
        """基于 Open3D 张量版（t.geometry）计算 SDF，仅支持 PyTorch 张量输入，可指定计算设备。

        适配 Open3D 生态，支持 CPU/GPU 灵活切换，适合与 Open3D 可视化/几何处理流程集成。

        Args:
            vertices (torch.Tensor): 网格顶点张量，形状为 (N, 3)，N 为顶点数；
            faces (torch.Tensor): 网格面索引张量，形状为 (M, 3)，M 为面数；
            query_pts (torch.Tensor): 查询点张量，形状为 (K, 3)，K 为查询点数；
            device (str, optional): 计算设备，格式为 "CPU:0"（默认）、"CUDA:0"、"CUDA:1" 等。

        Returns:
            o3d.core.Tensor: 每个查询点的 SDF 值，形状为 (K,)，数据类型为 float32，设备与指定 device 一致。

        Note:
            - Open3D 需安装 0.18+ 版本（支持张量版 RaycastingScene）；
            - 输入 PyTorch 张量会自动转换为 Open3D 张量，返回值需转为 PyTorch 可执行 `sdf.to_torch()`；
            - 面索引需为 int32 类型，顶点需为 float32 类型（代码已自动转换）。
        """
        import open3d as o3d
        scene = o3d.t.geometry.RaycastingScene()
        device = o3d.core.Device(device)
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32
        mesh = o3d.t.geometry.TriangleMesh(device)
        mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=dtype_f, device=device)
        mesh.triangle.indices = o3d.core.Tensor(faces, dtype=dtype_i, device=device)
        query_pts = o3d.core.Tensor(query_pts, dtype=dtype_f, device=device)
        scene.add_triangles(mesh)
        sdf = scene.compute_signed_distance(query_pts)
        return sdf

    def apply_pysdf(self, vertices: np.ndarray, faces: np.ndarray, query_pts: np.ndarray):
        """基于 pysdf 库计算 SDF，轻量级实现，仅支持 NumPy 数组输入。

        无需复杂依赖，适合快速验证、小规模查询点计算（如万级点云）。

        Args:
            vertices (np.ndarray): 网格顶点数组，形状为 (N, 3)，N 为顶点数；
            faces (np.ndarray): 网格面索引数组，形状为 (M, 3)，M 为面数；
            query_pts (np.ndarray): 查询点数组，形状为 (K, 3)，K 为查询点数。

        Returns:
            np.ndarray: 每个查询点的 SDF 值，形状为 (K,)，数据类型为 float32。

        Note:
            - pysdf 库需提前安装（pip install pysdf）；
            - 内部会构建网格的 SDF 缓存，首次调用稍慢，重复查询同一网格会更快；
            - 不支持 GPU 加速，大规模计算效率低于 cubvh/pcu。
        """
        from pysdf import SDF
        fsdf = SDF(vertices, faces)
        sdf = fsdf(query_pts)
        return sdf


class MarchingCubes:
    """封装多种 Marching Cubes（移动立方体）算法实现的工具类。

    核心功能：从 SDF（有符号距离函数）或占据场（occupancy）张量/数组重建 3D 网格模型，
    支持不同后端（diso/PyTorch3D/pymc/skimage），适配可微分/非可微分、张量/数组等不同场景。

    关键说明：
    - SDF（有符号距离函数）：值为负表示点在物体内部，正为外部，0 为表面；
    - 占据场（occ）：值为 1 表示点在物体内部，0 为外部，需转换为 `occ * -1` 后再输入（否则重建网格方向反转）。
    """

    def __init__(self):
        """初始化 Marching Cubes 工具类，延迟初始化后端组件（避免提前加载依赖）。"""
        pass

    def apply_diso(self, sdf: torch.Tensor, deform=None, isovalue=0, **kwargs):
        """基于 diso 库的可微分移动立方体（DiffDMC）实现，支持形变场，适用于可微渲染/重建场景。

        该方法返回的顶点/面张量支持反向传播，适合需要端到端训练的 3D 重建任务。

        Args:
            sdf (torch.Tensor): SDF 张量（或转换后的占据场），形状通常为 (D, H, W) 或 (B, D, H, W)，
                其中 D/H/W 为 3D 网格的深度/高度/宽度，B 为批次维度。
            deform (torch.Tensor, optional): 形变场张量，用于对 SDF 网格进行空间形变，默认为 None（无形变）。
            isovalue (float, optional): 等值面值，对应 SDF 表面的阈值，默认为 0（标准 SDF 表面）。
            **kwargs: 传递给 diso.DiffDMC 的额外参数（如网格分辨率、梯度计算方式等）。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - vertices: 重建网格的顶点张量，形状为 (N, 3)，N 为顶点数；
                - faces: 重建网格的面索引张量，形状为 (M, 3)，M 为面数（每个面对应 3 个顶点索引）。

        Note:
            - 若输入为占据场（occ），需先执行 `sdf = occ * -1`，否则重建的网格方向会反转；
            - diso 库依赖需提前安装，且仅支持 PyTorch 张量输入；
            - 该方法会自动将 DiffDMC 实例移动到 SDF 张量所在设备（GPU/CPU）。
        """
        from diso import DiffDMC
        # 延迟初始化 DiffDMC 实例，确保设备与 SDF 一致
        if not hasattr(self, "diffdmc"):
            self.diffdmc = DiffDMC(dtype=torch.float32).to(sdf.device)
        vertices, faces = self.diffdmc(sdf, deform, isovalue=isovalue, **kwargs)
        return vertices, faces

    def apply_pytorch3d(self, sdf: torch.Tensor, isovalue=0, **kwargs):
        """基于 PyTorch3D 库的 cubify 接口实现移动立方体，返回 PyTorch3D 原生 Mesh 对象。

        适用于与 PyTorch3D 生态集成的场景（如网格渲染、几何变换、损失计算）。

        Args:
            sdf (torch.Tensor): SDF 张量，形状为 (B, D, H, W)（批次）或 (D, H, W)（单例），
                D/H/W 为 3D 网格的深度/高度/宽度。
            isovalue (float, optional): 等值面值，对应 SDF 表面的阈值，默认为 0。
            **kwargs: 传递给 pytorch3d.ops.cubify 的额外参数（如边距、插值方式等）。

        Returns:
            pytorch3d.structures.Meshes: PyTorch3D 网格对象，包含批次内所有重建的网格，
                可直接用于 PyTorch3D 的渲染、采样等操作。

        Note:
            - PyTorch3D 库需提前安装（支持 PyTorch 2.0+）；
            - 输入张量需为 float32/float64 类型，建议与 PyTorch3D 默认 dtype 一致。
        """
        from pytorch3d.ops import cubify
        meshes = cubify(sdf, isovalue=isovalue,** kwargs)
        return meshes

    def apply_pymc(self, sdf: np.ndarray, isovalue=0,smooth=False):
        """基于 marching_cubes 库（pymc）的经典移动立方体实现，仅支持 NumPy 数组输入。

        适用于非可微分的离线 3D 网格重建场景，计算速度快，兼容性好。

        Args:
            sdf (np.ndarray): SDF 数组（或转换后的占据场），形状为 (D, H, W)，
                D/H/W 为 3D 网格的深度/高度/宽度。
            isovalue (float, optional): 等值面值，对应 SDF 表面的阈值，默认为 0。
            smooth: 是否对输入进行自动平滑,默认False

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - vertices: 重建网格的顶点数组，形状为 (N, 3)，N 为顶点数；
                - triangles: 重建网格的三角面索引数组，形状为 (M, 3)，M 为面数。

        Note:
            - 输入需为 NumPy 数组（不支持 PyTorch 张量），若为张量需先执行 `sdf = sdf.cpu().numpy()`；
            - marching_cubes 库需提前安装（pip install marching-cubes）。
        """
        import marching_cubes as mcubes
        if smooth:
            sdf = mcubes.smooth(sdf)
        vertices, triangles = mcubes.marching_cubes(sdf, isovalue=isovalue)
        return vertices, triangles

    def apply_skimage(self, sdf: np.ndarray, isovalue=0,method="lewiner", **kwargs):
        """基于 scikit-image 库的移动立方体实现，返回顶点、面、法向量和值，适配可视化/分析场景。

        额外返回法向量和表面值，适合网格后处理、可视化（如 Matplotlib/Open3D 渲染）。

        Args:
            sdf (np.ndarray): SDF 数组（或转换后的占据场），形状为 (D, H, W)，
                D/H/W 为 3D 网格的深度/高度/宽度。
            isovalue (float, optional): 等值面值，对应 SDF 表面的阈值，默认为 0。
            **kwargs: 传递给 skimage.measure.marching_cubes 的额外参数（如步长、梯度计算方式等）。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - verts: 重建网格的顶点数组，形状为 (N, 3)，N 为顶点数；
                - faces: 重建网格的三角面索引数组（已反转索引顺序），形状为 (M, 3)，M 为面数；
        Note:
            - 输入需为 NumPy 数组（不支持 PyTorch 张量），若为张量需先执行 `sdf = sdf.cpu().numpy()`；
            - 面索引已执行 `faces[:, [2, 1, 0]]` 反转，确保面的朝向符合右手定则；
            - scikit-image 库需提前安装（pip install scikit-image）。
        """
        from skimage import measure
        verts, faces, normals, values = measure.marching_cubes(sdf,level=isovalue,method=method,** kwargs)
        # 反转面索引顺序，保证面朝向正确（右手定则）
        faces = faces[:, [2, 1, 0]]
        return verts, faces



class DiagonalGaussianDistribution(object):
    """
    对对角高斯分布（各维度独立的多元高斯分布）的封装，主要用于变分自编码器（VAE）、扩散模型等生成模型中，处理分布的采样、KL 散度计算、负对数似然（NLL）计算等核心操作
    """
    def __init__(self, parameters: Union[torch.Tensor, List[torch.Tensor]], deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                        + self.var - 1.0 - self.logvar,
                                        dim=dims)
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=dims)

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class MeshSharpSDFProcessor:
    """
    网格锐边与SDF（有符号距离函数）的采样处理类

    功能说明：
    1. 对输入网格进行归一化处理，验证网格水密性；
    2. 实现网格表面、锐边的点采样，以及空间点/近表面点的SDF计算；
    3. 对采样点执行最远点采样（FPS），生成最终的特征数组。
    """
    def __init__(self,vertices,faces):
        """
        初始化网格SDF处理器

        Args:
            vertices (np.ndarray): 网格顶点数组，形状为(N, 3)，N为顶点数
            faces (np.ndarray): 网格面索引数组，形状为(M, 3)，M为面数

        Raises:
            AssertionError: 若输入网格非水密（watertight）时触发
        """
        # 采样数量配置（语义化变量名）
        self.fps_sample_count = 10000        # FPS采样点数
        self.surface_sample_count = 200000   # 表面采样总数
        self.sharp_sample_count = 200000     # 锐边采样总数
        self.space_sample_count = 200000     # 空间随机采样总数


        # 网格归一化与初始化
        import trimesh
        normalized_vertices = self.normalize_vertices(vertices)
        self.mesh = trimesh.Trimesh(normalized_vertices, faces)
        assert self.mesh.is_watertight(), "输入网格必须是水密的（watertight）"
        assert len(self.mesh.vertices)>1000, "顶点必须大于1k"

        # 网格基础属性
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces
        self.face_normals = self.mesh.face_normals

        # 初始化CUDA加速的BVH树（用于SDF快速计算）
        import cubvh #pip install git+https://github.com/ashawkey/cubvh --no-build-isolation
        self.bvh_tree = cubvh.cuBVH(self.vertices, self.faces)



    def normalize_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        将网格顶点归一化到[-1, 1]立方体空间

        Args:
            vertices (np.ndarray): 输入顶点数组，形状为(N, 3)

        Returns:
            np.ndarray: 归一化后的顶点数组，形状为(N, 3)
        """
        bb_min = vertices.min(axis=0)
        bb_max = vertices.max(axis=0)
        center = (bb_min + bb_max) / 2.0
        scale = 2.0 / (bb_max - bb_min).max()  # 最大轴长缩放到2（对应[-1,1]）
        normalized_vertices = (vertices - center) * scale
        return normalized_vertices

    def sample_surface_points(self) -> np.ndarray:
        """
        网格表面随机采样点，并拼接法向量生成特征数组

        Returns:
            np.ndarray: 表面采样特征数组，形状为(采样数, 6)，列顺序为(x,y,z,nx,ny,nz)
        """
        surface_points, face_indices = self.mesh.sample(
            self.surface_sample_count,
            return_index=True
        )
        face_normals = self.face_normals[face_indices]
        surface_feat = np.concatenate([surface_points, face_normals], axis=1)
        return surface_feat

    def sample_sharp_edges_points(self) -> np.ndarray:
        """
        网格锐边采样点，并拼接法向量生成特征数组

        Returns:
            np.ndarray: 锐边采样特征数组，形状为(采样数, 6)，列顺序为(x,y,z,nx,ny,nz)
        """
        from sindre.utils3d.sample import sample_mesh_sharp_edges
        sharp_points, sharp_normals = sample_mesh_sharp_edges(
            self.vertices,
            self.faces,
            self.sharp_sample_count
        )
        sharp_feat = np.concatenate([sharp_points, sharp_normals], axis=1)
        return sharp_feat



    def get_rand_points_sdf(self, surface_points: np.ndarray) -> np.ndarray:
        """
        计算空间随机点与表面近点的SDF，并生成特征数组

        Args:
            surface_points (np.ndarray): 表面采样点数组，形状为(N, 3)

        Returns:
            np.ndarray: 随机点SDF特征数组，形状为(总点数, 4)，列顺序为(x,y,z,sdf)
        """
        # 空间均匀采样点（范围略大于[-1,1]以覆盖边界）
        space_points = np.random.uniform(-1.05, 1.05, (self.space_sample_count, 3))

        # 表面近点采样（不同高斯噪声尺度）
        near_surface_points_list = [
            surface_points + np.random.normal(scale=0.001, size=surface_points.shape),
            surface_points + np.random.normal(scale=0.005, size=surface_points.shape)
        ]
        near_surface_points = np.concatenate(near_surface_points_list, axis=0)

        # 合并所有点并计算SDF
        all_rand_points = np.concatenate([near_surface_points, space_points], axis=0)
        rand_sdf = self.bvh_tree.signed_distance(all_rand_points)[0].reshape(-1, 1)
        rand_sdf_feat = np.concatenate([all_rand_points, rand_sdf], axis=1)

        return rand_sdf_feat

    def get_sharp_points_sdf(self, sharp_points: np.ndarray) -> np.ndarray:
        """
        计算锐边近表面点的SDF，并生成特征数组

        Args:
            sharp_points (np.ndarray): 锐边采样点数组，形状为(N, 3)

        Returns:
            np.ndarray: 锐边点SDF特征数组，形状为(总点数, 4)，列顺序为(x,y,z,sdf)
        """
        # 锐边近点采样（多种高斯噪声尺度）
        sharp_near_surface_points_list = [
            sharp_points + np.random.normal(scale=0.001, size=sharp_points.shape),
            sharp_points + np.random.normal(scale=0.005, size=sharp_points.shape),
            sharp_points + np.random.normal(scale=0.007, size=sharp_points.shape),
            sharp_points + np.random.normal(scale=0.01, size=sharp_points.shape)
        ]
        sharp_near_surface_points = np.concatenate(sharp_near_surface_points_list, axis=0)

        # 计算SDF并拼接特征
        sharp_sdf = self.bvh_tree.signed_distance(sharp_near_surface_points)[0].reshape(-1, 1)
        sharp_sdf_feat = np.concatenate([sharp_near_surface_points, sharp_sdf], axis=1)

        return sharp_sdf_feat

    def apply_farthest_point_sampling(self, points_feat: np.ndarray) -> np.ndarray:
        """
        对输入特征数组的点坐标执行最远点采样（FPS）

        Args:
            points_feat (np.ndarray): 输入特征数组，形状为(N, C)，前3列为点坐标(x,y,z)

        Returns:
            np.ndarray: FPS采样后的特征数组，形状为(fps_sample_count, C)
        """
        from sindre.utils3d.sample import sample_pcd_farthest
        points = points_feat[..., :3]
        fps_indices = sample_pcd_farthest(points, self.fps_sample_count)
        fps_feat = points_feat[fps_indices]
        return fps_feat

    def __call__(self) -> dict:
        """
        执行完整的网格采样与SDF计算流程

        Returns:
            dict: 包含各类采样特征的字典，键说明：
                - fps_surface_feat: FPS采样后的表面特征数组（float32）
                - fps_sharp_feat: FPS采样后的锐边特征数组（float32）
                - rand_sdf_feat: 随机点SDF特征数组（float32）
                - sharp_sdf_feat: 锐边近点SDF特征数组（float32）

        Raises:
            AssertionError: 若任意特征数组包含NaN值时触发（注：原逻辑assert np.isnan().all()为笔误，已修正为np.isnan().any()）
        """
        # 表面采样与处理
        surface_feat = self.sample_surface_points()
        # 锐边采样与处理
        sharp_feat = self.sample_sharp_edges_points()
        # 随机点SDF计算
        rand_sdf_feat = self.get_rand_points_sdf(surface_feat[..., :3])
        # 锐边点SDF计算
        sharp_sdf_feat = self.get_sharp_points_sdf(sharp_feat[..., :3])

        # FPS采样
        fps_surface_feat = self.apply_farthest_point_sampling(surface_feat)
        fps_sharp_feat = self.apply_farthest_point_sampling(sharp_feat)

        # 检查NaN值
        assert not np.isnan(surface_feat).any(), "表面采样特征包含NaN值"
        assert not np.isnan(sharp_feat).any(), "锐边采样特征包含NaN值"
        assert not np.isnan(rand_sdf_feat).any(), "随机点SDF特征包含NaN值"
        assert not np.isnan(sharp_sdf_feat).any(), "锐边点SDF特征包含NaN值"
        assert not np.isnan(fps_surface_feat).any(), "FPS表面特征包含NaN值"
        assert not np.isnan(fps_sharp_feat).any(), "FPS锐边特征包含NaN值"

        # 整理输出
        output_dict = {
            "fps_surface_feat": fps_surface_feat.astype(np.float32),
            "fps_sharp_feat": fps_sharp_feat.astype(np.float32),
            "rand_sdf_feat": rand_sdf_feat.astype(np.float32),
            "sharp_sdf_feat": sharp_sdf_feat.astype(np.float32)
        }

        return output_dict

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