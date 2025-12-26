

import numpy as np
import torch
try:
    import igl # pip install libigl==2.6.1
    import mcubes #  pip install pymcubes
    import nvdiffrast.torch as dr # pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
except ImportError as e:
    print(" pip install libigl==2.6.1 pymcubes ;pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation ")
from torch.nn.functional import grid_sample


def remesh_by_igl(V, F, epsilon = 2.0/256, grid_res = 256):
    """
    基于符号距离函数（SDF）和移动立方体（Marching Cubes）算法，将非水密3D网格转为水密网格的函数。
    核心思路：通过体素化方式构建连续的SDF场，再提取等值面生成无孔洞、无裂缝的水密网格。

    参数:
        V (np.ndarray): 输入网格的顶点数组，形状为 (n, 3)，n为顶点数，每行是x/y/z坐标。
        F (np.ndarray): 输入网格的面数组，形状为 (m, 3)，m为三角面数，每行是顶点索引（三角面）。
        epsilon (float, 可选): SDF阈值，控制水密化的容差（epsilon - |SDF| 作为等值面提取条件），默认2.0/256。
        grid_res (int, 可选): 体素网格的分辨率（x/y/z轴均使用该分辨率），默认256。

    返回:
        mc_verts (np.ndarray): 水密网格的顶点数组，形状为 (k, 3)，k为生成的水密网格顶点数。
        mc_faces (np.ndarray): 水密网格的面数组，形状为 (l, 3)，l为生成的水密网格三角面数。

    流程说明:
        1. 计算输入网格的包围盒并添加padding，避免等值面提取时截断边界；
        2. 构建均匀体素网格，生成所有网格点的三维坐标；
        3. 基于伪法向量计算网格点到输入网格的符号距离（SDF）；
        4. 用移动立方体算法提取 epsilon - |SDF| = 0 的等值面，生成水密网格。
    """
    # 1. 计算输入网格的包围盒（最小/最大顶点坐标）
    min_corner = V.min(axis=0)  # 包围盒最小角点，形状(3,)
    max_corner = V.max(axis=0)  # 包围盒最大角点，形状(3,)
    # 给包围盒添加5%的padding，避免等值面提取时截断网格边界
    padding = 0.05 * (max_corner - min_corner)
    min_corner -= padding
    max_corner += padding

    # 2. 创建均匀体素网格（grid_res×grid_res×grid_res）
    x = np.linspace(min_corner[0], max_corner[0], grid_res)  # x轴网格坐标
    y = np.linspace(min_corner[1], max_corner[1], grid_res)  # y轴网格坐标
    z = np.linspace(min_corner[2], max_corner[2], grid_res)  # z轴网格坐标
    # 生成三维网格点（indexing='ij'保证x/y/z轴与网格维度对应）
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # 展平网格点为N×3的数组（N=grid_res^3），每行是一个体素网格点的坐标
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    # 3. 计算网格点到输入网格的符号距离（SDF）
    # sign_type=PSEUDONORMAL：用伪法向量计算SDF，提升非水密网格的距离计算鲁棒性
    sdf, _, _ = igl.signed_distance(
        grid_points, V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL
    )
 
    # 4. 移动立方体（Marching Cubes）提取等值面，生成水密网格
    # 等值面条件：epsilon - |sdf| = 0 → |sdf| = epsilon
    # 输入：SDF场、体素网格点、网格分辨率、等值面阈值0.0
    mc_verts, mc_faces = igl.marching_cubes(epsilon - np.abs(sdf), grid_points, grid_res, grid_res, grid_res, 0.0)

    # 返回水密网格的顶点和面
    # mc_verts: (k×3) 水密网格顶点数组
    # mc_faces: (l×3) 水密网格三角面数组
    return mc_verts, mc_faces



def remesh_by_sdf(vertices,faces,restore=True, resolution: int = 256,device="cuda"):
    """
    全GPU端实现水密网格重构,输出可能会产生连通体

    Args:
        vertices: 输入网格顶点 (V, 3)
        faces: 输入网格面索引 (F, 3)
        restore:如果还原则还原到输出空间下，否则输出在[-1,1]下
        resolution: 密集网格分辨率
        device: 计算设备（固定为cuda）

    Returns:
        watertight_vertices: 水密网格顶点 (V', 3)
        watertight_faces: 水密网格面索引 (F', 3)
    """


    is_np=False
    if isinstance(vertices,np.ndarray):
        vertices = torch.from_numpy(np.ascontiguousarray(vertices)).to(device=device)
        faces = torch.from_numpy(np.ascontiguousarray(faces)).to(device=device)
        is_np=True


    # 1. 输入网格归一化到[-1, 1]（GPU端）
    bbmin = vertices.min(dim=0)[0]
    bbmax = vertices.max(dim=0)[0]
    center = (bbmin + bbmax) / 2.0
    scale = 2.0 / (bbmax - bbmin).max()
    vertices_norm = (vertices - center) * scale  # 归一化到[-1, 1]

    # 2. GPU端生成密集网格点
    bbox_min = torch.tensor([-1.05, -1.05, -1.05], dtype=torch.float32, device=device)
    bbox_max = torch.tensor([1.05, 1.05, 1.05], dtype=torch.float32, device=device)
    x = torch.linspace(bbox_min[0], bbox_max[0], resolution + 1, dtype=torch.float32, device=device)
    y = torch.linspace(bbox_min[1], bbox_max[1], resolution + 1, dtype=torch.float32, device=device)
    z = torch.linspace(bbox_min[2], bbox_max[2], resolution + 1, dtype=torch.float32, device=device)
    xs, ys, zs = torch.meshgrid(x, y, z, indexing="ij")
    grid_xyz = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
    grid_size = [resolution + 1, resolution + 1, resolution + 1]

    # 3. GPU加速计算UDF（unsigned distance function）
    import cubvh
    bvh = cubvh.cuBVH(vertices_norm, faces)
    grid_udf, _, _ = bvh.unsigned_distance(grid_xyz, return_uvw=False)
    grid_udf = grid_udf.view(grid_size[0], grid_size[1], grid_size[2])  # 重构为3D网格

    # 4. DiffDMC提取等值面（GPU端）
    from diso import DiffDMC
    eps = 2.0 / resolution  # 等值面阈值
    diffdmc = DiffDMC(dtype=torch.float32).to(device)
    mesh_verts, mesh_faces = diffdmc(grid_udf, isovalue=eps, normalize=False)

    # 5. 网格坐标反归一化到原包围盒（GPU端）
    bbox_size = bbox_max - bbox_min
    mesh_verts = (mesh_verts + 1.0) / grid_size[0] * bbox_size[0] + bbox_min[0]
    if restore:
        # 还原到输入空间
        mesh_verts=mesh_verts/scale+center

    if is_np:
        return mesh_verts.cpu().numpy(),mesh_faces.cpu().numpy()
    return mesh_verts, mesh_faces





class remesh_watertight_by_views:
    """
    网格水密化处理类：将非水密网格的顶点/面片转换为水密网格，仅保留外表面
    核心原理：多视角渲染+缠绕数判断内外 → UDF转SDF → Marching Cubes重建水密网格
    
    Note:
        watertight_processor = remesh_watertight(
                        grid_resolution=256,
                        device='cuda',
                        num_views=50,
                        restore=True
                        )
        # 执行水密化
        output_verts, output_faces = watertight_processor(input_verts, input_faces)
    """
    def __init__(
            self,
            grid_resolution: int = 256,      # 体素网格分辨率（越高精度越好但速度越慢）
            device: str = 'cuda',            # 计算设备（cuda/cpu）
            num_views: int = 50,             # 可见性检查的相机视角数
            sample_size: float = 2.1,        # 采样空间尺寸（归一化后网格的包围盒大小）
            winding_number_thres: float = 0.5,  # 缠绕数阈值（判断内外表面）
            render_resolution: int = 1024,    # 渲染深度图的分辨率
            restore:bool=False, # 是否归一化到原始空间,默认[-1,1]
    ):
        self.grid_resolution = grid_resolution
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.num_views = num_views
        self.sample_size = sample_size
        self.winding_number_thres = winding_number_thres
        self.render_resolution = render_resolution

        # 初始化渲染器（内部精简版）
        self.renderer = self._init_renderer()

        # 记录归一化参数
        self.restore=restore
        self.center, self.scale=None,None

    def _init_renderer(self):
        """初始化nvdiffrast渲染器（仅用于深度图渲染）"""
        class MeshRenderer:
            def __init__(self, resolution, near, far, device):
                self.resolution = resolution
                self.near = near
                self.far = far
                self.device = device
                # 初始化渲染上下文
                if device.type == 'cuda':
                    self.ctx = dr.RasterizeCudaContext(device=device)
                else:
                    self.ctx = dr.RasterizeGLContext(device=device)
                # 渲染器预热（避免首次卡顿）
                self._warmup()

            def _warmup(self):
                """渲染器预热"""
                pos = torch.tensor([[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]],
                                   dtype=torch.float32, device=self.device)
                tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=self.device)
                dr.rasterize(self.ctx, pos, tri, resolution=[256, 256])

            def render_depth(self, vertices, faces, cam2world, mvp):
                """
                渲染深度图
                Args:
                    vertices: (N,3) 顶点数组
                    faces: (M,3) 面片数组
                    cam2world: (V,4,4) 相机到世界矩阵
                    mvp: (V,4,4) 模型视图投影矩阵
                Returns:
                    depths: (V, H, W, 1) 深度图
                """
                # 转换为tensor
                v_pos = torch.tensor(vertices, dtype=torch.float32, device=self.device)
                t_idx = torch.tensor(faces, dtype=torch.int32, device=self.device)

                # 齐次坐标变换
                verts_homo = torch.cat([v_pos, torch.ones([v_pos.shape[0], 1], device=self.device)], dim=-1)
                v_pos_clip = torch.matmul(verts_homo, mvp.permute(0, 2, 1))

                # 光栅化
                rast, _ = dr.rasterize(self.ctx, v_pos_clip.float(), t_idx.int(), self.resolution)
                mask = rast[..., 3:] > 0

                # 计算相机空间深度
                v_pos_cam = verts_homo @ cam2world.inverse().transpose(-1, -2)
                v_depth = v_pos_cam[..., 2:3] * -1  # 取负使深度为正
                gb_depth, _ = dr.interpolate(v_depth.contiguous(), rast, t_idx.int())
                gb_depth[~mask] = self.far  # 非渲染区域设为远平面

                return gb_depth

        return MeshRenderer(
            resolution=(self.render_resolution, self.render_resolution),
            near=0.1, far=10.0, device=self.device
        )

    @staticmethod
    def _sample_sphere_poses(num_views, radius=4.0):
        """采样球面相机位姿（生成多视角相机位置）"""
        phi = (np.sqrt(5) - 1.0) / 2.0  # 黄金比例采样
        poses = []
        for n in range(1, num_views + 1):
            y = (2.0 * n - 1) / num_views - 1.0
            x = np.cos(2 * np.pi * n * phi) * np.sqrt(1 - y * y)
            z = np.sin(2 * np.pi * n * phi) * np.sqrt(1 - y * y)
            poses.append((x * radius, y * radius, z * radius))
        return np.array(poses)

    def _visibility_check(self, points, depths, cam2world, mvp):
        """可见性检查：判断3D点是否在网格外表面可见"""
        dist = torch.ones(points.shape[0], device=self.device)
        mask = torch.zeros(points.shape[0], dtype=torch.bool, device=self.device)

        # 齐次坐标转换
        points_homo = torch.cat([points, torch.ones([points.shape[0], 1], device=self.device)], dim=-1)

        for i in range(len(cam2world)):
            # 裁剪空间坐标
            points_clip = points_homo @ mvp[i].permute(1, 0)
            # 有效视锥内的点（排除超出屏幕范围的点）
            valid = (torch.abs(points_clip[..., 0]) < 0.999) & (torch.abs(points_clip[..., 1]) < 0.999)
            if not valid.any():
                continue

            # 相机空间深度
            v_pos_cam = points_homo @ cam2world[i].inverse().transpose(-1, -2)
            v_depth = v_pos_cam[..., 2:3] * -1

            # 采样深度图获取对应位置的表面深度
            sample_z = grid_sample(
                depths[i].view(1, 1, self.render_resolution, self.render_resolution).float(),
                points_clip[valid, :2].reshape(1, 1, -1, 2),
                align_corners=True, mode='bilinear'
            ).reshape(-1)

            # 可见性判断：点深度 < 渲染深度 → 可见（在网格外部）
            visible = v_depth[valid].squeeze() < sample_z
            mask[torch.where(valid)[0][visible]] = True
            # 更新点到表面的距离
            dist[valid] = torch.minimum(dist[valid], torch.abs(sample_z - v_depth[valid].squeeze()))

        return mask, dist


    def _normalize_vertices(self,vertices):
        """网格顶点归一化到[-1, 1]范围"""
        bb_min = vertices.min(axis=0)
        bb_max = vertices.max(axis=0)
        self.center = (bb_min + bb_max) / 2.0
        self.scale = 2.0 / (bb_max - bb_min).max()  # 最大轴缩放到2（对应[-1,1]）
        return (vertices - self.center) * self.scale

    def __call__(self, vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        执行网格水密化处理（核心方法）
        Args:
            vertices: (N, 3) 输入网格顶点数组（numpy.float32）
            faces: (M, 3) 输入网格面片数组（numpy.int32）
        Returns:
            tuple: (watertight_vertices, watertight_faces)
                - watertight_vertices: (N', 3) 水密化后的顶点数组
                - watertight_faces: (M', 3) 水密化后的面片数组
        """
        # 1. 输入校验与归一化
        vertices =np.asarray(vertices,dtype=np.float32)
        faces=np.asarray(faces,dtype=np.int32)
        vertices = self._normalize_vertices(vertices)


        # 2. 生成相机位姿和投影矩阵
        cam_poses = self._sample_sphere_poses(self.num_views)
        cam2world, mvp = [], []
        for pos in cam_poses:
            # 计算外参矩阵（相机到世界）
            backward = (np.array([0, 0, 0]) - pos) / np.linalg.norm(pos)
            right = np.cross(backward, [0, 1, 0])
            right = right / np.linalg.norm(right) if np.linalg.norm(right) > 1e-6 else np.array([1, 0, 0])
            up = np.cross(right, backward)
            R = np.stack([right, up, -backward], axis=0)
            t = -R @ pos
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = t
            cam2world.append(np.linalg.inv(extrinsic))

            # 正交投影矩阵
            proj = np.zeros((4, 4))
            proj[0, 0] = 1.0
            proj[1, 1] = -1.0
            proj[2, 2] = -2 / (10.0 - 0.1)
            proj[2, 3] = -(10.0 + 0.1) / (10.0 - 0.1)
            proj[3, 3] = 1.0
            mvp.append(proj @ extrinsic)

        # 转换为tensor
        cam2world = torch.tensor(np.array(cam2world), dtype=torch.float32, device=self.device)
        mvp = torch.tensor(np.array(mvp), dtype=torch.float32, device=self.device)

        # 3. 渲染深度图
        depths = self.renderer.render_depth(vertices, faces, cam2world, mvp)

        # 4. 生成体素网格点（用于SDF采样）
        grid = np.meshgrid(
            np.arange(self.grid_resolution, dtype=np.float32),
            np.arange(self.grid_resolution, dtype=np.float32),
            np.arange(self.grid_resolution, dtype=np.float32),
            indexing='ij'
        )
        grid_points = np.stack(grid, axis=-1).reshape(-1, 3)
        # 归一化体素点到采样空间
        grid_points = (grid_points + 0.5) / self.grid_resolution * self.sample_size - self.sample_size / 2.0
        grid_points = torch.tensor(grid_points, device=self.device)

        # 5. 可见性检查 + 缠绕数判断内外
        visibility, dist = self._visibility_check(grid_points, depths, cam2world, mvp)
        # 计算缠绕数（区分网格内部/外部）
        winding = igl.fast_winding_number(
            vertices,
            faces,
            grid_points.detach().cpu().numpy()
        )
        winding = torch.from_numpy(winding).to(self.device)
        # 结合缠绕数过滤：内部点标记为不可见
        visibility[visibility & (winding > self.winding_number_thres)] = False

        # 6. 近表面点距离精修（UDF转SDF）
        near_surface = dist < 1.0
        if near_surface.any():
            # 计算点到网格的最短距离
            sq_dist, _, _ = igl.point_mesh_squared_distance(
                grid_points[near_surface].detach().cpu().numpy(),
                vertices,
               faces,
            )
            dist[near_surface] = torch.sqrt(torch.from_numpy(sq_dist).to(dtype=torch.float32,device= self.device))
        # UDF转SDF：内部点距离取负
        dist[~visibility] *= -1

        # 7. Marching Cubes重建水密网格
        sdf = dist.view(self.grid_resolution, self.grid_resolution, self.grid_resolution).cpu().numpy()
        # 提取0等值面（带微小偏移避免数值问题）
        watertight_verts, watertight_faces = mcubes.marching_cubes(
            sdf, self.sample_size / self.grid_resolution
        )
        # 将每个三角面片的顶点顺序反转，确保法线朝外
        watertight_faces = watertight_faces[:, [0, 2, 1]]

        # 还原坐标到归一化空间
        watertight_verts = watertight_verts / self.grid_resolution * self.sample_size - self.sample_size / 2.0
        if self.restore:
            watertight_verts=watertight_verts/self.scale+self.center


        return watertight_verts.astype(np.float32), watertight_faces.astype(np.int32)


# 测试示例
if __name__ == "__main__":
    import vedo
    import trimesh
    os.environ["DISPLAY"]=":0"
    # 1. 初始化水密化处理器
    watertight_processor = remesh_watertight(
        grid_resolution=256,
        device='cuda',
        num_views=50,
    restore=True
    )

    # 2. 加载网格并提取vertices/faces
    input_mesh = trimesh.load("/home/up3d/项目/全冠生成/out.obj")
    input_verts = input_mesh.vertices
    input_faces = input_mesh.faces

    # 3. 执行水密化
    output_verts, output_faces = watertight_processor(input_verts, input_faces)
