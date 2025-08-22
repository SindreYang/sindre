import os

import matplotlib.pyplot as plt
import numpy as np

from typing import *

import torch


class MeshRendererByVTK:
    """
       使用VTK从3D网格模型生成多视角2D图像的渲染器

       支持从不同视角渲染3D网格，生成RGB和深度图像
    """

    def __init__(self, resolution: int = 224):
        """
        初始化渲染器

        Args:
            resolution: 输出图像的分辨率，默认为224×224
        """
        import vtk
        # 存储渲染参数
        self.resolution = resolution
        # 初始化VTK渲染组件
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(resolution, resolution)
        self.render_window.SetMultiSamples(0)  # 禁用抗锯齿
        self.render_window.OffScreenRenderingOn()

        # 用于捕获渲染结果的过滤器
        self.color_filter = vtk.vtkWindowToImageFilter()
        self.color_filter.SetInputBufferTypeToRGB()
        self.color_filter.SetInput(self.render_window)

        self.depth_filter = vtk.vtkWindowToImageFilter()
        self.depth_filter.SetInputBufferTypeToZBuffer()
        self.depth_filter.SetInput(self.render_window)
        self.depth_filter.SetScale(1,1)
        # 存储当前渲染的actor
        self.current_actor = None
    def set_mesh(self,
                 vertices: np.ndarray,
                 faces: np.ndarray,
                 colors:np.ndarray =None):
        """
        设置要渲染的网格

        Args:
            vertices: 网格顶点数组，形状为 (N, 3)
            faces: 网格面数组，形状为 (M, 3)
            colors: 顶点或面颜色数组，形状为 (N, 3) 或 (M, 3)
        """
        # 清理渲染器
        if self.current_actor is not None:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None
        import vedo
        # 归一化处理顶点,使网格完全包含在单位球体内
        center = np.mean(vertices, axis=0)
        centered_vertices = vertices - center
        max_dist = np.max(np.linalg.norm(centered_vertices, axis=1))
        self.scale_factor = 1.0 / max_dist
        vertices = centered_vertices * self.scale_factor

        # 创建VTK网格对象
        vd_mesh = vedo.Mesh([vertices, faces])
        vd_mesh.compute_normals()
        if colors is not  None:
            if len(colors)==vd_mesh.npoints:
                vd_mesh.pointcolors =colors
            else:
                vd_mesh.cellcolors=colors
        # 创建网格actor
        mapper = vtk.vtkPolyDataMapper()
        actor = vtk.vtkActor()
        mapper.SetInputData(vd_mesh.dataset)
        actor.SetMapper(mapper)
        # 设置材质属性
        # prop = actor.GetProperty()
        # prop.SetColor(0.8, 0.8, 0.8)  # 浅灰色
        # prop.SetAmbient(0.3)
        # prop.SetDiffuse(0.7)
        # prop.SetSpecular(0.2)
        # prop.SetSpecularPower(10)
        # 添加到渲染器
        self.renderer.AddActor(actor)
        self.current_actor = actor
        # 重置相机
        self.renderer.ResetCamera()



    def set_camera_method(self,
                            num_points: int,
                            method: str="",
                            radius: float = 4.0,
                            turns: int = 4) -> None:
        """
          设置相机位置采样方法

          Args:
              num_points: 采样点数
              method: 采样方法，"spiral"或"sphere"
              radius: 采样球半径
              turns: 螺旋方法的旋转圈数

          Returns:
              Tuple: (相机位置数组, 视图向上方向数组, 焦点数组)
        """
        import vedo
        if method == "spiral":
            # 使用螺旋方法生成采样点
            import math
            points = []
            c = 2.0 * float(turns)
            for i in range(num_points):
                angle = (i * math.pi) / num_points
                x = radius * math.sin(angle) * math.cos(c * angle)
                y = radius * math.sin(angle) * math.sin(c * angle)
                z = radius * math.cos(angle)
                points.append([x, y, z])

            sphere_points =  np.array(points)

            # 为螺旋点计算视图向上方向
            view_up_points = []
            for point in points:
                # 归一化点向量
                point_norm = point / np.linalg.norm(point)
                # 计算视图向上方向
                if abs(point_norm[2]) != 1:
                    view_up = np.array([0, 0, -1])
                elif point_norm[2] == 1:
                    view_up = np.array([1, 0, 0])
                elif point_norm[2] == -1:
                    view_up = np.array([-1, 0, 0])
                else:
                    view_up = np.array([0, 0, -1])

                view_up_points.append(view_up)
            view_up_points = np.array(view_up_points)
            focal_points = np.zeros((len(sphere_points), 3))
            return  sphere_points,view_up_points,focal_points
        else:
            # 使用vedo创建细分球面
            ico = vedo.IcoSphere(subdivisions=num_points, r=radius)
            sphere_points = np.array(ico.vertices)
            # 默认视图向上方向和焦点
            view_up_points = np.array([[0, 0, -1]] * len(self.sphere_points))
            focal_points = np.zeros((len(sphere_points), 3))
            return  sphere_points,view_up_points,focal_points

    def render_views(self,sphere_points,view_up_points,focal_points):
        """

        渲染所有视角的图像

        Args:
            sphere_points: 相机位置数组
            view_up_points: 视图向上方向数组
            focal_points: 焦点数组

        Returns:
            Tuple: (RGB图像列表, 深度彩色图像列表, 深度值列表)
        """

        from vtkmodules.util.numpy_support import vtk_to_numpy
        from sindre.utils3d.algorithm import depth2color
        # 设置背景色
        self.renderer.SetBackground(0,0,0)  # 黑色背景

        # 获取相机
        camera = self.renderer.GetActiveCamera()

        # 存储所有渲染结果
        rgb_images = []
        depth_images=[]
        depth_values = []
        print(f"开始渲染 {len(sphere_points)} 个视角...")
        for i, sphere_point in enumerate(sphere_points):
            # 设置相机位置
            camera.SetPosition(sphere_point[0], sphere_point[1], sphere_point[2])
            # 设置视图向上方向
            view_up = view_up_points[i]
            camera.SetViewUp(view_up[0], view_up[1], view_up[2])
            # 设置焦点
            camera.SetFocalPoint(focal_points[i][0], focal_points[i][1], focal_points[i][2])
            # 重置相机裁剪范围
            self.renderer.ResetCameraClippingRange()
            # 渲染RGB图像
            self.color_filter.Modified()
            self.color_filter.Update()
            rgb_image = self.color_filter.GetOutput()
            rgb_np = vtk_to_numpy(rgb_image.GetPointData().GetScalars())
            # 重塑RGB图像维度
            rgb_np = rgb_np.reshape((self.resolution,self.resolution,3))
            # 存储结果
            rgb_images.append(rgb_np)


            # 处理深度信息
            self.depth_filter.Modified()
            self.depth_filter.Update()
            depth_image = self.depth_filter.GetOutput()
            depth_np = vtk_to_numpy(depth_image.GetPointData().GetScalars())
            # 重塑深度图像维度
            depth_np = depth_np.reshape((self.resolution,self.resolution))
            # 深度值处理
            z_near, z_far = camera.GetClippingRange()
            # 将深度值从[0,1]转换为实际距离
            depth_np = 2.0 * z_far * z_near / (z_far + z_near - (z_far - z_near) * (2.0 * depth_np - 1.0))
            # 将超出范围的深度值设为0
            depth_np[depth_np > (z_far - 0.1)] = 0
            # 将深度转换为彩色图像（使用jet颜色映射）
            depth_color = depth2color(depth_np)
            depth_images.append(depth_color)
            depth_values.append(depth_np)
        print("渲染完成！")
        return rgb_images,depth_images,depth_values

    def __del__(self):
        """
        清理资源
        """
        if self.current_actor is not None:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None
        self.render_window.Finalize()










class MeshRendererByPyRender:
    """
    离线渲染器类，用于在没有显示设备的情况下渲染3D模型
    基于pyrender库实现，支持设置相机内参、模型姿态和光照
    """
    def __init__(self,  H, W,cam_K=None, zfar=100):
        '''
        初始化离线渲染器

        参数:
            cam_K: 相机内参矩阵 (3x3)，包含焦距和主点坐标
            H: 渲染图像的高度
            W: 渲染图像的宽度
            zfar: 相机远平面距离，默认100
        '''
        import pyrender
        if cam_K is None:
            # 从相机物理参数计算内参
            fx=fy=0.8* max(W, H) # 经验值
            self.K = np.array([
                [fx, 0, W*0.5],
                [0, fy, H*0.5],
                [0, 0, 1]
            ])
        else:
            self.K = cam_K  # 存储相机内参
        # OpenCV相机坐标系到OpenGL相机坐标系的转换矩阵
        # OpenCV相机坐标系: X右, Y下, Z前
        # OpenGL相机坐标系: X右, Y上, Z后
        # 该矩阵通过翻转Y轴和Z轴实现坐标系统的转换
        self.cvcam_in_glcam = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ])


        # 创建场景，设置环境光和背景色(黑色)
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])

        # 创建相机对象，使用给定的内参
        self.camera = pyrender.IntrinsicsCamera(
            fx=cam_K[0, 0],   # 水平焦距
            fy=cam_K[1, 1],   # 垂直焦距
            cx=cam_K[0, 2],   # 主点x坐标
            cy=cam_K[1, 2],   # 主点y坐标
            znear=0.001,      # 近平面距离
            zfar=zfar         # 远平面距离
        )

        # 将相机添加到场景，初始姿态为单位矩阵(世界原点)
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4), name='cam')
        self.mesh_nodes = []  # 存储场景中的模型节点

        self.H = H  # 渲染高度
        self.W = W  # 渲染宽度
        # 创建离线渲染器实例
        self.r = pyrender.OffscreenRenderer(self.W, self.H)



    def set_cam_pose(self, cam_pose):
        """
        设置相机在世界坐标系中的姿态

        参数:
            cam_pose: 4x4变换矩阵，表示相机的位姿
        """
        self.cam_node.matrix = cam_pose


    def add_mesh(self, mesh):
        """
        向场景中添加3D模型

        参数:
            mesh: trimesh格式的3D模型
        """
        import pyrender
        # 将trimesh模型转换为pyrender格式
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        # 将模型添加到场景，初始姿态为单位矩阵
        mesh_node = self.scene.add(pyrender_mesh, pose=np.eye(4), name='ob')
        self.mesh_nodes.append(mesh_node)


    def add_point_light(self, intensity=3):
        """
        向场景中添加方向光(与相机位置相同)

        参数:
            intensity: 光强，默认3
        """
        import pyrender
        light = pyrender.DirectionalLight(
            color=[1.0, 1.0, 1.0],  # 白色光
            intensity=intensity
        )
        # 光源位置与相机一致
        self.scene.add(light, pose=np.eye(4))


    def clear_mesh_nodes(self):
        """移除场景中所有已添加的模型节点"""
        for node in self.mesh_nodes:
            self.scene.remove_node(node)


    def render(self, mesh=None, ob_in_cvcam=None, get_normal=False):
        """
        渲染场景，返回彩色图和深度图

        参数:
            mesh: 可选，trimesh格式的模型，如果提供则临时添加到场景
            ob_in_cvcam: 4x4变换矩阵，表示模型在OpenCV相机坐标系下的位姿
            get_normal: 是否返回法向量图(当前未实现)

        返回:
            color: 渲染的彩色图像 (H, W, 3)，uint8格式
            depth: 渲染的深度图 (H, W)，float格式，值为距离
        """
        if mesh is not None :
            import pyrender
            # 复制模型以避免修改原始数据
            mesh_copy = mesh.copy()
            # 将模型从OpenCV相机坐标系转换到OpenGL坐标系
            # 先应用模型在OpenCV相机下的位姿，再转换坐标系
            transform = self.cvcam_in_glcam @ ob_in_cvcam
            mesh_copy.apply_transform(transform)
            # 转换为pyrender格式并添加到场景
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_copy, smooth=False)
            mesh_node = self.scene.add(pyrender_mesh, pose=np.eye(4), name='ob')

        # 执行渲染
        color, depth = self.r.render(self.scene)
        if mesh is not None:
            self.scene.remove_node(mesh_node)
        return color, depth




    def render_multi_views(self, mesh, poses, save_dir=None):
        """
        从多个视角渲染模型

        参数:
            mesh: trimesh格式的模型
            poses: 位姿列表，每个元素是4x4矩阵，表示模型在不同视角下的位姿
            save_dir: 可选，保存渲染结果的目录

        返回:
            渲染结果列表，每个元素是(color, depth)元组
        """
        results = []
        # 提前将模型添加到场景以提高效率
        self.add_mesh(mesh)

        for i, pose in enumerate(poses):
            # 设置模型姿态
            # 注意：这里需要先将位姿转换到OpenGL坐标系
            transform = self.cvcam_in_glcam @ pose
            self.mesh_nodes[-1].matrix = transform

            # 渲染
            color, depth = self.r.render(self.scene)
            results.append((color, depth))

            # 如果指定了保存目录，则保存渲染结果
            if save_dir is not None:
                from PIL import  Image
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(color).save(f"{save_dir}/color_{i:04d}.png")
                np.save(f"{save_dir}/depth_{i:04d}.npy", depth)

        # 清理场景
        self.clear_mesh_nodes()
        return results




class NvdiffrastRenderer:
    """nvdiffrast渲染器类，支持纹理和顶点颜色渲染。

    属性:
        glctx: nvdiffrast渲染上下文
        mesh_tensors: 网格张量数据
    """

    def __init__(self, mesh, device='cuda', context='cuda', max_tex_size=None):
        """初始化渲染器。

        Args:
            mesh: trimesh网格对象
            device: 计算设备 ('cuda' 或 'cpu')
            context: 渲染上下文类型 ('cuda' 或 'gl')
            max_tex_size: 纹理最大尺寸
        """

        import torch
        import numpy as np
        import trimesh
        import cv2
        import nvdiffrast.torch as dr
        import torch.nn.functional as F
        self.device = device
        self.mesh_tensors = self._make_mesh_tensors(mesh, device, max_tex_size)

        # 初始化渲染上下文
        if context == 'gl':
            self.glctx = dr.RasterizeGLContext()
        elif context == 'cuda':
            self.glctx = dr.RasterizeCudaContext()
        else:
            raise NotImplementedError(f"不支持的上下文类型: {context}")

    def _make_mesh_tensors(self, mesh, device, max_tex_size):
        """创建网格张量数据。

        Args:
            mesh: trimesh网格对象
            device: 计算设备
            max_tex_size: 纹理最大尺寸

        Returns:
            Dict: 包含网格张量的字典
        """
        import torch
        import cv2
        import trimesh
        mesh_tensors = {}

        # 处理纹理和UV坐标
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            img = np.array(mesh.visual.material.image.convert('RGB'))[..., :3]

            # 调整纹理尺寸
            if max_tex_size is not None:
                max_size = max(img.shape[0], img.shape[1])
                if max_size > max_tex_size:
                    scale = max_tex_size / max_size
                    img = cv2.resize(img, None, fx=scale, fy=scale)

            mesh_tensors['tex'] = torch.as_tensor(img, device=device, dtype=torch.float)[None] / 255.0
            mesh_tensors['uv_idx'] = torch.as_tensor(mesh.faces, device=device, dtype=torch.int)

            uv = torch.as_tensor(mesh.visual.uv, device=device, dtype=torch.float)
            uv[:, 1] = 1 - uv[:, 1]  # 翻转V坐标
            mesh_tensors['uv'] = uv
        else:
            # 处理顶点颜色
            if mesh.visual.vertex_colors is None:
                print("WARN: 网格没有顶点颜色，使用默认颜色")
                mesh.visual.vertex_colors = np.tile([128, 128, 128], (len(mesh.vertices), 1))

            mesh_tensors['vertex_color'] = torch.as_tensor(
                mesh.visual.vertex_colors[..., :3], device=device, dtype=torch.float) / 255.0

        # 添加基础网格数据
        mesh_tensors.update({
            'pos': torch.tensor(mesh.vertices, device=device, dtype=torch.float),
            'faces': torch.tensor(mesh.faces, device=device, dtype=torch.int),
            'vnormals': torch.tensor(mesh.vertex_normals, device=device, dtype=torch.float),
        })

        return mesh_tensors

    def _projection_matrix_from_intrinsics(self, K, height, width, znear, zfar, window_coords='y_down'):
        """从内参矩阵计算投影矩阵。

        Args:
            K: 3x3相机内参矩阵
            height: 图像高度
            width: 图像宽度
            znear: 近裁剪平面
            zfar: 远裁剪平面
            window_coords: 窗口坐标类型 ('y_up' 或 'y_down')

        Returns:
            ndarray: 4x4投影矩阵
        """
        depth = float(zfar - znear)
        q = -(zfar + znear) / depth
        qn = -2 * (zfar * znear) / depth

        if window_coords == 'y_up':
            proj = np.array([
                [2 * K[0, 0] / width, -2 * K[0, 1] / width, (-2 * K[0, 2] + width) / width, 0],
                [0, -2 * K[1, 1] / height, (-2 * K[1, 2] + height) / height, 0],
                [0, 0, q, qn],
                [0, 0, -1, 0]
            ])
        elif window_coords == 'y_down':
            proj = np.array([
                [2 * K[0, 0] / width, -2 * K[0, 1] / width, (-2 * K[0, 2] + width) / width, 0],
                [0, 2 * K[1, 1] / height, (2 * K[1, 2] - height) / height, 0],
                [0, 0, q, qn],
                [0, 0, -1, 0]
            ])
        else:
            raise NotImplementedError(f"不支持的窗口坐标类型: {window_coords}")

        return proj

    def _transform_pts(self, pts, tf):
        """变换点坐标。

        Args:
            pts: 点坐标张量 (..., N_pts, 3)
            tf: 变换矩阵 (..., 4, 4)

        Returns:
            Tensor: 变换后的点坐标
        """
        if len(tf.shape) >= 3 and tf.shape[-3] != pts.shape[-2]:
            tf = tf[..., None, :, :]
        return (tf[..., :-1, :-1] @ pts[..., None] + tf[..., :-1, -1:])[..., 0]

    def _to_homo_torch(self, pts):
        """转换为齐次坐标。

        Args:
            pts: 点坐标张量 (..., N, 3/2)

        Returns:
            Tensor: 齐次坐标
        """
        import torch
        ones = torch.ones((*pts.shape[:-1], 1), dtype=torch.float, device=pts.device)
        return torch.cat((pts, ones), dim=-1)

    def _transform_dirs(self, dirs, tf):
        """变换方向向量。

        Args:
            dirs: 方向向量 (..., 3)
            tf: 变换矩阵 (..., 4, 4)

        Returns:
            Tensor: 变换后的方向向量
        """
        if len(tf.shape) >= 3 and tf.shape[-3] != dirs.shape[-2]:
            tf = tf[..., None, :, :]
        return (tf[..., :3, :3] @ dirs[..., None])[..., 0]

    def render(self, K, H, W, ob_in_cams, get_normal=False, projection_mat=None,
               bbox2d=None, output_size=None, use_light=False, light_color=None,
               light_dir=np.array([0, 0, 1]), light_pos=np.array([0, 0, 0]),
               w_ambient=0.8, w_diffuse=0.5):
        """执行渲染。

        Args:
            K: 3x3相机内参矩阵
            H: 图像高度
            W: 图像宽度
            ob_in_cams: 物体到相机的变换矩阵 (N, 4, 4)
            get_normal: 是否获取法线图
            projection_mat: 投影矩阵 (可选)
            bbox2d: 渲染区域边界框
            output_size: 输出尺寸
            use_light: 是否使用光照
            light_color: 光源颜色
            light_dir: 光源方向
            light_pos: 光源位置
            w_ambient: 环境光权重
            w_diffuse: 漫反射光权重


        Returns:
            Tuple: (颜色图像, 深度图, 法线图)
        """
        import torch
        import nvdiffrast.torch as dr
        import torch.nn.functional as F
        # 准备变换矩阵
        glcam_in_cvcam = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ], device=self.device, dtype=torch.float)[None]

        ob_in_glcams = glcam_in_cvcam @ ob_in_cams

        # 计算投影矩阵
        if projection_mat is None:
            projection_mat = self._projection_matrix_from_intrinsics(
                K, H, W, 0.001, 100)
        projection_mat = torch.as_tensor(
            projection_mat.reshape(-1, 4, 4), device=self.device, dtype=torch.float)

        mtx = projection_mat @ ob_in_glcams  # 组合变换矩阵

        if output_size is None:
            output_size = np.array([H, W])

        # 变换顶点到相机空间
        pos = self.mesh_tensors['pos']
        pts_cam = self._transform_pts(pos, ob_in_cams)
        pos_homo = self._to_homo_torch(pos)
        pos_clip = (mtx[:, None] @ pos_homo[None, ..., None])[..., 0]

        # 处理边界框
        if bbox2d is not None:
            l, t, r, b = bbox2d[:, 0], H - bbox2d[:, 1], bbox2d[:, 2], H - bbox2d[:, 3]
            tf = torch.eye(4, dtype=torch.float, device=self.device)
            tf = tf.reshape(1, 4, 4).expand(len(ob_in_cams), 4, 4).contiguous()
            tf[:, 0, 0] = W / (r - l)
            tf[:, 1, 1] = H / (t - b)
            tf[:, 3, 0] = (W - r - l) / (r - l)
            tf[:, 3, 1] = (H - t - b) / (t - b)
            pos_clip = pos_clip @ tf

        # 光栅化
        rast_out, _ = dr.rasterize(
            self.glctx, pos_clip, self.mesh_tensors['faces'], resolution=output_size)

        # 插值计算属性
        xyz_map, _ = dr.interpolate(pts_cam, rast_out, self.mesh_tensors['faces'])
        depth = xyz_map[..., 2]

        # 计算颜色
        if 'tex' in self.mesh_tensors:
            texc, _ = dr.interpolate(
                self.mesh_tensors['uv'], rast_out, self.mesh_tensors['uv_idx'])
            color = dr.texture(self.mesh_tensors['tex'], texc, filter_mode='linear')
        else:
            color, _ = dr.interpolate(
                self.mesh_tensors['vertex_color'], rast_out, self.mesh_tensors['faces'])

        # 计算法线
        if use_light:
            get_normal = True

        if get_normal:
            vnormals_cam = self._transform_dirs(
                self.mesh_tensors['vnormals'], ob_in_cams)
            normal_map, _ = dr.interpolate(vnormals_cam, rast_out, self.mesh_tensors['faces'])
            normal_map = F.normalize(normal_map, dim=-1)
            normal_map = torch.flip(normal_map, dims=[1])
        else:
            normal_map = None

        # 应用光照
        if use_light:
            if light_dir is not None:
                light_dir_neg = -torch.as_tensor(light_dir, dtype=torch.float, device=self.device)
            else:
                light_dir_neg = torch.as_tensor(light_pos, dtype=torch.float, device=self.device)
                light_dir_neg = light_dir_neg.reshape(1, 1, 3) - pts_cam

            diffuse_intensity = (F.normalize(vnormals_cam, dim=-1) *
                                 F.normalize(light_dir_neg, dim=-1)).sum(dim=-1).clip(0, 1)[..., None]
            diffuse_intensity_map, _ = dr.interpolate(diffuse_intensity, rast_out, self.mesh_tensors['faces'])

            if light_color is None:
                light_color_tensor = color
            else:
                light_color_tensor = torch.as_tensor(light_color, device=self.device, dtype=torch.float)

            color = color * w_ambient + diffuse_intensity_map * light_color_tensor * w_diffuse

        # 后处理
        color = color.clip(0, 1)
        color = color * torch.clamp(rast_out[..., -1:], 0, 1)
        color = torch.flip(color, dims=[1])
        depth = torch.flip(depth, dims=[1])
        xyz_map = torch.flip(xyz_map, dims=[1])

        return color, depth, normal_map,xyz_map

def depth2xyzmap(depth, K, uvs=None):
    invalid_mask = (depth<0.001)
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    xyz_map[invalid_mask] = 0
    return xyz_map


def depth2xyzmap_batch(depths, Ks, zfar):
    '''
    depths: torch tensor (B,H,W)
    Ks: torch tensor (B,3,3)
    '''
    import torch
    bs = depths.shape[0]
    invalid_mask = (depths<0.001) | (depths>zfar)
    H,W = depths.shape[-2:]
    vs,us = torch.meshgrid(torch.arange(0,H),torch.arange(0,W), indexing='ij')
    vs = vs.reshape(-1).float().cuda()[None].expand(bs,-1)
    us = us.reshape(-1).float().cuda()[None].expand(bs,-1)
    zs = depths.reshape(bs,-1)
    Ks = Ks[:,None].expand(bs,zs.shape[-1],3,3)
    xs = (us-Ks[...,0,2])*zs/Ks[...,0,0]  #(B,N)
    ys = (vs-Ks[...,1,2])*zs/Ks[...,1,1]
    pts = torch.stack([xs,ys,zs], dim=-1)  #(B,N,3)
    xyz_maps = pts.reshape(bs,H,W,3)
    xyz_maps[invalid_mask] = 0
    return xyz_maps




class PcdRendererByNumpy:
    """
    基于numpy实现点云深度图渲染
    """
    def __init__(self, mesh_points, img_size=256):
        """
        :param mesh_points: 点云
        :param img_size: 深度图分辨率（默认256x256）
        """
        self.mesh_points = mesh_points  # 网格顶点
        self.n_points = len(self.mesh_points)  # 顶点数量
        self.img_size = img_size

        # 存储投影信息，用于深度图到点云的转换
        self.last_proj_info = None

    def _create_local_coordinate_system(self, plane_normal):
        """创建投影平面的局部坐标系"""
        # 归一化法向量
        plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)
        # 选择一个与法向量不平行的参考向量
        if abs(plane_normal[0]) < 0.9:
            ref_vector = np.array([1, 0, 0])
        else:
            ref_vector = np.array([0, 1, 0])

        # 计算第一个轴（与法向量垂直）
        axis1 = np.cross(plane_normal, ref_vector)
        axis1 = axis1 / np.linalg.norm(axis1)

        # 计算第二个轴（与法向量和第一个轴都垂直）
        axis2 = np.cross(plane_normal, axis1)
        axis2 = axis2 / np.linalg.norm(axis2)

        return axis1, axis2, plane_normal

    def _project_points_to_plane(self, points, plane_pos, plane_normal):
        """将点投影到平面并返回局部2D坐标"""
        axis1, axis2, _ = self._create_local_coordinate_system(plane_normal)
        # 计算每个点在平面上的投影
        projected_points = []
        local_coords = []
        for point in points:
            # 计算点到平面的向量
            vec_to_plane = point - plane_pos
            # 计算点在平面上的投影
            dot_product = np.dot(vec_to_plane, plane_normal)
            projection = point - dot_product * plane_normal
            projected_points.append(projection)
            # 计算点在局部坐标系中的2D坐标
            x = np.dot(projection - plane_pos, axis1)
            y = np.dot(projection - plane_pos, axis2)
            local_coords.append([x, y])

        return np.array(projected_points), np.array(local_coords)

    def mesh2depth(self, h_threshold=None, proj_plane_pos=(0, 0, 0), proj_plane_normal=(0, 0, 1)):
        """
        Vedo核心功能：网格 → 深度图（基于投影距离）
        :param h_threshold: 距离阈值（默认为None ，自动计算）
        :param proj_plane_pos: 投影平面中心点（默认(0,0,0)）
        :param proj_plane_normal: 投影平面法向量（默认Z轴(0,0,1)）
        :return: 深度图（numpy.ndarray, shape=(img_size, img_size)）
        """
        proj_plane_pos = np.array(proj_plane_pos, dtype=np.float32)
        proj_plane_normal = np.array(proj_plane_normal, dtype=np.float32)
        # 1. 将点投影到平面并获取局部2D坐标
        projected_points, local_coords = self._project_points_to_plane(
            self.mesh_points, proj_plane_pos, proj_plane_normal)
        # 2. 计算每个顶点到平面的距离（有符号距离）
        distances = []
        plane_normal = np.array(proj_plane_normal) / np.linalg.norm(proj_plane_normal)
        for point in self.mesh_points:
            vec_to_plane = point - np.array(proj_plane_pos)
            distance = np.dot(vec_to_plane, plane_normal)
            distances.append(distance)
        distances = np.array(distances)
        # 自动计算阈值
        if h_threshold is None:
            # 计算距离的统计信息
            mean_dist = np.mean(np.abs(distances))
            std_dist = np.std(distances)
            max_dist = np.max(np.abs(distances))
            # 基于统计信息自动确定阈值
            # 使用平均值加上两倍标准差，但不超过最大距离的90%
            h_threshold = min(mean_dist + 2 * std_dist, max_dist * 0.9)
            # 确保阈值至少为最大距离的10%，避免过小
            h_threshold = max(h_threshold, max_dist * 0.1)
            # 打印调试信息
            print(f"自动计算深度阈值: {h_threshold:.3f}")
        # 3. 归一化局部坐标到图像尺寸
        min_x, min_y = np.min(local_coords, axis=0)
        max_x, max_y = np.max(local_coords, axis=0)
        # 避免除
        range_x = max_x - min_x
        range_y = max_y - min_y
        if range_x == 0: range_x = 1
        if range_y == 0: range_y = 1
        # 归一化到图像尺寸
        normalized_coords = np.zeros_like(local_coords)
        normalized_coords[:, 0] = (local_coords[:, 0] - min_x) / range_x * (self.img_size - 1)
        normalized_coords[:, 1] = (local_coords[:, 1] - min_y) / range_y * (self.img_size - 1)
        normalized_coords = normalized_coords.astype(int)

        # 4. 生成深度图
        depth_image = np.zeros((self.img_size, self.img_size))

        # 对于每个投影点，在深度图中记录距离
        for i in range(self.n_points):
            x, y = normalized_coords[i]
            if 0 <= x < self.img_size and 0 <= y < self.img_size:
                d = distances[i]
                # 应用阈值
                if abs(d) > h_threshold:
                    depth_image[y, x] = 0  # 注意：图像坐标是(y,x)
                else:
                    # 归一化距离到0-255范围
                    normalized_d = int(255 * (1 - abs(d) / h_threshold))
                    depth_image[y, x] = normalized_d

        # 存储投影信息，用于深度图到点云的转换
        self.last_proj_info = {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'proj_plane_pos': proj_plane_pos,
            'proj_plane_normal': proj_plane_normal,
            'h_threshold': h_threshold,
            'axis1': self._create_local_coordinate_system(proj_plane_normal)[0],
            'axis2': self._create_local_coordinate_system(proj_plane_normal)[1]
        }

        return depth_image

    def depth2pointcloud(self, depth_img=None):
        """
        Vedo配套功能：深度图 → 点云
        :param depth_img: 输入深度图（默认用mesh2depth生成的结果）
        :return: 点云坐标（numpy数组）
        """
        # 获取投影信息
        min_x = self.last_proj_info['min_x']
        max_x = self.last_proj_info['max_x']
        min_y = self.last_proj_info['min_y']
        max_y = self.last_proj_info['max_y']
        proj_plane_pos = self.last_proj_info['proj_plane_pos']
        proj_plane_normal = self.last_proj_info['proj_plane_normal']
        h_threshold = self.last_proj_info['h_threshold']
        axis1 = self.last_proj_info['axis1']
        axis2 = self.last_proj_info['axis2']
        points = []
        height, width = depth_img.shape
        # 计算坐标范围
        range_x = max_x - min_x
        range_y = max_y - min_y
        if range_x == 0: range_x = 1
        if range_y == 0: range_y = 1
        for y in range(height):
            for x in range(width):
                d = depth_img[y, x]
                if d > 0:  # 只处理有效点
                    # 将图像坐标转换回局部2D坐标
                    local_x = min_x + (x / (width - 1)) * range_x
                    local_y = min_y + (y / (height - 1)) * range_y
                    # 计算点在平面上的位置
                    plane_point = proj_plane_pos + local_x * axis1 + local_y * axis2
                    # 计算深度值（根据深度图中的值）
                    depth_value = (1 - d / 255) * h_threshold
                    # 沿法线方向移动点
                    point_3d = plane_point + depth_value * proj_plane_normal
                    points.append(point_3d)

        return np.array(points)


class MeshRendererByOpen3d:
    # ------------------------------
    # 基于 Open3D 的网格深度渲染类
    # ------------------------------
    def __init__(self, o3d_mesh_t, img_size=256):
        """
        初始化Open3D渲染器
        :param o3d_mesh_t: open3d的mesh
        :param img_size: 深度图分辨率（默认256x256）
        """
        import open3d as o3d
        self.o3d_mesh_tensor = o3d_mesh_t
        #o3d.visualization.draw([self.o3d_mesh_tensor])
        # 核心参数
        self.img_size = img_size


    def mesh2depth(self,  fov_deg=60, eye=(0, 0, 0), center=(0, 0, 1), up=(0, 0, 1)):
        """
        Open3D核心功能：网格 → 深度图（基于光线投射）
        :param fov_deg: 针孔相机FOV（默认60°）---视野范围
        :param eye: 相机位置（默认(0,0,0)）---从哪里看
        :param center: 相机朝向中心点（默认(0,0,0)）--向量为 (center - eye），决定了 “看哪里”。
        :param up: 相机上方向（默认(0,0,-1)）---确定相机的姿态,若调整为 (0,0,1)，则相机可能 “侧躺” 观察
        :return: 深度图（numpy.ndarray, shape=(img_size, img_size)）
        """
        import open3d as o3d
        # 初始化光线投射场景
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(self.o3d_mesh_tensor)

        # 生成针孔相机光线,透射投影
        rays = scene.create_rays_pinhole(
            fov_deg=fov_deg,
            center=center,
            eye=eye,
            up=up,
            width_px=self.img_size,
            height_px=self.img_size
        )
        ans = scene.cast_rays(rays)
        depth_value = ans['t_hit'].numpy()
        assert len(np.unique(depth_value))!=1, "当前相机参数无法捕获图像"
        inf_mask = np.isinf(depth_value)
        depth_value_no_inf = np.where(inf_mask, depth_value[~inf_mask].max(), depth_value)
        depth_image = depth2color(depth_value_no_inf,[1,1,1])

        # 法线图
        normal_image = np.abs(ans['primitive_normals'].numpy())*255
        return depth_image,depth_value,normal_image,rays


    def depth2pointcloud(self, depth_img,rays):
        """
        Open3D配套功能：深度图 → 点云
        :param depth_img: 输入深度图（用mesh2depth生成的结果）
        """
        import open3d as o3d

        hit=o3d.core.Tensor(depth_img)
        bool_hit=hit.isfinite()
        points = rays[bool_hit][:,:3] + rays[bool_hit][:,3:]*hit[bool_hit].reshape((-1,1))
        return points.numpy()









class MeshRendererByPytorch3d:
    """
    封装PyTorch3D功能的牙齿网格渲染器，直接接受Meshes对象作为输入
    并生成对应的分割标签，方便用于训练2D分割模型

    注意: 输入的Meshes对象必须包含有效的纹理信息，否则渲染结果可能不正确
    """

    def __init__(self,  device="cpu",image_size=256):
        """
        初始化渲染器参数

        参数:
            image_size: 渲染图像的尺寸
            device: 运行设备 (cuda或cpu)
        """
        # 设置设备
        self.device = device
        # 渲染参数
        self.image_size = image_size
        # 初始化PyTorch3D组件
        self._init_renderer()

    def get_spherical_camera(self, center:np.array,radius:float,num_points:int=8, distance_factor=1):
        """在球面上均匀采样指定数量的点,并作为相机点"""
        from pytorch3d.renderer import look_at_rotation
        center=torch.from_numpy(center).to(dtype=torch.float32, device=self.device)
        # 生成均匀分布的球面坐标（Fibonacci采样法）
        indices = torch.arange(0, num_points, dtype=torch.float32, device=self.device)
        phi = torch.acos(1 - 2 * indices / num_points)  # 俯仰角 [0, pi]
        theta = torch.pi * (1 + 5**0.5) * indices       # 方位角 [0, 2pi]

        # 转换为笛卡尔坐标 (x, y, z)
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        sample_dirs = torch.stack((x, y, z), dim=1)
        camera_positions =  center+sample_dirs  * radius*distance_factor  # (num_views, 3)
        # 生成视角
        R = look_at_rotation(camera_positions,
                             at=center.expand(num_points, -1),
                             device=self.device,
                             up=((1, 0, 0),
                             ))
        # 计算平移向量 T = -R @ camera_position
        T = -torch.bmm(R.transpose(1, 2), camera_positions[:, :, None])[:, :, 0] # (num_views, 3)
        return R,T
    def _init_renderer(self):
        """初始化PyTorch3D渲染器组件"""
        from pytorch3d.renderer import (
            FoVPerspectiveCameras, look_at_rotation,
            RasterizationSettings, MeshRenderer, MeshRasterizer,
            HardPhongShader, AmbientLights,DirectionalLights
        )


        # 相机
        self.cameras = FoVPerspectiveCameras(device=self.device)

        # 光栅化设置
        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
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

    def render(self, mesh,  R ,T):
        """
        渲染网格为2D图像

        参数:
            mesh: PyTorch3D的Meshes对象（请确保已加入纹理信息）
            R: 旋转矩阵 (B, 3, 3)，如果为None则使用相机位置计算
            T: 平移矩阵 (B, 3)，如果为None则使用相机位置计算

        返回:
            image: 渲染的图像张量 (B, 4, H, W) - RGBA格式
            pix_to_face: 每个像素对应的面索引 (B, H, W, 1)
        """
        # 确保网格在正确的设备上
        if mesh.device != self.device:
            mesh = mesh.to(self.device)
        # 复制网格，以匹配视图维度
        mesh = mesh.extend(R.shape[0])
        # 渲染图像
        image = self.renderer(meshes_world=mesh, R=R, T=T)
        # 获取光栅化信息(像素-面映射)
        frags = self.rasterizer(meshes_world=mesh, R=R, T=T)
        return image, frags.pix_to_face

    def faceslabel2imagelabel(self, pix_to_face, face_labels):
        """
        从像素-面映射和面部标签生成像素级标签

        参数:
            pix_to_face: 每个像素对应的面索引 (B, H, W, 1)
            face_labels: 每个面的标签 (张量) (B,N_faces, 1)

        返回:
            pixel_labels: 像素级标签张量 (B, 1, H, W)
        """
        # 将面标签转换为张量并确保在正确设备上
        import torch
        # 验证维度
        B, H, W, _ = pix_to_face.shape
        B_f, N_faces, _ = face_labels.shape
        if B != B_f:
            raise ValueError(f"批处理大小不匹配: pix_to_face={B}, face_labels={B_f}")
        # 映射面标签到像素
        pixel_labels = torch.take(face_labels, pix_to_face)
        # 过滤背景像素(面索引为-1的像素)
        pixel_labels = pixel_labels * (pix_to_face >= 0)
        return pixel_labels



if __name__ == '__main__':
    from sindre.utils3d import *
    import os
    import vedo
    os.environ["DISPLAY"] = ":0"
    mesh = SindreMesh("/home/up3d/sindre/discard/312580.sm")
    print(np.unique(mesh.vertex_labels))
    mesh.get_curvature_igl()
    #mesh.show()

    # 测试vtk
    # mesh_render =  MeshRendererByVTK(resolution=225)
    # mesh_render.set_mesh(mesh.vertices, mesh.faces)
    # sphere_points,view_up_points,focal_points=mesh_render.set_camera_method(method = "spiral",num_points=4)
    # rgb_images,depth_images,depth_value = mesh_render.render_views(sphere_points,view_up_points,focal_points)
    # show_list = [vedo.Image(im) for im in rgb_images]
    # vedo.show(show_list,N=len(show_list)).close()


    # 测试vedo
    # pcd_render =  PcdRendererByNumpy(mesh.vertices)
    # depth_images = pcd_render.mesh2depth(h_threshold=None,proj_plane_pos=(0, 0, 0), proj_plane_normal=(0, 1, 0))
    # pcd = pcd_render.depth2pointcloud(depth_images)
    # vedo.show([vedo.Image(depth_images),(vedo.Points(pcd),mesh.to_vedo)],N=2).close()

    # 测试o3d
    # o3d_render =  MeshRendererByOpen3d(mesh.to_open3d_t)
    # depth_image,depth,normal_image,rays = o3d_render.mesh2depth(120,eye=(0, -40, 0),center=[0,0,0])
    # pcd = o3d_render.depth2pointcloud(depth,rays)
    # vedo.show([vedo.Image(depth_image),(vedo.Points(pcd), mesh.to_vedo)],N=2).close()

    # 测试NvdiffrastRenderer
    center = mesh.center
    radius =mesh.radius
    device="cuda"
    faces_labels=mesh.get_faces_labels()
    pytorch3d_render =  MeshRendererByPytorch3d(device=device)
    R,T =pytorch3d_render.get_spherical_camera(center, radius)
    #vedo.show([mesh.to_vedo,vedo.Points(camera_positions.cpu().numpy().reshape(-1,3))])
    images,pix_to_face = pytorch3d_render.render(mesh.to_pytorch3d(device=device),R,T)
    # 生成分割标签图
    segmentation_maps = pytorch3d_render.faceslabel2imagelabel(pix_to_face[0][None], torch.from_numpy(faces_labels[None]).reshape(1,-1,1).to(device))
    # 可视化结果（使用第一个视图）
    print(images[0].max(),images[0].shape,segmentation_maps.shape,torch.unique(segmentation_maps))
    s =[vedo.Image(im*255) for im in images.cpu().numpy()]
    seg_img = vedo.Image(segmentation_maps[0].cpu().numpy())
    seg_img.cmap("jet")
    s.append(seg_img)
    vedo.show(s,N=9).close()


