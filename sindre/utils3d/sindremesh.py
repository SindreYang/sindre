import logging
from functools import cached_property,lru_cache
import numpy as np
from sindre.utils3d.algorithm import *
from sindre.general.logs import CustomLogger

class SindreMesh:
    """三维网格中转类，假设都是三角面片 """
    def __init__(self, any_mesh=None) -> None:
        # 日志
        self.log = CustomLogger(logger_name="SindreMesh").get_logger()
        self.any_mesh = any_mesh
        self.vertices = None
        self.vertex_colors = None
        self.vertex_normals = None
        self.vertex_curvature =None
        self.vertex_labels = None
        self.vertex_kdtree=None
        self.face_normals = None
        self.faces = None
        if self.any_mesh is not None:
            self._update()


    def clone(self):
        """快速克隆当前网格对象"""
        new_mesh = SindreMesh()
        # 深拷贝数组属性, 因为所有属性一定存在默认值
        new_mesh.vertices = self.vertices.copy()
        new_mesh.vertex_colors = self.vertex_colors.copy()
        new_mesh.vertex_normals = self.vertex_normals.copy()
        new_mesh.vertex_curvature = self.vertex_curvature.copy()
        new_mesh.vertex_labels = self.vertex_labels.copy()
        new_mesh.vertex_kdtree=KDTree(new_mesh.vertices)
        new_mesh.face_normals = self.face_normals.copy()
        new_mesh.faces = self.faces.copy()
        return new_mesh


    def set_vertex_labels(self,vertex_labels):
        """设置顶点labels,并自动渲染颜色"""
        self.vertex_labels=np.array(vertex_labels).reshape(-1,1)
        self.vertex_colors=labels2colors(self.vertex_labels)[...,:3]
    def set_faces_labels(self,faces_labels):
        """将面片labels转换为顶点labels,并自动渲染颜色"""
        vertex_labels =face_labels_to_vertex_labels(self.vertices,self.faces,faces_labels)
        self.vertex_labels=np.array(vertex_labels).reshape(-1,1)
        self.vertex_colors=labels2colors(self.vertex_labels)[...,:3]



    def update_vertex(self,vertex_mask,rebuild=True):
        # 更新顶点属性
        if rebuild:
            log.warning("警告: 面片索引被破坏,开始重建......")
            self.update_geometry(self.vertices[vertex_mask],new_faces=None)
        else:
            self.vertices = self.vertices[vertex_mask]
            self.vertex_colors = self.vertex_colors[vertex_mask]
            self.vertex_normals = self.vertex_normals[vertex_mask]
            self.vertex_curvature = self.vertex_curvature[vertex_mask]
            self.vertex_labels = self.vertex_labels[vertex_mask]
            self.vertex_kdtree = KDTree(self.vertices)
            log.warning("警告: 面片索引被破坏,只能调用顶点属性,请开启rebuild=True")




    def update_faces(self,faces_mask):
        # 更新面片属性
        self.face_normals = self.face_normals[faces_mask]
        self.faces = self.faces[faces_mask]

    def update_geometry(self,new_vertices,new_faces=None):
        """

        更新网格的几何结构（顶点和面片），并通过最近邻算法将原有的顶点属性映射到新顶点上。

        适用于在保持网格拓扑结构基本不变的情况下，对网格进行变形，细化，简化的场景。

        args:
            new_vertices: 形状为(N,3)的浮点型数组，表示新的顶点坐标
            new_faces: 可选参数，形状为(M,3)的整数型数组，表示新的面片索引

        notes:
            - 当新顶点数量与原顶点数量不同时，原顶点属性会根据最近邻关系进行映射
            - 如果未提供新的面片信息，函数会尝试根据旧面片和顶点映射关系重建面片

        """

        new_vertices = np.array(new_vertices, dtype=np.float64)
        if new_vertices.ndim != 2 or new_vertices.shape[1] != 3:
            raise ValueError("顶点坐标必须是(N,3)的二维数组")

        # 重新映射
        near_indices = self.get_near_idx(new_vertices)
        self.vertex_labels = self.vertex_labels[near_indices]
        self.vertex_colors = self.vertex_colors[near_indices]
        self.vertex_curvature = self.vertex_curvature[near_indices]




        # 更新面片
        if new_faces is None:
            new_vertex_count=len(new_vertices)
            # 使用 np.array_equal 比较顶点数组，包括顺序和值
            vertices_changed = not np.array_equal(self.vertices, new_vertices)
            if np.max(self.faces) >= new_vertex_count or vertices_changed:
                self.log.warning(f"警告: 顶点数量/索引发生改变({len(self.vertices)} → {new_vertex_count})，但未提供新的面片信息, 开始重新映射")
                # # 创建旧顶点到新顶点的映射字典
                # vertex_map = {}
                # for new_idx, old_idx in enumerate(near_indices):
                #     if old_idx not in vertex_map:
                #         vertex_map[old_idx] = new_idx

                # # 重新映射面片索引
                # new_faces_list = []
                # invalid_count = 0

                # for  f in self.faces:
                #     new_f = np.array([vertex_map.get(v_idx, -1) for v_idx in f])
                #     # 检查映射后的面片是否有效(不为-1，且face为有效，)
                #     if (-1 not in new_f) and (len(np.unique(new_f)) == 3):
                #         new_faces_list.append(new_f)
                #     else:
                #         invalid_count += 1

                # # 只保留有效映射的面片
                # new_faces = np.array(new_faces_list)
                # self.log.debug(f"成功映射 {len(new_faces)} 个面片，丢弃 {invalid_count} 个无效面片")
                # self.faces=new_faces

                # 创建旧顶点到新顶点的映射数组
                vertex_map_arr = np.full(len(self.vertices), -1, dtype=np.int64)
                unique_old, first_indices = np.unique(near_indices, return_index=True)
                vertex_map_arr[unique_old] = first_indices

                # 批量转换面片索引（向量化操作替代循环）
                new_faces_arr = vertex_map_arr[self.faces]

                # 生成有效性掩码（向量化替代逐面片检查）
                valid_no_invalid = np.all(new_faces_arr != -1, axis=1)
                # 检查三个顶点是否各不相同
                v0, v1, v2 = new_faces_arr.T  # 将面片分解为三个顶点数组
                valid_no_duplicate = (v0 != v1) & (v0 != v2) & (v1 != v2)
                valid_mask = valid_no_invalid & valid_no_duplicate

                # 应用有效性筛选
                new_faces = new_faces_arr[valid_mask]
                invalid_count = len(new_faces_arr) - len(new_faces)

                self.log.debug(f"成功映射 {len(new_faces)} 个面片，丢弃 {invalid_count} 个无效面片")
                self.faces = new_faces

        else:
            self.faces = new_faces

        # 设置新的顶点
        self.vertices = new_vertices
        # 重置kdtree
        self.vertex_kdtree=None


        # 检测面片合法性
        if np.max(self.faces) >= len(self.vertices):
            raise IndexError(f"面片包含非法索引: {np.max(self.faces)}，新顶点数量: {len(self.vertices)}")

        # 重新计算法线
        self.compute_normals(force=True)




    def compute_normals(self,force=False):
        """计算顶点法线及面片法线.force代表是否强制重新计算"""
        if force or self.vertex_normals is None:
            self.vertex_normals = compute_vertex_normals(self.vertices, self.faces)
        if force or self.face_normals is None:
            self.face_normals = compute_face_normals(self.vertices, self.faces)

    def apply_transform_normals(self,mat):
        """处理顶点法线的变换（支持非均匀缩放和反射变换）---废弃，在复杂非正定矩阵，重新计算法线比变换更快，更加准确"""
        RuntimeError("apply_transform_normals ---废弃，在复杂非正定矩阵，重新计算法线比变换更快，更加准确 ")
        # 提取3x3线性变换部分
        linear_mat = mat[:3, :3] if mat.shape == (4, 4) else mat
        # 计算法线变换矩阵（逆转置矩阵）(正交的逆转置是本身)
        try:
            inv_transpose = np.linalg.inv(linear_mat).T
        except np.linalg.LinAlgError:
            inv_transpose = np.eye(3)  # 退化情况处理

        # 应用变换并归一化
        self.vertex_normals = np.dot(self.vertex_normals,inv_transpose)
        norms = np.linalg.norm(self.vertex_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6  # 防止除零
        self.vertex_normals /= norms


        # 将面片法线重新计算
        self.face_normals = None
        self.compute_normals()

    def apply_transform(self,mat):
        """对顶点应用4x4/3x3变换矩阵(支持非正交矩阵)"""
        if mat.shape[0]==4:
            #齐次坐标变换
            homogeneous = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
            self.vertices = (homogeneous @ mat.T)[:, :3]
        else:
            """对顶点应用3*3旋转矩阵"""
            self.vertices = np.dot(self.vertices,mat.T)

        # 计算法线
        self.compute_normals(force=True)

    def apply_inv_transform(self,mat):
        """对顶点应用4x4/3x3变换矩阵进行逆变换(支持非正交矩阵)"""
        mat=np.linalg.inv(mat)
        self.apply_transform(mat)

    def shift_xyz(self,dxdydz):
        """平移xyz指定量,支持输入3个向量和1个向量"""
        dxdydz = np.asarray(dxdydz, dtype=np.float64)  # 统一转换为数组
        if dxdydz.size == 1:
            delta = np.full(3, dxdydz.item())  # 标量扩展为三维
        elif dxdydz.size == 3:
            delta = dxdydz.reshape(3)  # 确保形状正确
        else:
            raise ValueError("dxdydz 应为标量或3元素数组")

        self.vertices += delta
    def scale_xyz(self,dxdydz):
        """缩放xyz指定量,支持输入3个向量和1个向量"""
        dxdydz = np.asarray(dxdydz, dtype=np.float64)
        if dxdydz.size == 1:
            scale = np.full(3, dxdydz.item())
        elif dxdydz.size == 3:
            scale = dxdydz.reshape(3)
        else:
            raise ValueError("dxdydz 应为标量或3元素数组")
        self.vertices *= scale

    def rotate_xyz(self,angles_xyz,return_mat=False):
        """按照给定xyz角度列表进行xyz对应旋转"""
        # 将角度转换为弧度
        angles_xyz_rad = np.radians(angles_xyz)
        Rx = angle_axis_np(angles_xyz_rad[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis_np(angles_xyz_rad[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis_np(angles_xyz_rad[2], np.array([0.0, 0.0, 1.0]))
        rotation_matrix = np.matmul(np.matmul(Rz, Ry), Rx)
        if return_mat:
            return rotation_matrix
        else:
            self.apply_transform(rotation_matrix)



    def _update(self):
        """内部函数，更新相关变量"""

        # 自动转换类型
        if self.any_mesh is not None:
            self._convert()
            # 用完置空，防止使用混乱
            self.any_mesh=None
        # 给定默认颜色
        if self.vertex_colors is None:
            self.vertex_colors = np.ones_like(self.vertices)*np.array([255,0,0]).astype(np.uint8)
        # 给定默认标签
        if self.vertex_labels is None:
            self.vertex_labels = np.ones(len(self.vertices))
        # 给定默认曲率
        if self.vertex_curvature is None:
            self.vertex_curvature = np.zeros(len(self.vertices))
        # 默认启动kdtree
        if self.vertex_kdtree is None:
            self.vertex_kdtree= KDTree( self.vertices)
        # 默认检测法线
        if self.vertices is not None and self.faces is not None:
            self.compute_normals()


    def _convert(self):
        """内部函数，将模型转换到类中"""
        inputobj_type = str(type(self.any_mesh))
        # 专用格式
        if isinstance(self.any_mesh, str) and self.any_mesh.endswith((".sm",".smesh")):
            self.load(self.any_mesh)


        # Trimesh 转换
        elif "Trimesh" in inputobj_type or "primitives" in inputobj_type:
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.faces, dtype=np.int32)
            self.vertex_normals = np.asarray(self.any_mesh.vertex_normals, dtype=np.float64)
            self.face_normals = np.asarray(self.any_mesh.face_normals, dtype=np.float64)

            if self.any_mesh.visual.kind == "face":
                self.vertex_colors = np.asarray(self.any_mesh.visual.face_colors, dtype=np.uint8)
            else:
                self.vertex_colors = np.asarray(self.any_mesh.visual.to_color().vertex_colors, dtype=np.uint8)




        # MeshLab 转换
        elif "MeshSet" in inputobj_type:
            mmesh = self.any_mesh.current_mesh()
            self.vertices = np.asarray(mmesh.vertex_matrix(), dtype=np.float64)
            self.faces = np.asarray(mmesh.face_matrix(), dtype=np.int32)
            self.vertex_normals =np.asarray(mmesh.vertex_normal_matrix(), dtype=np.float64)
            self.face_normals = np.asarray(mmesh.face_normal_matrix(), dtype=np.float64)
            if mmesh.has_vertex_color():
                self.vertex_colors = (np.asarray(mmesh.vertex_color_matrix())[...,:3] * 255).astype(np.uint8)



        # Open3D 转换
        elif "open3d" in inputobj_type:
            import open3d as o3d
            self.any_mesh.compute_vertex_normals()
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.triangles, dtype=np.int32)
            self.vertex_normals = np.asarray(self.any_mesh.vertex_normals, dtype=np.float64)
            self.face_normals = np.asarray(self.any_mesh.triangle_normals, dtype=np.float64)

            if self.any_mesh.has_vertex_colors():
                self.vertex_colors = (np.asarray(self.any_mesh.vertex_colors)[...,:3] * 255).astype(np.uint8)




        # Vedo 转换
        elif (isinstance(self.any_mesh, str)
              or (isinstance(self.any_mesh, list)  and len(self.any_mesh)==2)
              or "vedo" in inputobj_type
              or "vtk" in inputobj_type
              or "meshlib" in inputobj_type
              or "meshio" in inputobj_type
        ):
            import vedo
            if "vedo" not in inputobj_type:
                self.any_mesh=vedo.Mesh(self.any_mesh)
            self.any_mesh.compute_normals()
            self.vertices = np.asarray(self.any_mesh.vertices, dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.cells, dtype=np.int32)
            self.vertex_normals =self.any_mesh.vertex_normals
            self.face_normals =self.any_mesh.cell_normals
            if self.any_mesh.pointdata["PointsRGBA"] is not  None:
                self.vertex_colors = np.asarray(self.any_mesh.pointdata["PointsRGBA"][...,:3], dtype=np.uint8)
            if self.any_mesh.pointdata["RGB"] is not  None:
                self.vertex_colors = np.asarray(self.any_mesh.pointdata["RGB"][...,:3], dtype=np.uint8)



        # pytorch3d 转换
        elif "pytorch3d.structures.meshes.Meshes" in inputobj_type:
            self.any_mesh._compute_vertex_normals(True)
            self.vertices = np.asarray(self.any_mesh.verts_padded().cpu().numpy()[0] ,dtype=np.float64)
            self.faces = np.asarray(self.any_mesh.faces_padded().cpu().numpy()[0], dtype=np.int32)
            self.vertex_normals =self.any_mesh.verts_normals_padded().cpu().numpy()[0]
            self.face_normals =self.any_mesh.faces_normals_padded().cpu().numpy()[0]
            if self.any_mesh.textures is not None:
                self.vertex_colors = np.asarray(self.any_mesh.textures.verts_features_padded().cpu().numpy()[0]*255, dtype=np.uint8)


        elif "OCC" in inputobj_type:
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_FACE
            from OCC.Core.TopLoc import TopLoc_Location
            from OCC.Core.TopoDS import topods
            from OCC.Core.BRep import BRep_Tool

            BRepMesh_IncrementalMesh(self.any_mesh, 0.1).Perform()
            vertices = []
            faces = []
            vertex_index_map = {}
            current_index = 0
            explorer = TopExp_Explorer(self.any_mesh, TopAbs_FACE)
            while explorer.More():
                face = topods.Face(explorer.Current())
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation(face, location)

                if triangulation:
                    nb_nodes = triangulation.NbNodes()
                    for i in range(1, nb_nodes + 1):
                        pnt = triangulation.Node(i)
                        vertex = (pnt.X(), pnt.Y(), pnt.Z())
                        if vertex not in vertex_index_map:
                            vertex_index_map[vertex] = current_index
                            vertices.append(vertex)
                            current_index += 1
                    triangles = triangulation.Triangles()
                    for i in range(1, triangles.Length() + 1):
                        triangle = triangles.Value(i)
                        n1, n2, n3 = triangle.Get()
                        face_indices = [
                            vertex_index_map[(triangulation.Node(n1).X(), triangulation.Node(n1).Y(), triangulation.Node(n1).Z())],
                            vertex_index_map[(triangulation.Node(n2).X(), triangulation.Node(n2).Y(), triangulation.Node(n2).Z())],
                            vertex_index_map[(triangulation.Node(n3).X(), triangulation.Node(n3).Y(), triangulation.Node(n3).Z())]
                        ]
                        faces.append(face_indices)
                explorer.Next()
            self.vertices = np.array(vertices, dtype=np.float64)
            self.faces = np.array(faces, dtype=np.int64)
        else:
            raise RuntimeError(f"不支持类型：{inputobj_type}")
    @property
    def to_occ(self):
        try:
            from OCC.Core.BRepBuilderAPI import (
                BRepBuilderAPI_MakePolygon,
                BRepBuilderAPI_MakeFace,
                BRepBuilderAPI_Sewing,
                BRepBuilderAPI_MakeSolid,
            )
            from OCC.Core.gp import gp_Pnt
            from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
            from OCC.Core.TopAbs import TopAbs_SHELL
            from OCC.Core.TopoDS import topods
        except ImportError:
            raise f"请安装 ：conda install -c conda-forge pythonocc-core=7.8.1.1"
        vertices =self.vertices
        faces =self.faces
        sewing = BRepBuilderAPI_Sewing(0.1)
        for face_indices in faces:
            polygon = BRepBuilderAPI_MakePolygon()
            for idx in face_indices:
                x = float(vertices[idx][0])
                y = float(vertices[idx][1])
                z = float(vertices[idx][2])
                polygon.Add(gp_Pnt(x, y, z))  #
            polygon.Close()
            wire = polygon.Wire()
            face_maker = BRepBuilderAPI_MakeFace(wire)
            if face_maker.IsDone():
                sewing.Add(face_maker.Face())
            else:
                raise ValueError("无法从顶点创建面")
        sewing.Perform()
        sewed_shape = sewing.SewedShape()
        if sewed_shape.ShapeType() == TopAbs_SHELL:
            shell = topods.Shell(sewed_shape)
            solid_maker = BRepBuilderAPI_MakeSolid(shell)
            if solid_maker.IsDone():
                solid = solid_maker.Solid()
                # 网格化确保几何质量
                BRepMesh_IncrementalMesh(solid, 0.1).Perform()
                return solid
            else:
                self.log.warning("警告：Shell无法生成Solid，返回Shell")
                return shell
        else:
            self.log.info("返回原始缝合结果（如Compound）")
            return sewed_shape

    @property
    def to_trimesh(self):
        """转换成trimesh"""
        import trimesh
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.vertex_normals,
            face_normals=self.face_normals
        )
        mesh.visual.vertex_colors = self.vertex_colors
        return mesh
    @property
    def to_meshlab(self):
        """转换成meshlab"""
        import pymeshlab
        ms = pymeshlab.MeshSet()
        v_color_matrix = np.hstack([self.vertex_colors/255, np.ones((len(self.vertices), 1),dtype=np.float64)])
        mesh  =pymeshlab.Mesh(
            vertex_matrix=np.asarray(self.vertices, dtype=np.float64),
            face_matrix=np.asarray(self.faces, dtype=np.int32),
            v_normals_matrix=np.asarray(self.vertex_normals, dtype=np.float64),
            v_color_matrix=v_color_matrix,
        )
        ms.add_mesh(mesh)
        return ms
    @property
    def to_vedo(self):
        """转换成vedo"""
        from vedo import Mesh
        vedo_mesh = Mesh([self.vertices, self.faces])
        vedo_mesh.pointdata["Normals"]=self.vertex_normals
        vedo_mesh.pointdata["labels"]=self.vertex_labels
        vedo_mesh.pointdata["curvature"]=self.vertex_curvature
        vedo_mesh.celldata["Normals"]=self.face_normals
        vedo_mesh.pointcolors = self.vertex_colors
        return vedo_mesh

    @property
    def to_vedo_pointcloud(self):
        """转换成vedo点云"""
        from vedo import Points
        vedo_points = Points(self.vertices)
        vedo_points.pointdata["Normals"]=self.vertex_normals
        vedo_points.pointdata["labels"]=self.vertex_labels
        vedo_points.pointdata["curvature"]=self.vertex_curvature
        vedo_points.pointcolors = self.vertex_colors
        return vedo_points



    @property
    def to_open3d(self):
        """转换成open3d"""
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors[...,:3]/255.0)
        return mesh

    @property
    def to_open3d_t(self,device="CPU:0"):
        """转换成open3d_t"""
        import open3d as o3d
        device = o3d.core.Device(device)
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32
        mesh = o3d.t.geometry.TriangleMesh(device)
        mesh.vertex.positions  = o3d.core.Tensor(self.vertices,dtype=dtype_f,device=device)
        mesh.triangle.indices= o3d.core.Tensor(self.faces,dtype=dtype_i,device=device)
        mesh.vertex.normals = o3d.core.Tensor(self.vertex_normals,dtype=dtype_f,device=device)
        mesh.vertex.colors= o3d.core.Tensor(self.vertex_colors[...,:3]/255.0,dtype=dtype_f,device=device)
        mesh.vertex.labels=o3d.core.Tensor(self.vertex_labels,dtype=dtype_f,device=device)
        return mesh

    @property
    def to_dict(self):
        """将属性转换成python字典"""
        return {
            'vertices': self.vertices ,
            'vertex_colors': self.vertex_colors,
            'vertex_normals': self.vertex_normals,
            'vertex_curvature':self.vertex_curvature,
            'vertex_labels':self.vertex_labels,
            'faces': self.faces ,
        }
    @property
    def to_json(self):
        """转换成json"""
        return json.dumps(self.to_dict,cls=NpEncoder)



    def save(self,write_path):
        """保存mesh,pickle(.sm .smesh),其他由vedo支持 """
        try:
            if write_path.endswith((".sm",".smesh")):
                import pickle
                with open(write_path, 'wb') as f:
                    pickle.dump(self.to_dict, f)
            else:
                self.to_vedo.write(write_path)
            self.log.info(f"Mesh saved to {os.path.abspath(write_path)}")
        except Exception as e:
            self.log.error(f"Failed to save mesh: {e}")


    def load(self,load_path):
        """读取(.sm .smesh)文件"""
        if not os.path.exists(load_path):
            self.log.error(f"File {load_path} does not exist.")
            return
        if not load_path.endswith((".sm",".smesh")):
            self.log.error(f"Only .sm/.smesh format is supported.")
            return
        try:
            import pickle
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            # 从读取的数据中赋值给对象的属性
            self.vertices = data['vertices']
            self.vertex_colors = data['vertex_colors']
            self.vertex_normals = data['vertex_normals']
            self.vertex_curvature = data['vertex_curvature']
            self.vertex_labels = data['vertex_labels']
            self.faces = data['faces']
            self.compute_normals()

            self.log.info(f"Mesh loaded from {os.path.abspath(load_path)}")

        except Exception as e:
            self.log.error(f"Failed to load mesh: {e}")



    def to_torch(self,device="cpu"):
        """将顶点&面片转换成torch形式

        Returns:
            vertices,faces,vertex_normals,vertex_colors: 顶点，面片,法线，颜色（没有则为None)
        """
        import torch
        vertices= torch.from_numpy(self.vertices).to(device,dtype=torch.float32)
        faces= torch.from_numpy(self.faces).to(device,dtype=torch.float32)

        vertex_normals = torch.from_numpy(self.vertex_normals).to(device,dtype=torch.float32)
        if self.vertex_colors is not None:
            vertex_colors = torch.from_numpy(self.vertex_colors).to(device,dtype=torch.int8)
        else:
            vertex_colors = None
        return vertices,faces,vertex_normals,vertex_colors
    def to_pytorch3d(self,device="cpu"):
        """转换成pytorch3d形式

        Returns:
            mesh : pytorch3d类型mesh
        """
        import torch
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import  TexturesVertex
        vertices= torch.from_numpy(self.vertices).to(device,dtype=torch.float32)
        faces= torch.from_numpy(self.faces).to(device,dtype=torch.float32)
        if self.vertex_colors is not None:
            verts_rgb = torch.from_numpy(self.vertex_colors)/255
        else:
            verts_rgb = torch.ones_like(vertices)
        textures = TexturesVertex(verts_features=verts_rgb[None].to(device))
        mesh = Meshes(verts=vertices[None], faces=faces[None],textures=textures)
        return mesh



    def show(self,show_append =[],labels=None,exclude_list=[0],create_axes=True,return_obj=False,by_open3d=False):
        """
        渲染并展示网格数据，支持根据标签着色、添加标记及坐标系显示。

        Args：
            show_append (list)：需要与主网格一起渲染的额外vedo对象列表
            labels (numpy.ndarray, 可选)：网格顶点的标签数组。若提供，将：
                - 根据标签值为顶点分配颜色
                - 为每个不在排除列表中的标签添加标记
            exclude_list (list, 可选)：默认值为[0]，指定不显示标记的标签列表
            create_axes (bool)：是否强制显示世界坐标系，默认为True
            return_obj (bool)：是否返回vedo显示对象列表而非直接渲染，默认为False
            by_open3d (bool)：是否使用Open3D引擎进行渲染，默认为False（使用vedo引擎）

        Returns：
            若return_vedo_obj为True，返回包含所有待显示对象的列表；
            否则无返回值，直接弹出渲染窗口。
        """
        import vedo
        from sindre.utils3d.algorithm import labels2colors
        show_list=[]+show_append
        if by_open3d:
            import open3d as o3d
            mesh_o3d=self.to_open3d
            show_list.append(mesh_o3d)
            if return_obj:
                return show_list
            o3d.visualization.draw(show_list,title='SindreMesh',show_ui=True)
        else:
            mesh_vd=self.to_vedo
            if labels is None and self.vertex_labels is not None and sum(self.vertex_labels) > 0:
                labels=self.vertex_labels
            if labels is not None:
                labels = labels.reshape(-1)
                fss =self._labels_flag(mesh_vd,labels,exclude_list=exclude_list)
                show_list=show_list+fss
                colors = labels2colors(labels)
                mesh_vd.pointcolors=colors
                self.vertex_colors=colors

            show_list.append(mesh_vd)
            if create_axes:
                show_list.append(self._create_vedo_axes(mesh_vd))
            if return_obj:
                return show_list
            # 渲染
            vp = vedo.Plotter(N=1, title="SindreMesh", bg2="black", axes=3)
            vp.at(0).show(show_list)
            vp.close()





    def _create_vedo_axes(self,mesh_vd):
        """
        创建vedo的坐标轴对象。

        Args:
            mesh_vd (vedo.Mesh): vedo的网格对象，用于确定坐标轴的长度。

        Returns:
            vedo.Arrows: 表示坐标轴的vedo箭头对象。
        """
        import vedo
        bounds = mesh_vd.bounds()
        max_length = max([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
        start_points = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        end_points = [(max_length, 0, 0),(0, max_length, 0),(0, 0, max_length)]
        colors = ['red', 'green', 'blue']
        arrows = vedo.Arrows(start_points, end_points, c=colors, shaft_radius=0.005,head_radius=0.01,head_length=0.02)
        return arrows



    def _labels_flag(self,mesh_vd,labels,exclude_list=[0]):
        """
        根据标签为网格的每个非排除类别创建标记。

        Args:
            mesh_vd (vedo.Mesh): vedo的网格对象。
            labels (numpy.ndarray): 网格顶点的标签数组。
            exclude_list (list, optional): 要排除的标签列表，默认为[0]。列表中的标签对应的标记不会被创建。

        Returns:
            list: 包含所有标记对象的列表。
        """
        fss = []
        for i in np.unique(labels):
            if int(i) not in exclude_list:
                v_i = mesh_vd.vertices[labels == i]
                cent = np.mean(v_i, axis=0)
                fs = mesh_vd.flagpost(f"{i}",point=cent)
                fss.append(fs)
        return fss



    def _count_duplicate_vertices(self):
        """统计重复顶点"""
        return len(self.vertices) - len(np.unique(self.vertices, axis=0))

    def _count_degenerate_faces(self):
        """统计退化面片"""
        areas = np.linalg.norm(self.face_normals, axis=1)/2
        return np.sum(areas < 1e-8)

    def _count_connected_components(self):
        """计算连通体数量"""
        from scipy.sparse.csgraph import connected_components
        n_components, labels =connected_components(self.get_adj_matrix, directed=False)
        return n_components, labels

    def _count_unused_vertices(self):
        """统计未使用顶点"""
        used = np.unique(self.faces)
        return len(self.vertices) - len(used)

    def _is_watertight(self):
        """判断是否闭合"""
        unique_edges = np.unique(np.sort(self.get_edges, axis=1), axis=0)
        return len(self.get_edges) == 2*len(unique_edges)


    def split_component(self,return_largest=True):
        """

        将网格按照连通分量分割,并返回最大和其余连通分量的顶点索引

        Returns:
            tuple: 包含三个数组的元组
                - 第一个元素: 连通分量数量
                - 第二个元素: 最大连通分量的节点索引
                - 第三个元素: 剩余部分的节点索引（即非最大连通分量的所有节点）
        """
        # 计算连通分量
        n_components, labels = self._count_connected_components()
        # 提取最大连通标签
        largest_component_label =np.argmax(np.bincount(labels))
        # 提取最大连通分量的节点索引
        largest_component_indices = np.where(labels == largest_component_label)[0]
        if return_largest:
            new_sm = self.clone()
            new_sm.update_geometry(self.vertices[largest_component_indices])
            return new_sm
        # 通过取反而得到剩余部分的节点索引
        remaining_indices = np.where(labels != largest_component_label)[0]
        # 返回最大连通体索引和最小连通索引
        return n_components,largest_component_indices ,remaining_indices

    def get_vertex_labels(self):
        return self.vertex_labels
    def get_faces_labels(self):
        """
        将顶点标签转换成面片标签
        Returns:面片标签

        """
        return vertex_labels_to_face_labels(self.faces,self.vertex_labels)



    def get_color_mapping(self,value):
        """将向量映射为颜色，遵从vcg映射标准"""
        import matplotlib.colors as mcolors
        colors = [
            (1.0, 0.0, 0.0, 1.0),  # 红
            (1.0, 1.0, 0.0, 1.0),  # 黄
            (0.0, 1.0, 0.0, 1.0),  # 绿
            (0.0, 1.0, 1.0, 1.0),  # 青
            (0.0, 0.0, 1.0, 1.0)   # 蓝
        ]
        # colors = [

        #     # 红→黄过渡（R保持1.0，G从0→1）
        #     (1.0, 0.0, 0.0, 1.0),    # 红
        #     (1.0, 0.2, 0.0, 1.0),    # 红→黄过渡1
        #     (1.0, 0.4, 0.0, 1.0),    # 红→黄过渡2
        #     (1.0, 0.6, 0.0, 1.0),    # 红→黄过渡3
        #     (1.0, 0.8, 0.0, 1.0),    # 红→黄过渡4
        #     (1.0, 1.0, 0.0, 1.0),    # 黄

        #     # 黄→绿过渡（G保持1.0，R从1→0）
        #     (0.8, 1.0, 0.0, 1.0),    # 黄→绿过渡1
        #     (0.6, 1.0, 0.0, 1.0),    # 黄→绿过渡2
        #     (0.4, 1.0, 0.0, 1.0),    # 黄→绿过渡3
        #     (0.2, 1.0, 0.0, 1.0),    # 黄→绿过渡4
        #     (0.0, 1.0, 0.0, 1.0),    # 绿

        #     # 绿→青过渡（G保持1.0，B从0→1）
        #     (0.0, 1.0, 0.2, 1.0),    # 绿→青过渡1
        #     (0.0, 1.0, 0.4, 1.0),    # 绿→青过渡2
        #     (0.0, 1.0, 0.6, 1.0),    # 绿→青过渡3
        #     (0.0, 1.0, 0.8, 1.0),    # 绿→青过渡4
        #     (0.0, 1.0, 1.0, 1.0),    # 青

        #     # 青→蓝过渡（B保持1.0，G从1→0）
        #     (0.0, 0.8, 1.0, 1.0),    # 青→蓝过渡1
        #     (0.0, 0.6, 1.0, 1.0),    # 青→蓝过渡2
        #     (0.0, 0.4, 1.0, 1.0),    # 青→蓝过渡3
        #     (0.0, 0.2, 1.0, 1.0),    # 青→蓝过渡4
        #     (0.0, 0.0, 1.0, 1.0),    # 蓝

        #     # # 蓝延伸过渡（B从1→0，R、G保持0）
        #     (0.0, 0.0, 0.8, 1.0),    # 蓝延伸过渡1
        #     (0.0, 0.0, 0.6, 1.0),    # 蓝延伸过渡2
        #     (0.0, 0.0, 0.4, 1.0),    # 蓝延伸过渡3
        #     (0.0, 0.0, 0.2, 1.0)     # 蓝延伸过渡4
        # ]
        cmap = mcolors.LinearSegmentedColormap.from_list("VCG", colors)
        value = np.asarray(value)
        #  10%~90%分位数裁剪（MeshLab固定逻辑，不可调整）
        perc_low = np.percentile(value, 10)
        perc_high = np.percentile(value, 90)
        norm = mcolors.Normalize(vmin=perc_low, vmax=perc_high)
        normalized = norm(value)
        rgb= cmap(normalized)[..., :3]
        return (rgb * 255).astype(np.uint8)





    def subdivison(self,face_mask,iterations=3,method="mid"):
        """局部细分"""

        assert len(face_mask)==len(self.faces),"face_mask长度不匹配:要求每个面片均有对应索引"
        import pymeshlab
        if int(face_mask).max()!=1:
            # # 索引值转bool值
            face_mask = np.any(np.isin(self.faces, face_mask), axis=1)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=self.vertices, face_matrix=self.faces,f_scalar_array=face_mask))
        ms.compute_selection_by_condition_per_face(condselect="fq == 1")

        if method == "mid":
            ms.meshing_surface_subdivision_midpoint(
                iterations=iterations,
                threshold=pymeshlab.PercentageValue(1e-4),
                selected=True
            )
        else:
            ms.meshing_surface_subdivision_ls3_loop(
                iterations=iterations,
                threshold=pymeshlab.PercentageValue(1e-4),
                selected=True
            )
        self.any_mesh=ms
        self._convert()
        return  self




    def texture2colors(self,image_path ="texture_uv.png", uv=None ):
        """将纹理贴图转换成顶点颜色"""
        from PIL import Image
        from scipy.ndimage import map_coordinates
        if uv is None:
            uv=self.get_uv()
        texture = np.array(Image.open(image_path))

        # 转换UV到图像坐标（考虑翻转V轴）
        h, w = texture.shape[:2]
        u_coords = uv[:, 0] * (w - 1)
        v_coords = (1 - uv[:, 1]) * (h - 1)
        coords = np.vstack([v_coords, u_coords])  # scipy的坐标格式为(rows, cols)
        channels = []
        for c in range(3):
            sampled = map_coordinates(texture[:, :, c], coords, order=1, mode='nearest')
            channels.append(sampled)

        self.vertex_colors = np.stack(channels, axis=1).astype(np.uint8)
        return self


    def get_texture(self, image_size = (512, 512),uv=None ):
        """将颜色转换为纹理贴图,  Mesh([v, f]).texture(write_path,uv)"""
        from PIL import Image
        from scipy.interpolate import griddata
        if uv is None:
            uv=self.get_uv()

        def compute_interpolation_map(shape, tcoords, values):
            points = (tcoords * np.asarray(shape)[None, :]).astype(np.int32)
            x = np.arange(shape[0])
            y = np.flip(np.arange(shape[1]))
            X, Y = np.meshgrid(x, y)
            res = griddata(points, values, (X, Y),  method='nearest')
            res[np.isnan(res)] = 0
            return res

        texture_map = compute_interpolation_map(image_size, uv, self.vertex_colors)
        return Image.fromarray(texture_map.astype(np.uint8))

    def sample(self,density=1, num_samples=None):
        """
        网格表面上进行点云重采样
        Args:
            density (float, 可选): 每单位面积的采样点数，默认为1
            num_samples (int, 可选): 指定总采样点数N，若提供则忽略density参数

        Returns:
            numpy.ndarray: 重采样后的点云数组，形状为(N, 3)，N为总采样点数

        """

        return resample_mesh(vertices=self.vertices,faces=self.faces,density=density,num_samples=num_samples)

    def cut_mesh(self,loop_points,get_bigger_part: bool = False,smooth_boundary: bool = False):
        """
        将输入的3D点环投影到网格表面，根据参数决定是生成新的切割边还是仅提取投影轮廓。

        Args:
            loop_points (iterable): 3D点列表/数组，定义要投影到网格上的环状路径。
            get_bigger_part (bool): 选择面积大的mesh；
            smooth_boundary(bool): 优化边界;

        Returns:
            sindremesh:裁剪后的sindremesh
        """

        loop_points=np.array(loop_points).reshape(-1,3)
        v,f,_,_=cut_mesh_with_meshlib(self.vertices,self.faces, loop_points,get_bigger_part, smooth_boundary)
        sm_clone = self.clone()
        sm_clone.update_geometry(v,f)
        return sm_clone


    def decimate(self,n=10000,method=0):

        """
        网格下采样到指定目标顶点数，支持4种简化算法，自动传递顶点属性（标签/曲率）。

        核心逻辑：通过面塌陷（Edge Collapse）或空间分箱实现简化，确保简化后网格的顶点属性（标签、曲率）
        按合理策略聚合（如平均），避免属性丢失。

        Args:
            n: 目标顶点数（int），需满足 3 ≤ n ≤ 原始顶点数（过小会导致网格拓扑无效）。
                注意：method=2（Binned）无法精确控制顶点数，仅能通过分箱间接影响。
            method: 简化算法选择（int，0-3）：
                0: vedo.decimate → 基于VTK QuadricDecimation，速度中，精度高（优先推荐）；
                1: vedo.decimate_pro → 基于VTK DecimatePro，速度中，精度中，支持强拓扑控制；
                2: vedo.decimate_binned → 基于VTK BinnedDecimation，速度高，精度低，无法指定n；
                3: 3: fast-simplification → 基于快速边折叠，速度最高，精度高，支持属性精确聚合，建议百万顶点上调用。

        Returns:
            Self: 简化后的网格对象（更新自身的顶点、面、顶点标签、顶点曲率）。

        """
        if n>=self.npoints:
            self.log.warning(f"网格顶点数小于目标值,无需简化： {self.npoints} <= {n}")
            return self
        vd_ms= self.to_vedo
        if method==0:
            vd_ms.decimate(n=n)
            self.update_geometry(np.asarray( vd_ms.vertices),np.asarray(vd_ms.cells))
            return self
        elif method==1:
            # 自带保留标签
            vd_ms.decimate_pro(n=n)
            self.any_mesh = vd_ms
            self._convert()
            self.vertex_labels=vd_ms.pointdata["labels"]
            self.vertex_curvature=vd_ms.pointdata["curvature"]
        elif method==2:
            # 自带保留标签
            vd_ms.decimate_binned()
            self.any_mesh = vd_ms
            self._convert()
            self.vertex_labels=vd_ms.pointdata["labels"]
            self.vertex_curvature=vd_ms.pointdata["curvature"]
        else:
            import fast_simplification as fs
            points_decim, faces_decim, collapses = fs.simplify(
                np.asarray(self.vertices,dtype=np.float32), self.faces, target_reduction=1.0-(n/self.npoints), return_collapses=True
            )
            points_decim, faces_decim, index_mapping = fs.replay_simplification(
                np.asarray(self.vertices,dtype=np.float32), self.faces, collapses
            )
            # 组建新网格
            self.vertices = np.asarray(points_decim, dtype=np.float64)
            self.faces = np.asarray(faces_decim, dtype=np.int32)

            # 曲率算平均
            unique_values, first_occurrence, counts = np.unique(index_mapping, return_index=True, return_counts=True)
            curvature = np.zeros(len(unique_values))
            np.add.at( curvature,index_mapping,  self.vertex_curvature)
            self.vertex_curvature=curvature/ counts

            # 标签取第一个顶点标签,避免三个标签都不一样
            self.vertex_labels= self.vertex_labels[first_occurrence]

            # 更新法线
            self.compute_normals(True)
            return self

    def homogenize(self,n=10000):
        """ 均匀化网格到指定点数，采用聚类"""
        vd_ms=isotropic_remeshing_by_acvd(self.to_vedo, target_num=n)
        self.update_geometry(np.asarray(vd_ms.vertices),np.asarray(vd_ms.cells))
        return self



    def check(self):
        """检测数据完整性,正常返回True"""
        checks = [
            self.vertices is not None,
            self.faces is not None,
            not np.isnan(self.vertices).any() if self.vertices is not None else False,
            not np.isinf(self.vertices).any() if self.vertices is not None else False,
            not np.isnan(self.vertex_normals).any() if self.vertex_normals is not None else False
        ]
        return all(checks)

    def remove_faces_by_vertex_indices(self, vertex_indices):
        """
        删除所有包含指定顶点索引的面片
        """
        # 创建顶点索引的布尔数组
        vertex_mask = np.zeros(np.max(self.faces) + 1, dtype=bool)
        vertex_mask[vertex_indices] = True
        # 检查每个面片是否包含要删除的顶点
        face_contains_vertex = vertex_mask[self.faces]
        mask = ~np.any(face_contains_vertex, axis=1)
        # 保留不包含指定顶点的面片
        self.update_faces(mask)
        return mask

    def remove_degenerate_faces(self):
        """检测并删除退化面片"""
        # 计算每个面片的面积（使用面法向量的模长/2）
        areas = np.linalg.norm(self.face_normals, axis=1) / 2
        # 创建非退化面片的掩码（面积 >= 1e-8）
        non_degenerate_mask = areas >= 1e-8
        # 统计退化面片数量
        num_degenerate = np.sum(~non_degenerate_mask)
        if num_degenerate > 0:
            # 删除退化面片
            self.update_faces(non_degenerate_mask)
        return num_degenerate


    def get_normalize(self, method="ball"):
        """对顶点、曲率和颜色数据进行归一化处理

        该方法支持两种归一化方式，可将顶点坐标、曲率值和颜色值
        归一化到特定范围，便于后续处理和分析。

        Args:
            method (str, optional): 归一化方法。可选值为"ball"或其他。
                "ball"表示将顶点中心移至原点并缩放至单位球内；
                其他值表示将顶点缩放到[0,1]范围。默认值为"ball"。

        Returns:
            dict: 包含归一化后的数据字典，键值对如下：
                - "vertices":归一化后的顶点坐标数组（已转换method参数为范围）
                - "normals": 法线（已转换为[-1,1]范围）
                - "curvature": 归一化后的曲率值数组（已转换为[-1,1]范围）
                - "colors": 归一化后的颜色值数组（已转换为[0,1]范围）
        """
        vertices = self.vertices
        if method == "ball":
            # 计算顶点中心并平移到原点
            centroid = np.mean(vertices, axis=0)
            vertices -= centroid
            # 计算最大距离并缩放到单位球内
            m = np.max(np.sqrt(np.sum(vertices**2, axis=1)))
            vertices /= m
        else:
            # 计算顶点坐标范围并缩放到[0,1]
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            vertices = (vertices - vmin) / (vmax - vmin).max()

        # 标准化曲率值到[0,1]范围（使用1%和99%分位数去除极端值）
        curvature = self.vertex_curvature
        if sum(curvature)!=0:
            k_low = np.percentile(curvature, 1)
            k_high = np.percentile(curvature, 99)
            curvature = np.clip(curvature,k_low,k_high)
            curvature = (curvature - k_low) / (k_high - k_low )
            curvature =curvature*2-1

        # 将颜色值从[0,255]范围转换到[0,1]
        colors = self.vertex_colors / 255

        return {
            "vertices": vertices,
            "normals":self.vertex_normals,
            "curvature": curvature,
            "colors": colors,

        }

    def get_unused_vertices(self):
        """获取未使用顶点的索引"""
        # 获取所有在faces中出现过的顶点索引
        used_indices = np.unique(self.faces)
        # 生成所有顶点索引
        all_indices = np.arange(len(self.vertices))
        # 找出未使用的顶点索引
        unused_indices = np.setdiff1d(all_indices, used_indices)
        # 返回索引列表
        return unused_indices.tolist()


    def get_curvature(self):
        """获取曲率"""
        vd_ms = self.to_vedo.compute_curvature(method=1)
        self.vertex_curvature =vd_ms.pointdata["Mean_Curvature"]
        self.vertex_colors =self.get_color_mapping(self.vertex_curvature)
        return self
    def get_curvature_igl(self):
        if self._count_degenerate_faces()>0:
            log.warning("网格存在退化面片，执行删除面片")
            self.remove_degenerate_faces()
        self.vertex_curvature =compute_curvature_by_igl(self.vertices,self.faces)
        self.vertex_colors =self.get_color_mapping(self.vertex_curvature)
        return self



    def get_curvature_meshlab(self):
        """使用MeshLab获取更加精确曲率，自动处理非流形几何"""
        # 限制太多，舍弃
        # assert self.npoints<100000,"顶点必须小于10W"
        # assert len(self.get_non_manifold_edges)==0,"存在非流形"
        # assert self._count_connected_components()[0]==1,"连通体数量应为1"
        ms = self.to_meshlab
        # 检查非流形边并修复
        if len(self.get_non_manifold_edges) > 0:  # 修复：添加括号调用函数，并修正判断条件
            log.warning("网格存在非流形，开始进行删除非流形面片处理")
            ms.meshing_repair_non_manifold_edges(method='Remove Faces')
            ms.meshing_remove_unreferenced_vertices()

        # 计算主曲率方向
        ms.compute_curvature_principal_directions_per_vertex(method= 'Quadric Fitting',curvcolormethod='Mean Curvature', autoclean=False)
        mmesh = ms.current_mesh()
        verts = np.array(mmesh.vertex_matrix())
        curvature = mmesh.vertex_scalar_array()
        colors =self.get_color_mapping(curvature) #(mmesh.vertex_color_matrix() * 255)[..., :3]
        # 检查顶点数量是否变化
        if len(verts) != self.npoints:
            log.warning("检测到修复后顶点被删除，执行曲率/颜色映射...")
            # 为原始网格每个顶点找到最近的点
            from scipy.spatial import cKDTree
            kdtree = cKDTree(verts)
            _, indices = kdtree.query(self.vertices, k=1)
            # 映射曲率和颜色
            self.vertex_curvature = curvature[indices]
            self.vertex_colors = colors[indices]
        else:
            self.vertex_curvature =curvature
            self.vertex_colors = colors
        return self



    def get_near_idx(self,query_vertices):
        """获取最近索引"""
        if self.vertex_kdtree is None:
            self.vertex_kdtree= KDTree( self.vertices)
        _,idx = self.vertex_kdtree.query(query_vertices,workers=-1)
        return idx

    def get_boundary_by_ref_normal_angle(self,ref_normal=[0, 0, -1],angle=30):
        """

        通过参考法线和角度阈值获取网格边界顶点


        Note:
            将输入的参考法线转换为 numpy 数组
            计算网格所有面的法线与参考法线的余弦相似度
            筛选出与参考法线夹角小于阈值角度的面（余弦值大于阈值角度的余弦值）
            对筛选出的面进行处理，提取并返回其边界顶点

        Args:

            self: 网格对象，需包含面法线 (face_normals)、顶点 (vertices) 和面 (faces) 属性；

            ref_normal: 参考法线向量，默认值为 [0, 0, -1] 朝向-z方向;

            angle: 角度阈值 (度)，默认 30 度，用于筛选与参考法线夹角小于该值的面；

        Returns:

            边界顶点坐标；

        """
        ref_normal = np.array(ref_normal)  # 参考法线
        cos_angle = np.dot(self.face_normals, ref_normal) / (np.linalg.norm(self.face_normals, axis=1) * np.linalg.norm(ref_normal))
        faces_mask= cos_angle>np.cos(np.radians(angle))
        new_faces = self.faces[faces_mask]
        from vedo import Mesh
        boundary = Mesh([self.vertices,new_faces]).split()[0].clean().boundaries().join(reset=True).vertices
        return boundary


    def get_boundary_by_normal_angle(self, angle_threshold=30,max_boundary=True):
        """
        通过相邻三角面法线夹角识别特征边界环

        Note:
            1. 计算所有三角面的归一化法向量
            2. 遍历网格所有边，筛选出相邻两面法线夹角大于阈值的边（特征边）
            3. 将特征边连接成有序封闭环

        Args:
            self: 必须为水密网格;
            angle_threshold: 法线夹角阈值(度),默认30度;
            max_boundary: 是否仅返回最长边界环,默认True;

        Returns:
            边界环顶点索引列表(若max_boundary=True则返回单个环)
        """
        """
        # 确保网格是水密的
        if not self._is_watertight():
            raise ValueError("Mesh is not watertight")

        # 计算面法线（已归一化）
        face_normals = self.face_normals.copy()
        from collections import defaultdict
        
        
        # 获取所有边及其相邻面
        edges = self.get_edges
        edge_faces = defaultdict(list)
        for i, face in enumerate(self.faces):
            for edge in ([face[0], face[1]], [face[1], face[2]], [face[2], face[0]]):
                edge_faces[tuple(sorted(edge))].append(i)
        
        # 找出特征边（法线夹角大于阈值）
        feature_edges = []
        for edge, faces in edge_faces.items():
            if len(faces) == 2:
                n1 = face_normals[faces[0]]
                n2 = face_normals[faces[1]]
                angle = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
                if angle > angle_threshold:
                    feature_edges.append((edge[0], edge[1]))
        
        # 构建边连接图
        edge_graph = defaultdict(list)
        for u, v in feature_edges:
            edge_graph[u].append(v)
            edge_graph[v].append(u)

        
        # 连接特征边形成有序环
        boundaries = []
        visited_edges = set()
        
        for edge in feature_edges:
            if tuple(sorted(edge)) in visited_edges:
                continue
                
            # 开始新环
            loop = []
            start = edge[0]
            current = edge[1]
            prev = start
            
            # 追踪边界环
            while True:
                loop.append(prev)
                visited_edges.add(tuple(sorted((prev, current))))
                
                # 查找下一个顶点
                neighbors = [n for n in edge_graph[current] if n != prev]
                if not neighbors:
                    break
                    
                next_v = neighbors[0]
                prev, current = current, next_v
                
                # 回到起点形成闭环
                if current == start:
                    loop.append(prev)
                    break
            # 排除异常环
            if len(loop)>len(feature_edges)*0.1:
                boundaries.append(np.array(loop))
        if max_boundary:
            return sorted(boundaries, key=lambda b: len(b), reverse=True)[0]
        return boundaries
        """
        """"
        DFS迭代替代递归：

            1. 使用显式栈 (stack) 替代递归调用

            2. 每个栈元素保存 (prev, current, path) 三元组

            3. 避免递归深度过大导致内存溢出
        """

        if not self._is_watertight():
            raise ValueError("Mesh is not watertight")

        # 计算面法线（已归一化）
        face_normals = self.face_normals.copy()
        from collections import defaultdict

        # 获取所有边及其相邻面
        edges = self.get_edges
        edge_faces = defaultdict(list)
        for i, face in enumerate(self.faces):
            for edge in ([face[0], face[1]], [face[1], face[2]], [face[2], face[0]]):
                edge_faces[tuple(sorted(edge))].append(i)

        # 找出特征边（法线夹角大于阈值）
        feature_edges = []
        for edge, faces in edge_faces.items():
            if len(faces) == 2:
                n1 = face_normals[faces[0]]
                n2 = face_normals[faces[1]]
                angle = np.degrees(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))
                if angle > angle_threshold:
                    feature_edges.append((edge[0], edge[1]))

        # 构建边连接图（增加边计数）
        edge_graph = defaultdict(list)
        edge_counter = defaultdict(int)
        for u, v in feature_edges:
            edge_graph[u].append(v)
            edge_graph[v].append(u)
            edge_counter[u] += 1
            edge_counter[v] += 1

        # 连接特征边形成有序环（DFS迭代版）
        boundaries = []
        visited_edges = set()
        stack = []

        # 创建边访问标记（使用冻结集合）
        for edge in feature_edges:
            frozen_edge = frozenset(edge)
            if frozen_edge in visited_edges:
                continue

            stack.append((edge[0], edge[1], [edge[0]]))  # (prev, current, path)
            visited_edges.add(frozen_edge)
            current_loop = None

            while stack:
                prev, current, path = stack.pop()
                path.append(current)

                # 成功闭合环路
                if current == path[0] and len(path) > 1:
                    current_loop = path.copy()
                    break

                # 获取未访问的邻居边
                neighbors = []
                for neighbor in edge_graph[current]:
                    frozen = frozenset({current, neighbor})
                    if frozen not in visited_edges:
                        neighbors.append(neighbor)

                # 死胡同处理
                if not neighbors:
                    continue

                # 优先选择度数为2的顶点（减少分支）
                neighbors.sort(key=lambda x: edge_counter[x])

                # 处理第一个邻居
                next_hop = neighbors[0]
                new_edge = frozenset({current, next_hop})
                visited_edges.add(new_edge)
                stack.append((current, next_hop, path.copy()))

                # 处理其他邻居（新分支）
                for neighbor in neighbors[1:]:
                    new_edge = frozenset({current, neighbor})
                    visited_edges.add(new_edge)
                    stack.append((current, neighbor, path[:-1].copy()))

            # 保存有效环路
            if current_loop is not None and len(current_loop) > 3:
                # 闭合环检查（首尾重复）
                if current_loop[0] == current_loop[-1]:
                    boundaries.append(np.array(current_loop))
                # 清理栈状态
                stack = []

        # 返回结果
        if not boundaries:
            return np.array([], dtype=int) if max_boundary else []

        if max_boundary:
            return sorted(boundaries, key=len, reverse=True)[0]
        return boundaries

    def get_boundary(self,return_points=True,max_boundary=False):
        """获取非水密网格的边界环（可能有多个环）;

            Method:
            1. 获取所有的边（未去重），并统计每条边出现的次数。在三角网格中，内部边会被两个三角形共享，而边界边只被一个三角形使用。
            2. 筛选出只出现一次的边，这些边就是边界边。
            3. 将边界边连接成有序的环（或多个环）。通过构建边界边的图，然后进行深度优先搜索或广度优先搜索来连接相邻的边。

            Args:
                return_points (bool): True返回顶点坐标,False返回顶点索引
                max_boundary (bool): True时只返回最大的边界环(按顶点数量)
            Return:
                list: 边界环列表，每个环是顶点索引的有序序列（闭合环，首尾顶点相同）,当max_boundary=True,单个边界环数组
        """
        # 1. 获取所有边并统计出现次数
        edges = self.get_edges  # 获取所有边（已排序去重）
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]  # 筛选边界边

        if len(boundary_edges) == 0:
            return []  # 无水密边界

        # 2. 构建邻接字典（高效版本）
        adj = {}
        for u, v in boundary_edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)

        loops = []
        # 3. 遍历连接边界点
        while adj:
            start = next(iter(adj))  # 取任意起点
            loop = [start]
            current = start

            while True:
                # 检查当前节点是否仍在邻接字典中
                if current not in adj or not adj[current]:
                    # 当前节点已被移除或没有邻接点，结束当前环
                    break
                # 获取下一顶点并移除已使用边
                next_vertex = adj[current].pop()
                # 移除反向连接（如果存在）
                if next_vertex in adj and current in adj[next_vertex]:
                    adj[next_vertex].remove(current)

                # 清理空节点
                if not adj[current]:
                    del adj[current]
                if next_vertex in adj and not adj[next_vertex]:
                    del adj[next_vertex]

                # 闭环检测
                if next_vertex == start:
                    loops.append(loop + [start])  # 闭合环
                    break

                # 继续遍历
                loop.append(next_vertex)
                current = next_vertex
                # 防止无限循环的安全措施
                if len(loop) > len(boundary_edges) * 2:
                    break



        # 4. 根据max_boundary参数筛选结果
        if max_boundary:
            # 找到顶点最多的边界环
            longest_idx = np.argmax([len(loop) for loop in loops])
            result = loops[longest_idx]
        elif max_boundary:
            return []
        else:
            result = loops

        # 5. 根据return_points参数转换为坐标
        if return_points:
            if max_boundary:
                return self.vertices[result]
            elif max_boundary:
                return []
            else:
                return [self.vertices[loop] for loop in result]
        else:
            return result

    def clean(self):
        """
        删除退化面片,删除孤立顶点
        """
        # 删除退化面片
        self.remove_degenerate_faces()
        # 删除未使用的顶点
        unused_vertices_indices = self.get_unused_vertices()
        if unused_vertices_indices:
            # 创建顶点掩码，标记哪些顶点需要保留
            vertex_mask = np.ones(len(self.vertices), dtype=bool)
            vertex_mask[unused_vertices_indices] = False
            # 更新面片索引
            index_map = np.cumsum(vertex_mask) - 1
            new_faces = index_map[self.faces]
            # 移除无效面片（包含已删除顶点的面片）
            valid_faces_mask = np.all(new_faces >= 0, axis=1)
            # 更新面片
            self.faces = new_faces
            self.update_faces(valid_faces_mask)
            self.update_vertex(vertex_mask)
        return self

    def fix_mesh(self):
        """修复基本mesh错误"""
        ms = self.to_meshlab
        # 去除较小连通体
        fix_component_by_meshlab(ms)
        # 先修复非流形
        fix_topology_by_meshlab(ms)
        # 清理无效结构
        fix_invalid_by_meshlab(ms)
        # 更新信息
        mmesh = ms.current_mesh()
        self.update_geometry( np.asarray(mmesh.vertex_matrix(), dtype=np.float64),
                              np.asarray(mmesh.face_matrix(), dtype=np.int32))


    @lru_cache(maxsize=None)
    def get_uv(self,return_circle=False):
        """ 获取uv映射 与顶点一致(npoinst,2) """
        uv,_= harmonic_by_igl(self.vertices,self.faces,map_vertices_to_circle=return_circle)
        return uv

    @cached_property
    def npoints(self):
        """获取顶点数量"""
        return len(self.vertices)
    @cached_property
    def nfaces(self):
        """获取顶点数量"""
        return len(self.faces)


    @cached_property
    def faces_vertices(self):
        """将面片索引用顶点来表示"""
        return  self.vertices[self.faces]


    @cached_property
    def faces_area(self):
        """
        计算每个三角形面片的面积。

        Notes:
            使用叉乘公式计算面积：
            面积 = 0.5 * ||(v1 - v0) × (v2 - v0)||
        """
        tri_vertices =self.faces_vertices
        v0, v1, v2 = tri_vertices[:, 0], tri_vertices[:, 1], tri_vertices[:, 2]
        area = 0.5 * np.linalg.norm(np.cross((v1 - v0), (v2 - v0)), axis=1)
        return area

    @cached_property
    def faces_barycentre(self):
        """每个三角形的中心（重心 [1/3,1/3,1/3]）"""
        return  self.faces_vertices.mean(axis=1)


    @cached_property
    def center(self) -> np.ndarray:
        """计算网格的加权质心（基于面片面积加权）。

        Returns:
            np.ndarray: 加权质心坐标，形状为 (3,)。

        Notes:
            使用三角形面片面积作为权重，对三角形质心坐标进行加权平均。
            该结果等价于网格的几何中心。
        """
        return np.average(self.faces_barycentre, weights=self.faces_area, axis=0)
    @cached_property
    def radius(self)->float:
        return np.linalg.norm(self.vertices -self.center, axis=1).max()





    @cached_property
    def get_adj_matrix(self):
        """基于去重边构建邻接矩阵"""
        from scipy.sparse import csr_matrix
        n = len(self.vertices)
        edges = np.unique(self.get_edges, axis=0)
        data = np.ones(edges.shape[0] * 2)  # 两条边（无向图）
        rows = np.concatenate([edges[:,0], edges[:,1]])
        cols = np.concatenate([edges[:,1], edges[:,0]])
        return csr_matrix((data, (rows, cols)), shape=(n, n))

    @cached_property
    def get_adj_list(self):
        """邻接表属性"""
        edges = np.unique(self.get_edges, axis=0)  # 去重
        adj = [[] for _ in range(len(self.vertices))]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    @cached_property
    def get_edges(self):
        """未去重边缘属性 """
        edges = np.concatenate([self.faces[:, [0,1]],
                                self.faces[:, [1,2]],
                                self.faces[:, [2,0]]], axis=0)
        edges = np.sort(edges, axis=1)  # 确保边无序性
        return edges

    @cached_property
    def get_non_manifold_edges(self):
        # 提取有效边并排序
        edges =self.get_edges
        valid_edges = edges[edges[:,0] != edges[:,1]]
        edges_sorted = np.sort(valid_edges, axis=1)
        unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        # 返回非流形边
        return unique_edges[counts >= 3]




    def __repr__(self):
        """网格质量检测"""

        stats = [
            "\033[91m\t网格质量检测(numpy): \033[0m",
            f"\033[94m顶点数:             {len(self.vertices) if self.vertices is not None else 0}\033[0m",
            f"\033[94m面片数:             {len(self.faces) if self.faces is not None else 0}\033[0m",
            f"\033[94m顶点范围:           {self.vertices.min():.2f} ~ {self.vertices.max():.2f} \033[0m",
            f"\033[94m曲率范围:           {self.vertex_curvature.min():.2f} ~ {self.vertex_curvature.max():.2f}\033[0m",
            f"\033[94m标签类别:           {', ' .join([f'{label}' for label in np.unique(self.vertex_labels)])}\033[0m",
            f"\033[94m最大半径:           {self.radius:.2f}\033[0m",
            f"\033[94m几何中心:           {', '.join(f'{x:.3f}' for x in self.center)}\033[0m",
            f"\033[94m曲率大小:           {self.vertex_curvature.min():.2f} ~ {self.vertex_curvature.max():.2f}\033[0m",
            f"\033[94m网格水密(闭合):     {self._is_watertight()}\033[0m",
            f"\033[94m连通体数量:         {self._count_connected_components()[0]}\033[0m",
            f"\033[94m未使用顶点:         {self._count_unused_vertices()}\033[0m",
            f"\033[94m重复顶点:           {self._count_duplicate_vertices()}\033[0m",
            f"\033[94m网格退化:           {self._count_degenerate_faces()}\033[0m",
            f"\033[94m法线异常:           {np.isnan(self.vertex_normals).any()}\033[0m",
            f"\033[94m边流形:             {len(self.get_non_manifold_edges)==0}\033[0m",
        ]

        return "\n".join(stats)



    def print_o3d(self):
        """使用open3d网格质量检测"""
        mesh = self.to_open3d
        edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
        edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
        vertex_manifold = mesh.is_vertex_manifold()
        orientable = mesh.is_orientable()



        stats = [
            "\033[91m\t网格质量检测(open3d): \033[0m",
            f"\033[94m顶点数:             {len(self.vertices) if self.vertices is not None else 0}\033[0m",
            f"\033[94m面片数:             {len(self.faces) if self.faces is not None else 0}\033[0m",
            f"\033[94m顶点范围:           {self.vertices.min():.2f} ~ {self.vertices.max():.2f} \033[0m",
            f"\033[94m曲率范围:           {self.vertex_curvature.min():.2f} ~ {self.vertex_curvature.max():.2f}\033[0m",
            f"\033[94m标签类别:           {', '.join([f'{label}' for label in np.unique(self.vertex_labels)])}\033[0m",
            f"\033[94m最大半径:           {self.radius:.2f}\033[0m",
            f"\033[94m几何中心:           {', '.join(f'{x:.3f}' for x in self.center)}\033[0m",
            f"\033[94m网格水密(闭合):     {self._is_watertight()}\033[0m",
            f"\033[94m连通体数量:         {self._count_connected_components()[0]}\033[0m",
            f"\033[94m未使用顶点:         {self._count_unused_vertices()}\033[0m",
            f"\033[94m重复顶点:           {self._count_duplicate_vertices()}\033[0m",
            f"\033[94m网格退化:           {self._count_degenerate_faces()}\033[0m",
            f"\033[94m法线异常:           {np.isnan(self.vertex_normals).any() if self.vertex_normals is not None else True}\033[0m",
            f"\033[94m边为流形:           {edge_manifold}\033[0m",
            f"\033[94m边的边界为流形:     {edge_manifold_boundary}\033[0m",
            f"\033[94m顶点为流形:         {vertex_manifold}\033[0m",
            f"\033[94m可定向:             {orientable}\033[0m",
        ]

        print("\n".join(stats))




