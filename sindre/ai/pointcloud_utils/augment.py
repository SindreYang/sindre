import numpy  as np
import scipy
import random

class ElasticDistortion(object):
    """弹性畸变（通过高斯噪声 + 卷积平滑，模拟非刚性变形"""
    def __init__(self, distortion_params=None):
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
        blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
        blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(
                noise, blurx, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blury, mode="constant", cval=0
            )
            noise = scipy.ndimage.filters.convolve(
                noise, blurz, mode="constant", cval=0
            )

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(
                coords_min - granularity,
                coords_min + granularity * (noise_dim - 2),
                noise_dim,
                )
        ]
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(
                        data_dict["coord"], granularity, magnitude
                    )
        return data_dict


 


def angle_axis_np(angle, axis):
    """
    计算绕给定轴旋转指定弧度的旋转矩阵。
    罗德里格斯公式;

    Args:
        angle (float): 旋转弧度。
        axis (np.ndarray): 旋转轴，形状为 (3,) 的 numpy 数组。

    Returns:
        np.array: 3x3 的旋转矩阵，数据类型为 np.float32。
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                        [u[2], 0.0, -u[0]],
                                        [-u[1], u[0], 0.0]])
    R =cosval * np.eye(3)+ sinval * cross_prod_mat+ (1.0 - cosval) * np.outer(u, u)
    return R

class Flip_np:
    def __init__(self, axis_x=True,axis_y=True,axis_z=True):
        """
        用于随机翻转点云数据。

        Args:
            axis_x (bool): 是否在 x 轴上进行翻转，默认为 True。
            axis_y (bool): 是否在 y 轴上进行翻转，默认为 True。
            axis_z (bool): 是否在 z 轴上进行翻转，默认为 True。
        """
        self.axis_x,self.axis_y,self.axis_z=axis_x,axis_y,axis_z

    def __call__(self, points):
        # 初始化翻转因子，默认为1（不翻转）
        flip_factors = np.ones(3)

        # 根据指定的轴生成伯努利分布的翻转因子（-1或1）
        if self.axis_x:
            flip_factors[0] = np.random.choice([-1, 1], p=[0.5, 0.5])
        if self.axis_y:
            flip_factors[1] = np.random.choice([-1, 1], p=[0.5, 0.5])
        if self.axis_z:
            flip_factors[2] = np.random.choice([-1, 1], p=[0.5, 0.5])
        self.flip_factors=flip_factors
        # 仅影响指定轴方向
        normals = points.shape[1] > 3
        points[:, 0:3] *= flip_factors
        if normals:
            points[:, 3:6] *= flip_factors

        return points
    def get_F(self):
        return self.flip_factors
    
          
 


class Scale_np:
    def __init__(self, lo=0.8, hi=1.25):
        """
        初始化 Scale 类，用于随机缩放点云数据。

        Args:
            lo (float): 缩放因子的下限，默认为 0.8。
            hi (float): 缩放因子的上限，默认为 1.25。
        """
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        self.scaler = np.random.uniform(self.lo, self.hi)

        points[:, 0:3] *= self.scaler
        return points
    def get_S(self):
        return self.scaler


class RotateAxis_np:
    def __init__(self, axis:list=[0.0, 0.0, 1.0], angle:list=[0,360]):
        """
        初始化 RotateAxis 类，用于绕指定轴随机旋转点云数据。

        Args:
            axis (list): 旋转轴，形状为 (3,),默认为 [0.0, 0.0, 1.0]（z 轴）。
            angle (list):在 [min, max] 范围内随机生成角度（单位：度）,默认为 [0, 360],
        """
        self.axis = np.array(axis)
        self.angle =angle

    def __call__(self, points):
        min_angle, max_angle = self.angle
        rotation_angle = np.radians(np.random.uniform(min_angle, max_angle)) # 角度转弧度

        rotation_matrix = angle_axis_np(rotation_angle, self.axis)


        normals = points.shape[1] > 3
        if not normals:
            return np.matmul(points, rotation_matrix.T)
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:6]
            points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
            points[:, 3:6] = np.matmul(pc_normals, rotation_matrix.T)

            return points


class RotateXYZ_np(object):
    def __init__(self, angle_sigma=2, angle_clip=np.pi):
        """
        用于在三个轴上随机微扰旋转点云数据。

        Args:
            angle_sigma (float): 旋转弧度的高斯分布标准差，默认为 2;
            angle_clip (float): 旋转弧度的裁剪范围，默认为 np.pi。
        """
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        


    def __call__(self, points):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )
        
        Rx = angle_axis_np(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis_np(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis_np(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = np.matmul(np.matmul(Rz, Ry), Rx)
        self.R=rotation_matrix

        normals = points.shape[1] > 3
        if not normals:
            return np.matmul(points, rotation_matrix.T)
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:6]
            points[:, 0:3] = np.matmul(pc_xyz, rotation_matrix.T)
            points[:, 3:6] = np.matmul(pc_normals, rotation_matrix.T)

            return points

    def get_R(self):
        return self.R


class Jitter_np(object):
    def __init__(self, std=0.01, clip=0.05):
        """
        用于给点云数据添加随机抖动。

        Args:
            std (float): 抖动的高斯分布标准差，默认为 0.01。
            clip (float): 抖动的裁剪范围，默认为 0.05。
        """
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = np.random.normal(loc=0.0, scale=self.std, size=(points.shape[0], 3))
        self.jittered_data = np.clip(jittered_data, -self.clip, self.clip)
        points[:, 0:3] +=  self.jittered_data
        return points
    def get_J(self):
        return self.jittered_data


class Translate_np(object):
    def __init__(self, translate_range=0.1):
        """
        用于随机平移点云数据。

        Args:
            translate_range (float): 平移的范围，默认为 0.1。
        """
        self.translate_range = translate_range

    def __call__(self, points):
        self.translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += self.translation
        return points
    def get_T(self):
        return  self.translation





class RandomDropout_np(object):
    def __init__(self, max_dropout_ratio=0.2,return_idx=False): 
        """
        用于随机丢弃点云数据中的点。

        Args:
            max_dropout_ratio (float): 最大丢弃比例，范围为 [0, 1)，默认为 0.2。
        """
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio
        self.return_idx = return_idx

    def __call__(self, pc):
        dropout_ratio = np.random.random() * self.max_dropout_ratio  
        ran = np.random.random(pc.shape[0])
        # 找出需要保留的点的索引
        keep_idx = np.where(ran > dropout_ratio)[0]
        if self.return_idx:
            return keep_idx
        else:
            return pc[keep_idx]



class Normalize_np:
    def __init__(self,method="ball",v_range=[0,1]):
        self.method = method
        self.v_range=v_range
        if len(self.v_range) != 2 or not (self.v_range[0] < self.v_range[1]):
            raise ValueError("v_range 需为包含两个元素的列表/元组，且 min_val < max_val")


    def __call__(self,points):
        vertices =points[:,0:3]
        if self.method =="ball":
            # 单位球归一化
            self.centroid = np.mean(vertices, axis=0)
            vertices = vertices -  self.centroid
            self.m = np.max(np.sqrt(np.sum(vertices**2, axis=1)))
            vertices = vertices /  self.m
        else:
            # 矩形归一化
            self.vmax = vertices.max(0, keepdims=True)
            self.vmin = vertices.min(0, keepdims=True)
            vertices = (vertices - self.vmin) / (self.vmax - self.vmin).max()
            # 再从 [0, 1] 区间缩放到 [min_val, max_val] 区间
            vertices = vertices * (self.v_range[-1] - self.v_range[0]) + self.v_range[0]
            
        points[:, 0:3]=vertices
        return points
    
    def get_info(self):
        if self.method =="ball":
            return self.centroid,self.m
        else:
            return self.vmax,self.vmin


class RandomCrop_np:
    def __init__(self, radius=0.15):
        """
        随机移除一个点周围指定半径内的所有点
        
        Args:
            radius (float): 移除半径，默认为0.15
        """
        assert radius >= 0, "Radius must be non-negative"
        self.radius = radius

    def __call__(self, inputs):
        from scipy.spatial import KDTree
        # 提取点云坐标（保留所有通道）
        points = inputs[..., :3]
        # 随机选择一个中心点
        center_idx = np.random.randint(len(points))
        center = points[center_idx]
        # 构建KDTree加速搜索
        tree = KDTree(points)
        # 查询半径范围内的点索引
        remove_indices = tree.query_ball_point(center, self.radius)
        # 生成保留掩码（排除要删除的点）
        mask = np.ones(len(points), dtype=bool)
        mask[remove_indices] = False
        
        return inputs[mask]



if __name__ =="__main__":
    from torchvision import transforms


    transforms_np = transforms.Compose(
        [
            Normalize_np(method="MaxMix",v_range=[0,1]),
            RotateAxis_np(axis=[0,1,0]),
            RotateXYZ_np(angle_sigma=0.05,angle_clip=0.15),
            Scale_np(lo=0.8,hi=1.25),
            Translate_np(translate_range=0.1),
            Jitter_np(std=0.01,clip=0.05),
            RandomDropout_np(max_dropout_ratio=0.2),
            Flip_np(axis_x=False,axis_y=False,axis_z=True),
        ]
    )


    # 示例数据
    points = np.random.randn(1024, 6)  
    points[:,3:6] = np.random.rand(1024,3)
    import time
    e1=time.time()
    for i  in range(50):
        transformed_points_np = transforms_np(points)
    e2=time.time()

    print(e2-e1,transformed_points_np.shape,transformed_points_np.max(),transformed_points_np.min())

    # 0.39881253242492676 torch.Size([910, 6]) tensor(1.3718, device='cuda:0') tensor(-0.5744, device='cuda:0')
    # 0.031032800674438477 torch.Size([856, 6]) tensor(1.3002, device='cuda:0') tensor(-1.2248, device='cuda:0')