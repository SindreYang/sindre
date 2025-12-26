import numpy  as np
import scipy
import random

class ElasticDistortion_np(object):
    """
       点云弹性畸变增强类（通过高斯噪声生成+三维卷积平滑+三线性插值，模拟非刚性几何变形）

       核心原理：
       1. 在点云坐标空间的网格上生成高斯噪声场；
       2. 对噪声场进行三维卷积平滑，模拟连续的弹性变形；
       3. 通过三线性插值将噪声场映射到每个点的坐标位置，叠加畸变偏移量实现弹性变形。
   """
    def __init__(self, distortion_params=None):

        """
       初始化弹性畸变参数

       Args:
           distortion_params (list of list, optional): 弹性畸变参数列表，每个元素为[granularity, magnitude]，其中：
               - granularity (float): 噪声网格的粒度（单位与点云坐标一致，如m/cm），决定畸变的精细程度，值越小畸变越精细；
               - magnitude (float): 噪声幅值乘数，决定畸变的强度，值越大变形越明显。
               若为None，使用默认参数[[0.2, 0.4], [0.8, 1.6]]，即依次应用两组畸变参数增强变形多样性。

       Example:
           distortion_params=[[0.1, 0.3], [0.5, 1.0]] 表示先以粒度0.1、幅值0.3处理，再以粒度0.5、幅值1.0处理。

       输入格式要求：

       """
        self.distortion_params = (
            [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params
        )

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        执行单组参数的弹性畸变处理

        Args:
            coords (np.ndarray): 点云坐标数组，形状为(N, 3)，N为点数，3对应x/y/z轴坐标（浮点型）；
            granularity (float): 噪声网格粒度（坐标空间单位），网格越小，畸变细节越丰富；
            magnitude (float): 畸变幅值乘数，控制畸变的整体强度。

        Returns:
            np.ndarray: 畸变后的点云坐标数组，形状与输入coords一致（N, 3）。

        处理步骤：
            1. 计算点云坐标的最小值，确定噪声网格的起始位置；
            2. 根据网格粒度计算噪声场的维度，生成高斯随机噪声场（三维+3个坐标轴通道）；
            3. 对噪声场进行三次三维卷积平滑（x/y/z轴分别卷积），增强噪声场的连续性；
            4. 构建噪声场的规则网格插值器，通过三线性插值计算每个点的畸变偏移量；
            5. 将畸变偏移量乘以幅值乘数后叠加到原始坐标，得到畸变结果。
        """
        # 定义三维卷积核（x/y/z轴分别为3×1×1、1×3×1、1×1×3，实现轴向平滑）
        blurx = np.ones((3, 1, 1, 1), dtype=np.float32) / 3  # x轴平滑核
        blury = np.ones((1, 3, 1, 1), dtype=np.float32) / 3  # y轴平滑核
        blurz = np.ones((1, 1, 3, 1), dtype=np.float32) / 3  # z轴平滑核

        # 计算点云坐标的最小值，作为噪声网格的空间起始基准
        coords_min = coords.min(axis=0)

        # 计算噪声场的维度：基于点云坐标范围与网格粒度，扩展3个网格单元避免边界效应
        coords_range = (coords - coords_min).max(axis=0)  # 点云在各轴的坐标范围
        noise_dim = (coords_range // granularity).astype(int) + 3  # 噪声场的三维维度（x/y/z）
        # 生成三维高斯噪声场（shape=(noise_dim_x, noise_dim_y, noise_dim_z, 3)），3对应x/y/z轴的畸变偏移
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # 对噪声场进行二维卷积平滑（重复2次增强平滑效果，模拟连续弹性变形）
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)  # x轴平滑
            noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)  # y轴平滑
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)  # z轴平滑

        # 构建噪声场的规则网格插值器（三线性插值）
        # 生成各轴的网格坐标值（从coords_min - granularity到coords_min + granularity*(noise_dim-2)，共noise_dim个点）
        ax = [
            np.linspace(d_min - granularity, d_min + granularity * (dim - 2), dim, dtype=np.float32)
            for d_min, dim in zip(coords_min, noise_dim)
        ]
        # 创建规则网格插值器，边界外填充0（避免点云边界点的畸变偏移过大）
        interp = scipy.interpolate.RegularGridInterpolator(
            ax, noise, bounds_error=False, fill_value=0
        )

        # 插值计算每个点的畸变偏移量，乘以幅值乘数后叠加到原始坐标
        coords += interp(coords) * magnitude

        return coords

    def __call__(self, feat):
        coord=feat[...,:3].copy()
        if random.random() < 0.95:
            for granularity, magnitude in self.distortion_params:
                coord = self.elastic_distortion(coord, granularity, magnitude)
        return coord


 


def get_angle_axis_np(angle, axis):
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


class FlipXYZ_np:
    def __init__(self, axis_x=True, axis_y=True, axis_z=True):
        """
        用于随机翻转点云数据（支持任意3倍数列数输入）。

        Args:
            axis_x (bool): 是否在 x 轴上进行翻转，默认为 True。
            axis_y (bool): 是否在 y 轴上进行翻转，默认为 True。
            axis_z (bool): 是否在 z 轴上进行翻转，默认为 True。

        输入格式要求：
            传入的数组形状为(N, C)，C为3的倍数，列顺序按「点云(3列)、法线(3列)、点云(3列)、法线(3列)……」循环排列。
        """
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z
        self.flip_factors = None  # 存储翻转因子（x/y/z轴，1为不翻转，-1为翻转）

    def get_filp(self):
        # 初始化翻转因子为1（不翻转）
        flip_factors = np.ones(3, dtype=np.float32)
        # 按指定轴生成翻转因子（伯努利分布，50%概率翻转）
        if self.axis_x:
            flip_factors[0] = np.random.choice([-1, 1], p=[0.5, 0.5])
        if self.axis_y:
            flip_factors[1] = np.random.choice([-1, 1], p=[0.5, 0.5])
        if self.axis_z:
            flip_factors[2] = np.random.choice([-1, 1], p=[0.5, 0.5])
        self.flip_factors = flip_factors
        return flip_factors
    def apply_flip(self,feat,flip_factors):
        # 检查输入列数
        C = feat.shape[1]
        assert C % 3 == 0, f"输入列数{C}需为3的倍数，当前输入形状：{feat.shape}"
        # 复制数组避免原地修改
        feat_flipped = feat.copy()
        for i in range(0, C, 3):
            feat_flipped[:, i:i+3] *= flip_factors

        return feat_flipped

    def __call__(self, feat):
        """
        执行翻转操作：对所有点云组和法线组按翻转因子同步翻转。

        Args:
            feat (np.ndarray): 输入数组，形状为(N, C)，C为3的倍数。

        Returns:
            np.ndarray: 翻转后的数组，形状与输入一致。

        Raises:
            AssertionError: 输入列数非3的倍数时触发。
        """
        flip_factors = self.get_filp()
        feat_flipped = self.apply_flip(feat,flip_factors)
        return feat_flipped


    def get_info(self):
        """获取翻转因子"""
        if self.flip_factors is None:
            raise RuntimeError("尚未执行翻转操作（请先调用__call__方法）")
        return self.flip_factors


class FlipAxis_np:
    """
    实现点云数据的随机单轴翻转增强（Numpy版，支持任意3倍数列数输入）

    功能说明：
    1. 以指定概率触发单轴翻转操作，未触发时不进行任何翻转；
    2. 触发翻转时，随机选择x/y/z轴中的一个轴，对该轴所有点云组和法线组进行符号翻转；
    3. 支持输入为任意3倍数列数的数组，列顺序按「点云(3列)、法线(3列)……」循环排列。
    """
    def __init__(self, flip_prob: float = 0.75):
        """
        初始化单轴翻转增强器

        Args:
            flip_prob (float): 触发单轴翻转操作的总概率，取值范围[0,1]，默认0.75（75%概率触发翻转）
        """
        if not 0 <= flip_prob <= 1:
            raise ValueError(f"flip_prob需在[0,1]范围内，当前值：{flip_prob}")
        self.flip_prob = flip_prob  # 触发单轴翻转的总概率
        self.flip_factors = None   # 轴翻转因子（1为不翻转，-1为翻转），对应x/y/z轴


    def get_flip(self):
        # 初始化翻转因子为1（不翻转）
        flip_factors = np.ones(3, dtype=np.float32)
        # 以指定概率触发单轴翻转逻辑
        if np.random.rand() < self.flip_prob:  # 原代码错误：self.flip_factors → self.flip_prob
            # 随机选择一个待处理的轴（x:0, y:1, z:2）
            selected_axis = np.random.choice([0, 1, 2])
            # 对选中轴执行翻转
            flip_factors[selected_axis] = -1.0
        self.flip_factors = flip_factors
        return self.flip_factors
    def apply_flip(self, feat,flip_factors):
        # 检查输入列数
        C = feat.shape[1]
        assert C % 3 == 0, f"输入列数{C}需为3的倍数，当前输入形状：{feat.shape}"
        # 复制数组避免原地修改
        feat_flipped = feat.copy()
        for i in range(0, C, 3):
            feat_flipped[:, i:i+3] *= flip_factors

        return feat_flipped


    def __call__(self, feat: np.ndarray) -> np.ndarray:
        """
        执行点云的随机单轴翻转操作

        Args:
            feat (np.ndarray): 输入数组，形状为(N, C)，C为3的倍数。

        Returns:
            np.ndarray: 翻转后的数组，形状与输入保持一致。

        Raises:
            AssertionError: 输入列数非3的倍数时触发。
        """
        flip_factors=self.get_flip()
        feat_flipped = self.apply_flip(feat,flip_factors)
        return feat_flipped

    def get_info(self):
        """获取翻转因子"""
        if self.flip_factors is None:
            raise RuntimeError("尚未执行翻转操作（请先调用__call__方法）")
        return self.flip_factors


class ScaleXYZ_np:
    def __init__(self, lo=0.8, hi=1.25):
        """
        初始化 Scale 类，用于对**点云位置向量**进行随机各向同性缩放（法线方向向量不缩放）。

        Args:
            lo (float): 缩放因子的下限，默认为 0.8。
            hi (float): 缩放因子的上限，默认为 1.25。

        输入格式要求：
            传入的feat数组形状为(N, C)，其中C为3的倍数，列顺序按「点云(3列)、法线(3列)、点云(3列)、法线(3列)……」循环排列。
            例如：
            - C=3：仅1组点云；
            - C=6：点云(0-2)+法线(3-5)；
            - C=12：点云(0-2)+法线(3-5)+点云(6-8)+法线(9-11)；
            以此类推。
        """
        self.lo = lo
        self.hi = hi
        self.scaler = None  # 初始化缩放因子为None

    def get_scaler(self):
        # 随机生成各向同性缩放因子
        self.scaler = np.random.uniform(self.lo, self.hi)
        return self.scaler

    def apply_scaler(self, feat,scaler):
        N, C = feat.shape
        # 检查输入列数是否为3的倍数
        if C % 3 != 0:
            raise AssertionError(f"输入列数{C}需为3的倍数（按点云、法线循环分组规则），当前输入形状为{feat.shape}")
        # 复制数组避免原地修改输入数据（无副作用设计）
        feat_scaled = feat.copy()

        # 按3列步长遍历，通过i//3计算组索引，判断奇偶组执行缩放
        for i in range(0, C, 3):
            group_idx = i // 3  # 计算当前组索引（0,1,2,...）
            if group_idx % 2 == 0:  # 偶数组为点云，执行缩放
                feat_scaled[:, i:i+3] *= scaler

        return feat_scaled



    def __call__(self, feat):
        """
        执行缩放操作：按分组规则对点云位置向量缩放，法线方向向量保持不变。

        Args:
            feat (np.ndarray): 输入数组，形状为(N, C)，C为3的倍数，符合指定列顺序要求。

        Returns:
            np.ndarray: 缩放后的数组，形状与输入一致。

        Raises:
            AssertionError: 输入列数非3的倍数时触发；输入为非二维数组时触发。
            RuntimeError: 输入数组为空时触发。
        """
        scaler=self.get_scaler()
        return self.apply_scaler(feat,scaler)


    def get_info(self):
        """
        获取本次缩放使用的缩放因子。

        Returns:
            float: 缩放因子scaler。

        Raises:
            RuntimeError: 未执行__call__方法（scaler未初始化）时触发。
        """
        if self.scaler is None:
            raise RuntimeError("尚未执行缩放操作（请先调用__call__方法）")
        return self.scaler

class RotateAxis_np:
    def __init__(self, axis=[0.0, 0.0, 1.0], angle=[0,360]):
        """
        初始化 RotateAxis 类，用于绕指定轴随机旋转点云数据。

        Args:
            axis (list): 旋转轴，形状为 (3,),默认为 [0.0, 0.0, 1.0]（z 轴）。
            angle (list):在 [min, max] 范围内随机生成角度（单位：度）,默认为 [0, 360],

        输入格式要求：
            传入的feat数组形状为(N, C)，其中C为3的倍数，列顺序按「点云(3列)、法线(3列)、点云(3列)、法线(3列)……」循环排列。
            例如：
            - C=3：仅1组点云；
            - C=6：点云(0-2)+法线(3-5)；
            - C=12：点云(0-2)+法线(3-5)+点云(6-8)+法线(9-11)；
            以此类推。
        """
        self.axis = np.array(axis)
        self.angle =angle
        self.params = {}  # 存储归一化参数

    def get_rotation_matrix(self):
        min_angle, max_angle = self.angle
        rotation_angle = np.radians(np.random.uniform(min_angle, max_angle)) # 角度转弧度
        rotation_matrix = get_angle_axis_np(rotation_angle, self.axis)
        self.params["R"]=rotation_matrix
        return rotation_matrix


    def apply_rotation_matrix(self,feat, rotation_matrix):
        C = feat.shape[1]
        assert C % 3 == 0, f"输入列数{C}必须是3的倍数（当前输入形状：{feat.shape}）"
        feat_rot = feat.copy()  # 避免原地修改输入
        for i in range(0, C, 3):
            feat_rot[:, i:i+3] = np.matmul(feat[:, i:i+3], rotation_matrix.T)  # 旋转（正交矩阵R.T=R⁻¹）
        return feat_rot

    def __call__(self, feat):
        rotation_matrix=self.get_rotation_matrix()
        return self.apply_rotation_matrix(feat, rotation_matrix)


    def get_info(self):
        return self.params
class RotateXYZ_np:
    def __init__(self, angle_sigma=2, angle_clip=np.pi):
        """
        用于在三个轴上随机微扰旋转点云数据。

        Args:
            angle_sigma (float): 旋转弧度的高斯分布标准差，默认为 2;
            angle_clip (float): 旋转弧度的裁剪范围，默认为 np.pi。

        输入格式要求：
            传入的feat数组形状为(N, C)，其中C为3的倍数，列顺序按「点云(3列)、法线(3列)、点云(3列)、法线(3列)……」循环排列。
            例如：
            - C=3：仅1组点云；
            - C=6：点云(0-2)+法线(3-5)；
            - C=12：点云(0-2)+法线(3-5)+点云(6-8)+法线(9-11)；
            以此类推。
        """
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip
        self.params = {}  # 存储归一化参数
    def get_rotation_matrix(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        Rx = get_angle_axis_np(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = get_angle_axis_np(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = get_angle_axis_np(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = np.matmul(np.matmul(Rz, Ry), Rx)
        self.params["R"]=rotation_matrix
        return rotation_matrix

    def apply_rotation_matrix(self,feat, rotation_matrix):
        C = feat.shape[1]
        assert C % 3 == 0, f"输入列数{C}必须是3的倍数（当前输入形状：{feat.shape}）"
        # 按3列分块遍历，应用旋转矩阵
        feat_rot = feat.copy()  # 避免原地修改输入
        for i in range(0, C, 3):
            feat_rot[:, i:i+3] = np.matmul(feat[:, i:i+3], rotation_matrix.T)  # 旋转（正交矩阵R.T=R⁻¹）
        return feat_rot



    def __call__(self, feat):
        rotation_matrix=self.get_rotation_matrix()
        return self.apply_rotation_matrix(feat, rotation_matrix)


    def get_info(self):
        return self.params


class Jitter_np(object):
    def __init__(self, std=0.01, clip=0.05):
        """
        用于给点云数据添加随机抖动（仅对点云位置向量添加，法线方向向量不添加）。

        Args:
            std (float): 抖动的高斯分布标准差，默认为 0.01。
            clip (float): 抖动的裁剪范围，默认为 0.05。

        输入格式要求：
            传入的数组形状为(N, C)，C为3的倍数，列顺序按「点云(3列)、法线(3列)、点云(3列)、法线(3列)……」循环排列。
        """
        self.std = std
        self.clip = clip
        self.jittered_data =None  # 存储各点云组的抖动数据

    def get_jitter(self):
        # 生成高斯抖动并裁剪
        jitter = np.random.normal(loc=0.0, scale=self.std, size=(N, 3))
        jitter_clipped = np.clip(jitter, -self.clip, self.clip)
        self.jittered_data=jitter_clipped
        return self.jittered_data
    def apply_jitter(self,feat, jitter_clipped):
        N, C = feat.shape
        if C % 3 != 0:
            raise AssertionError(f"输入列数{C}需为3的倍数，当前输入形状：{feat.shape}")
        # 复制数组避免原地修改
        feat_jittered = feat.copy()
        # 按3列步长遍历组
        for i in range(0, C, 3):
            group_idx = i // 3
            if group_idx % 2 == 0:  # 偶数组：点云，添加抖动
                # 添加抖动
                feat_jittered[:, i:i+3] += jitter_clipped
            # 奇数组：法线，不处理

        return feat_jittered


    def __call__(self, feat):
        """
        执行抖动操作：仅对偶数索引组（点云）添加高斯抖动，奇数索引组（法线）保持不变。

        Args:
            feat (np.ndarray): 输入数组，形状为(N, C)，C为3的倍数。

        Returns:
            np.ndarray: 添加抖动后的数组，形状与输入一致。

        Raises:
            AssertionError: 输入维度非二维/列数非3的倍数时触发；
        """
        jitter= self.get_jitter()
        return self.apply_jitter(feat, jitter)
    def get_info(self):
        """
        获取各点云组的抖动数据。

        Returns:
            list: 按点云组顺序排列的抖动数据列表，每个元素为shape=(N,3)的数组。
        """
        return self.jittered_data


class Translate_np(object):
    def __init__(self, translate_range=0.1):
        """
        用于随机平移点云数据（仅对点云位置向量平移，法线方向向量不平移）。

        Args:
            translate_range (float): 平移范围（x/y/z轴独立在[-translate_range, translate_range]随机），默认为 0.1。

        输入格式要求：
            传入的数组形状为(N, C)，C为3的倍数，列顺序按「点云(3列)、法线(3列)、点云(3列)、法线(3列)……」循环排列。
        """
        self.translate_range = translate_range
        self.translation = None  # 存储三维平移向量（x/y/z轴）

    def get_translation(self):
        # 生成三维平移向量（x/y/z轴独立随机）
        self.translation = np.random.uniform(
            -self.translate_range, self.translate_range, size=3
        ).astype(np.float32)
        return self.translation
    def apply_translation(self,feat, translation):
        # 输入校验
        N, C = feat.shape
        if C % 3 != 0:
            raise AssertionError(f"输入列数{C}需为3的倍数，当前输入形状：{feat.shape}")
        # 复制数组避免原地修改
        feat_translated = feat.copy()
        # 按3列步长遍历组，偶数索引组（点云）执行平移
        for i in range(0, C, 3):
            group_idx = i // 3
            if group_idx % 2 == 0:  # 偶数组：点云，添加平移
                feat_translated[:, i:i+3] += self.translation

        return feat_translated

    def __call__(self, feat):
        """
        执行平移操作：仅对偶数索引组（点云）添加三维随机平移，奇数索引组（法线）保持不变。

        Args:
            feat (np.ndarray): 输入数组，形状为(N, C)，C为3的倍数。

        Returns:
            np.ndarray: 平移后的数组，形状与输入一致。

        Raises:
            AssertionError: 输入维度非二维/列数非3的倍数时触发；
            RuntimeError: 输入数组为空时触发。
        """
        translation = self.get_translation()
        return self.apply_translation(feat, translation)

    def get_info(self):
        """
        获取本次平移的三维向量（x/y/z轴）。

        Returns:
            np.ndarray: 形状为(3,)的平移向量。

        Raises:
            RuntimeError: 未执行平移操作（translation未初始化）时触发。
        """
        if self.translation is None:
            raise RuntimeError("尚未执行平移操作（请先调用__call__方法）")
        return self.translation







class Normalize_np:

    def __init__(self,method="ball"):
        """
        归一化处理类（固定常见区间/统计特征）
        Args:
            method (str): 归一化方法，可选['ball','box','rect','zscore']
                - ball: 单位球归一化（坐标∈[-1,1]，中心在原点，最大模长1）
                - box: 对称包围盒归一化（坐标∈[-1,1]，中心在原点，最大边长2）
                - rect: 矩形归一化（坐标∈[0,1]，最小值对齐0，最大边长1）
                - zscore: Z-Score标准化（均值0，标准差1）

            Note:
               [0,1]-->[-1,1]  执行*2-1;
               [-1,1]-->[0,1]  执行(+1)/2;
        """

        self.method = method.lower()
        self.params = {}  # 存储归一化参数（中心、缩放因子等）

    def __call__(self,points):
        """执行归一化，修改points的前3列（坐标）"""
        vertices = points[:, 0:3].copy()  # 避免原地修改影响计算

        if self.method == "ball":
            # 单位球归一化（常见：中心原点，最大模长1，坐标∈[-1,1]）
            centroid = np.mean(vertices, axis=0)  # 点云中心
            vertices_centered = vertices - centroid
            max_norm = np.max(np.linalg.norm(vertices_centered, axis=1)) + 1e-8  # 最大欧氏距离（防除零）
            scale = 1.0 / max_norm  # 缩放至最大模长1
            vertices_norm = vertices_centered * scale  # 中心原点，模长≤1
            self.params = {"centroid": centroid, "scale": scale}

        elif self.method == "box":
            # 对称包围盒归一化（常见：[-1,1]，中心原点，最大边长2）
            bb_min = vertices.min(axis=0)
            bb_max = vertices.max(axis=0)
            ori_center = (bb_min + bb_max) / 2.0  # 原始包围盒中心
            ori_max_side = (bb_max - bb_min).max() + 1e-8  # 原始最大边长（防除零）
            scale = 2.0 / ori_max_side  # 缩放至最大边长2（对应[-1,1]区间）

            vertices_norm = (vertices - ori_center) * scale  # 中心原点，坐标∈[-1,1]
            self.params = {"ori_center": ori_center, "scale": scale}

        elif self.method == "rect":
            # 矩形归一化（常见：[0,1]，最小值对齐0，最大边长1）
            vmin = vertices.min(axis=0)
            vmax = vertices.max(axis=0)
            ori_max_side = (vmax - vmin).max() + 1e-8  # 原始最大边长（防除零）
            scale = 1.0 / ori_max_side  # 缩放至最大边长1

            vertices_norm = (vertices - vmin) * scale  # 最小值0，最大值1，坐标∈[0,1]
            self.params = {"vmin": vmin, "scale": scale}

        elif self.method == "zscore":
            # Z-Score标准化（常见：均值0，标准差1，无固定坐标区间）
            mean = np.mean(vertices, axis=0)  # 各维度均值
            std = np.std(vertices, axis=0) + 1e-8  # 各维度标准差（防除零）

            vertices_norm = (vertices - mean) / std  # 标准化
            self.params = {"mean": mean, "std": std}

        else:
            raise ValueError(f"不支持的归一化方法: {self.method}，可选['ball','box','rect','zscore']")

        points[:, :3] = vertices_norm
        return points
    def reverse(self, points):
        """逆向还原归一化后的点到原始坐标空间"""
        if not self.params:
            raise RuntimeError("请先执行归一化（__call__方法）再调用逆向还原")

        vertices = points[:, 0:3].copy()

        if self.method == "ball":
            centroid = self.params["centroid"]
            scale = self.params["scale"]
            vertices_ori = vertices / scale + centroid

        elif self.method == "box":
            ori_center = self.params["ori_center"]
            scale = self.params["scale"]
            vertices_ori = vertices / scale + ori_center

        elif self.method == "rect":
            vmin = self.params["vmin"]
            scale = self.params["scale"]
            vertices_ori = vertices / scale + vmin

        elif self.method == "zscore":
            mean = self.params["mean"]
            std = self.params["std"]
            vertices_ori = vertices * std + mean

        else:
            raise ValueError(f"不支持的归一化方法: {self.method}")

        points[:, :3] = vertices_ori
        return points


    
    def get_info(self):
        return {"method": self.method, "params": self.params}



if __name__ =="__main__":
    from torchvision import transforms


    transforms_np = transforms.Compose(
        [
            Normalize_np(method="box"),
            RotateAxis_np(axis=[0,1,0]),
            RotateXYZ_np(angle_sigma=0.05,angle_clip=0.15),
            ScaleXYZ_np(lo=0.8,hi=1.25),
            Translate_np(translate_range=0.1),
            Jitter_np(std=0.01,clip=0.05),
            RandomDropout_np(max_dropout_ratio=0.2),
            FlipXYZ_np(axis_x=False,axis_y=False,axis_z=True),
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