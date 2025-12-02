
import numpy as np
from PIL.Image import Image


class SindreImage:
    """OpenCV、PIL、matplotlib、skimage统一处理库
    
    支持图像模式:
    - 1: 1位像素，黑白图像
    - L: 8位灰度图
    - RGB: 24位真彩色
    - RGBA: 带透明通道
    - CMYK, YCbCr, LAB等
    
    支持输入类型:
    - 文件路径(.png, .jpg, .si, .simage)
    - PIL图像
    - numpy数组
    - OpenCV图像
    - base64编码字符串
    - base64编码字节
    
    统一规范:
    1. 内部存储: numpy(uint8)数组，HWC格式，RGB颜色空间
    2. 以numpy为核心处理引擎， matplotlib用于渲染，其他特殊方法用OpenCV、matplotlib、skimage快速高效实现；
    
    Attributes:
        image_array (np.ndarray): 图像数据(HWC, uint8, RGB)
        image_labels (np.ndarray): 像素标签数组(HW)
        image_mode (str): 图像模式
        image_labels2text (dict): 标签ID到文本描述的映射
    """
    
    def __init__(self, any_obj = None) -> None:
        self.image_array = None
        self.image_labels = None
        self.image_mode = None
        self.image_labels2text = {}
        
        if any_obj is not None:
            self._update(any_obj)
    
    def _update(self, any_obj) -> None:
        """转换输入对象为统一格式
        
        Args:
            any_obj: 输入对象，可以是文件路径、PIL图像、numpy数组等
            
        Raises:
            ValueError: 当输入类型不支持时
        """
        # Base64字节处理
        if isinstance(any_obj, bytes):
            if any_obj.startswith(b'\x89PNG') or any_obj.startswith(b'\xff\xd8'):
                # 原始图像字节
                self._from_bytes(any_obj)
            else:
                # Base64编码字节
                self._from_base64(any_obj)
            return
        
        # 字符串处理
        if isinstance(any_obj, str):
            # Base64字符串
            if any_obj.startswith("data:image/") or (len(any_obj) > 100):
                self._from_base64(any_obj)
            # 自定义格式
            elif any_obj.endswith((".si", ".simage")):
                self.load(any_obj)
            # 文件路径
            else:
                self._from_file(any_obj)
            return
        
        # PIL图像处理
        if 'Image' in globals() and isinstance(any_obj, Image.Image):
            self._from_pil(any_obj)
            return
        
        # numpy数组处理
        if isinstance(any_obj, np.ndarray):
            self._from_numpy(any_obj)
            return
        
        # 不支持的类型
        raise ValueError(f"不支持的类型: {type(any_obj)}")
    
    def _from_file(self, file_path: str) -> None:
        """从文件路径加载图像
        
        Args:
            file_path: 图像文件路径
        """
        from PIL import Image
        pil_img = Image.open(file_path)
        self._from_pil(pil_img)
    
    def _from_pil(self, pil_img) -> None:
        """从PIL图像初始化
        
        Args:
            pil_img: PIL图像对象
        """
        self.image_mode = pil_img.mode
        
        # 特殊处理1位图像
        if self.image_mode == '1':
            self.image_array = np.array(pil_img.convert('L'))
            self.image_array = np.stack([self.image_array]*3, axis=-1)
        # 处理调色板图像
        elif self.image_mode == 'P':
            self.image_array = np.array(pil_img.convert('RGB'))
            self.image_mode = 'RGB'
        # 其他模式
        else:
            self.image_array = np.array(pil_img)
        
        # 确保3通道
        if len(self.image_array.shape) == 2:
            self.image_array = np.stack([self.image_array]*3, axis=-1)
    
    def _from_numpy(self, arr) -> None:
        """从numpy数组初始化
        
        Args:
            arr: numpy数组形式的图像数据
        """
        # 处理1位图像
        if arr.dtype == bool or (arr.dtype == np.uint8 and np.max(arr) <= 1):
            arr = arr.astype(np.uint8) * 255
            self.image_array = np.stack([arr]*3, axis=-1)
            self.image_mode = '1'
        # 处理灰度图
        elif len(arr.shape) == 2:
            self.image_array = np.stack([arr]*3, axis=-1)
            self.image_mode = 'L'
        # 处理BGR图像
        elif arr.shape[2] == 3:
            import cv2
            self.image_array = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            self.image_mode = 'RGB'
        # 处理其他通道图像
        else:
            self.image_array = arr.copy()
            self.image_mode = self._get_mode(self.image_array)
    
    def _from_bytes(self, image_bytes) -> None:
        """从原始图像字节初始化
        
        Args:
            image_bytes: 原始图像字节数据
        """
        from PIL import Image
        from io import BytesIO
        pil_img = Image.open(BytesIO(image_bytes))
        self._from_pil(pil_img)
    
    def _from_base64(self, base64_data) -> None:
        """从base64数据初始化
        
        Args:
            base64_data: base64编码的图像数据
        """
        import base64
        from io import BytesIO
        
        # 如果是字符串且包含前缀
        if isinstance(base64_data, str):
            if base64_data.startswith("data:image/"):
                # 移除前缀
                header, base64_data = base64_data.split(",", 1)
            # 解码base64字符串
            image_bytes = base64.b64decode(base64_data)
        else:
            # 直接解码base64字节
            image_bytes = base64.b64decode(base64_data)
        
        self._from_bytes(image_bytes)
    
    @staticmethod
    def _get_mode(arr) -> str:
        """根据数组形状确定图像模式
        
        Args:
            arr: numpy数组形式的图像数据
            
        Returns:
            str: 图像模式字符串
        """
        if arr.ndim == 2:
            return 'L'
        channels = arr.shape[2] if arr.ndim > 2 else 1
        return {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(channels, 'UNKNOWN')
    
    @property
    def to_pil(self) -> 'Image.Image':
        """转换为PIL图像对象
        
        Returns:
            Image.Image: PIL图像对象
        """
        from PIL import Image
        if self.image_mode == '1':
            arr = self.image_array[:, :, 0]  # 取单通道
            return Image.fromarray(arr).convert('1')
        return Image.fromarray(self.image_array)
    
    @property
    def to_cv2(self) -> np.ndarray:
        """转换为OpenCV格式(BGR)
        
        Returns:
            np.ndarray: OpenCV格式的图像数组
        """
        import cv2
        if self.image_mode == '1':
            arr = self.image_array[:, :, 0].astype(np.uint8)
            return np.where(arr > 127, 255, 0)
        return cv2.cvtColor(self.image_array, cv2.COLOR_RGB2BGR)
    
    @property
    def to_dict(self) -> dict:
        """转换为字典格式
        
        Returns:
            dict: 包含图像数据的字典
        """

        return {
                'image_array': self.image_array,
                'image_labels': self.image_labels,
                'image_mode': self.image_mode,
                'image_labels2text': self.image_labels2text
            }
       
    
    def to_base64(self, format: str = "PNG", include_prefix: bool = True) -> str:
        """转换为base64编码字符串
        
        Args:
            format: 图像格式 (PNG, JPEG等)
            include_prefix: 是否包含data URI前缀
            
        Returns:
            str: base64编码的图像字符串
        """
        import base64
        from io import BytesIO
        
        img = self.to_pil
        buffered = BytesIO()
        img.save(buffered, format=format)
        img_bytes = buffered.getvalue()
        
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        
        if include_prefix:
            mime_type = f"image/{format.lower()}"
            return f"data:{mime_type};base64,{base64_str}"
        return base64_str
    
    def to_bytes(self, format: str = "PNG") -> bytes:
        """转换为图像字节
        
        Args:
            format: 图像格式 (PNG, JPEG等)
            
        Returns:
            bytes: 原始图像字节
        """
        from io import BytesIO
        
        img = self.to_pil
        buffered = BytesIO()
        img.save(buffered, format=format)
        return buffered.getvalue()
    
    @property
    def shape(self) -> tuple:
        """获取图像形状
        
        Returns:
            tuple: 图像形状 (高度, 宽度, 通道)
        """
        return self.image_array.shape
    
    @property
    def mode(self) -> str:
        """获取图像模式
        
        Returns:
            str: 图像模式
        """
        return self.image_mode
    
    def split(self):
        """分离图像通道为numpy数组
        
        Returns:
            List[np.ndarray]: 通道分离后的数组列表
        """
        return [self.image_array[:, :, i].copy() for i in range(self.image_array.shape[2])]
    
    def merge(self, channels) :
        """合并通道生成新图像
        
        Args:
            channels: 通道列表
            
        Returns:
            SindreImage: 合并后的新图像对象
        """
        merged = np.stack(channels, axis=-1)
        return SindreImage(merged)
    
    def save(self, out_path: str) -> None:
        """保存图像到文件
        
        Args:
            out_path: 输出文件路径
        """
        if out_path.endswith(('.si', '.simage')):
            self._save_custom(out_path)
        else:
            self.to_pil.save(out_path)
    
    def resize(self, size: tuple, resample='bilinear') -> 'SindreImage':
        """调整图像尺寸
        
        Args:
            size: 目标尺寸 (宽度, 高度)
            resample: 重采样方法 ('nearest', 'bilinear', 'bicubic')
            
        Returns:
            SindreImage: 调整尺寸后的新图像
        """
        from PIL import Image
        
        # 转换重采样方法字符串为PIL常量
        resample_methods = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC
        }
        pil_resample = resample_methods.get(resample.lower(), Image.BILINEAR)
        
        pil_img = self.to_pil.resize(size, pil_resample)
        return SindreImage(pil_img)
    
    def convert(self, mode: str) -> 'SindreImage':
        """转换颜色空间
        
        Args:
            mode: 目标颜色模式
            
        Returns:
            SindreImage: 转换后的新图像
        """
        pil_img = self.to_pil.convert(mode)
        return SindreImage(pil_img)
    
    def crop(self, box: tuple) -> 'SindreImage':
        """裁剪图像区域
        
        Args:
            box: 裁剪区域 (左, 上, 右, 下)
            
        Returns:
            SindreImage: 裁剪后的新图像
        """
        pil_img = self.to_pil.crop(box)
        return SindreImage(pil_img)
    
    def paste(self, img: 'SindreImage', box: tuple) -> None:
        """粘贴图像到当前图像
        
        Args:
            img: 要粘贴的图像
            box: 粘贴位置 (左, 上, 右, 下)
        """
        from PIL import Image
        pil_self = self.to_pil
        pil_self.paste(img.to_pil, box)
        self.image_array = np.array(pil_self)
    
    def segmentation(self) :
        """根据标签分割图像为numpy数组
        
        Returns:
            Dict[int, np.ndarray]: 标签ID到图像片段的映射
        """
        if self.image_labels is None:
            return {}
            
        segments = {}
        for label_id in np.unique(self.image_labels):
            mask = (self.image_labels == label_id).astype(np.uint8)
            segments[label_id] = mask
        return segments
    
    def get_mask(self, labels = None):
        """生成指定标签的掩码(numpy数组)
        
        Args:
            labels: 标签ID或标签列表
            
        Returns:
            Dict: 标签ID到掩码的映射
        """
        if self.image_labels is None:
            return {}
            
        if isinstance(labels, int):
            return (self.image_labels == labels).astype(np.uint8)
        
        masks = {}
        target_labels = labels if labels else np.unique(self.image_labels)
        
        for label in target_labels:
            masks[str(label)] = (self.image_labels == label).astype(np.uint8)
       
        return masks
    
    def get_boundary(self, label: int, simplify_tolerance: float = 1):
        """获取标签边界的有序点集
        
        Args:
            label: 目标标签ID
            simplify_tolerance: 轮廓简化阈值
            
        Returns:
            list: 有序边界点列表 [轮廓1, 轮廓2, ...]
        """
        import cv2
        
        if self.image_labels is None:
            return []
            
        mask = (self.image_labels == label).astype(np.uint8)*255
        if np.max(mask) == 0:  # 没有该标签
            return []
        
        contours, _ = cv2.findContours(
            mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 简化和转换格式
        simplified_contours = []
        for contour in contours:
            # 简化轮廓
            if simplify_tolerance > 0:
                contour = cv2.approxPolyDP(contour, simplify_tolerance, True)
            # 转换为 (N,2) 格式
            contour = contour.reshape(-1,2)
            # 闭合
            closed_contour = np.vstack([contour, contour[0]])
            simplified_contours.append(closed_contour)
        
        return simplified_contours
    
    def set_labels2text(self, label: int, name: str) -> None:
        """设置标签对应的文本描述
        
        Args:
            label: 标签ID
            name: 文本描述
        """
        self.image_labels2text[label] = name
    
    def set_labels_by_boundary(self, boundary_points, label: int) -> None:
        """通过边界有序点设置标签
        
        Args:
            boundary_points: 有序边界点列表轮廓[[100,100],[200,200]]
            label: 要设置的标签ID
            
        Raises:
            AssertionError: 如果尝试设置背景标签(0)
        """
        import cv2
        assert label != 0, "图像背景默认为0，请不要设置为0"
        
        if self.image_labels is None:
            h, w = self.image_array.shape[:2]
            self.image_labels = np.zeros((h, w), dtype=np.int32)
        
        # 确保是整数坐标
        boundary_points = np.array(boundary_points).astype(np.int32)
        
        # 创建掩码并填充
        mask = np.zeros_like(self.image_labels, dtype=np.uint8)
        
        # 确保多边形是闭合的
        if not np.array_equal(boundary_points[0], boundary_points[-1]):
            boundary_points = np.vstack([boundary_points, boundary_points[0]])
        
        # 填充多边形
        cv2.fillPoly(mask, [boundary_points], color=255)
        
        # 更新标签
        self.image_labels[mask > 0] = label
    
    def set_labels_by_mask(self, mask, label: int) -> None:
        """通过掩码设置标签
        
        Args:
            mask: 掩码数组
            label: 要设置的标签ID
            
        Raises:
            AssertionError: 如果尝试设置背景标签(0)
        """
        assert label != 0, "图像背景默认为0，请不要设置为0"
        if self.image_labels is None:
            self.image_labels = np.zeros_like(mask, dtype=np.int32)
        self.image_labels[mask > 0] = label
    
    def _display_annotated_image(self) -> None:
        """显示带有标注的图像"""
        import matplotlib.pyplot as plt
        from sindre.utils3d import labels2colors
        
        annotated_img = self.image_array.copy()
        
        # 获取所有标签ID (跳过背景0)
        unique_labels = [l for l in np.unique(self.image_labels) if l != 0]
        if not unique_labels:
            plt.imshow(annotated_img)
            plt.axis('off')
            return
        
        # 创建颜色映射
        cmap = labels2colors(np.array(unique_labels))
        
        plt.imshow(annotated_img)
        
        # 创建图例元素
        legend_elements = []
        
        # 绘制每个标签的边界
        for i, label in enumerate(unique_labels):
            boundaries = self.get_boundary(label, simplify_tolerance=1.0)
            color = cmap[i][:3] / 255.0  # 转换为0-1范围
            
            for boundary in boundaries:
                if len(boundary) < 3:  # 需要至少3个点形成多边形
                    continue
                    
                # 闭合多边形
                closed_boundary = np.vstack([boundary, boundary[0]])
                plt.plot(closed_boundary[:, 0], closed_boundary[:, 1], 
                        color=color, linewidth=2, linestyle='-')

            label_text = self.image_labels2text.get(label, f"Label {label}")
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=label_text))
        
        # 添加图例 (放在图像上方)
        plt.legend(handles=legend_elements, loc='upper center', 
                bbox_to_anchor=(0.5, -0.05),
                ncol=min(4, len(unique_labels)))
        
        plt.axis('off')

    def show(self, others = None) -> None:
        """使用matplotlib显示图像
        
        Args:
            others: 其他要显示的图像列表
        """
        import matplotlib.pyplot as plt
        
        # 计算需要显示的子图数量
        num_images = len(others) + 1 if others else 1
        if self.image_labels is not None:
            num_images += 1
            
        # 创建画布
        plt.figure(figsize=(6 * num_images, 6))
        
        # 显示原始图像
        plt.subplot(1, num_images, 1)
        if self.mode == '1':
            plt.imshow(self.image_array[:, :, 0], cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(self.image_array)
        plt.axis('off')
        plt.title("Original Image")
        
        # 如果有标签，显示标注图像
        if self.image_labels is not None:
            plt.subplot(1, num_images, 2)
            self._display_annotated_image()
            plt.title("Annotated Image")
        
        # 显示其他图像
        if others:
            start_idx = 3 if self.image_labels is not None else 2
            for i, img in enumerate(others, start_idx):
                plt.subplot(1, num_images, i)
                
                if isinstance(img, np.ndarray) and img.ndim == 2 and img.shape[1] == 2:
                    # 绘制线图
                    plt.imshow(self.image_array.copy())#载入图像，避免因为图像坐标系与笛卡尔坐标系原因进行y轴反转
                    plt.plot(img[:, 0], img[:, 1], 'ro-')
                    plt.title(f"Others Boundary {i}")
                elif isinstance(img, np.ndarray) and img.ndim == 2:
                    # 显示灰度图
                    plt.imshow(img, cmap='gray')
                    plt.title(f"Others Mask {i}")
                else:
                    # 显示彩色图像
                    plt.imshow(img)
                    plt.title(f"Others Image {i}")
                
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def _save_custom(self, path: str) -> None:
        """保存为自定义.si格式
        
        Args:
            path: 文件保存路径
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.to_dict, f)
    
    def load(self, path: str) -> None:
        """从自定义.si格式加载
        
        Args:
            path: 文件加载路径
        """
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.image_array = data['image_array']
            self.image_labels = data['image_labels'] if 'image_labels' in data else None
            self.image_mode = data['image_mode']
            self.image_labels2text = data.get('image_labels2text', {})
    
    def __repr__(self) -> str:
        """对象表示字符串
        
        Returns:
            str: 对象的描述字符串
        """
        return (f"SindreImage(shape={self.shape}, mode='{self.mode}', "
                f"labels={self.image_labels is not None})")
        
    def refine_labels_with_de(self, label: int) -> None:
        """通过膨胀腐蚀优化特定标签边界
        
        Args:
            label: 要优化的标签ID
            
        Raises:
            AssertionError: 如果没有标签或标签为背景
        """
        from scipy.ndimage import binary_dilation, binary_erosion
        
        assert self.image_labels is not None, "无标签"
        assert label != 0, "无效输入，标签为背景"
        
        # 创建边界区域（标签区域的边界）
        boundary_mask = np.zeros_like(self.image_labels, dtype=bool)
        label_mask = (self.image_labels == label)
        
        # 膨胀和腐蚀以获取边界
        dilated = binary_dilation(label_mask, iterations=3)
        eroded = binary_erosion(label_mask, iterations=3)
        boundary = dilated & ~eroded
        boundary_mask |= boundary
        
        # 更新标签
        self.image_labels[boundary_mask > 0] = label

    def refine_labels_with_random_walker(self, mask) :
        """使用随机游走算法优化标签
        
        Args:
            mask: 初始掩码
            
        Returns:
            np.ndarray: 优化后的标签数组
        """
        from skimage.color import rgb2gray
        from skimage.segmentation import random_walker
        
        gray_img = rgb2gray(self.image_array)
        optimized_labels = random_walker(gray_img, mask + 1, mode='bf')
        return optimized_labels


