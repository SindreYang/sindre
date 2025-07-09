from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')

class DepthEstimation:
    def __init__(self, model_name="facebook/dpt-dinov2-small-kitti"):
        """
        初始化深度估计模型
        :param model_name: 预训练模型名称
        """
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DPTForDepthEstimation.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded depth estimation model: {model_name}")
        print(f"Using device: {self.device}")
    
    
    def get_feature(self, image):
        """
        提取深度特征
        :param image: 输入图像
        :return: 深度特征张量
        """
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # 返回所有隐藏状态（特征图）
        return outputs.hidden_states
    
    def predict_depth(self, image):
        """
        预测深度图
        :param image: 输入图像
        :return: 深度图(numpy数组), 原始图像(PIL)
        """
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # 插值到原始大小
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],  # (width, height) -> (height, width)
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map = prediction.squeeze().cpu().numpy()
        return depth_map, image
    
    def show_image(self, image, figsize=(12, 8)):
        """
        渲染深度图
        :param image: 输入图像
        :param figsize: 图像大小
        """
        depth_map, orig_image = self.predict_depth(image)
        
        plt.figure(figsize=figsize)
        
        # 原始图像
        plt.subplot(1, 2, 1)
        plt.imshow(orig_image)
        plt.title("Original Image")
        plt.axis("off")
        
        # 深度图
        plt.subplot(1, 2, 2)
        # 归一化并转换为灰度
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        plt.imshow(normalized_depth, cmap="inferno")
        plt.title("Depth Map")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    def show_pcd(self, image):
        """
        使用vedo渲染点云
        :param image: 输入图像
        """
        import vedo
        depth_map, orig_image = self.predict_depth(image)
        img_array = np.array(orig_image)
        
        
        h, w = depth_map.shape
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        
        # 归一化深度
        normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        
        # 创建点云数据
        points = np.column_stack((x.flatten(), y.flatten(), depth_map.flatten()*2))
        colors = img_array.reshape(-1, 3)
        
        # 创建vedo点云
        cloud = vedo.Points(points)
        cloud.pointcolors=colors
        
        vedo.show(cloud, f"Depth Point Cloud (Points: {len(points)})", 
                 axes=1, viewup='z', interactive=True)


# 使用示例
if __name__ == "__main__":
    from PIL import Image
    import requests
    # 初始化深度估计器
    depth_estimator = DepthEstimation()
    
    # 测试图像URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    
    # 提取特征
    features = depth_estimator.get_feature(image)
    print(f"Number of feature layers: {len(features)}")
    print(f"Feature shapes: {[f.shape for f in features]}")
    
    # # 显示深度图
    depth_estimator.show_image(image)
    
    # # 显示点云
    # depth_estimator.show_pcd(image)
    