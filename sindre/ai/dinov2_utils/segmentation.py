from transformers import AutoImageProcessor, AutoModel

class DinoV2FeatureExtractor:
    def __init__(self, model_name='facebook/dinov2-base'):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.patch_size = self.model.config.patch_size
        self.hidden_size = self.model.config.hidden_size
    
    
    def extract_features(self, image):
        """提取特征并返回字典"""
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
        # 解析特征形状
        batch_size, _, h, w = inputs.pixel_values.shape
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        
        # 分类特征 (CLS token)
        cls_token = last_hidden_states[:, 0, :]
        
        # 分割特征 (重组为空间特征图)
        patch_tokens = last_hidden_states[:, 1:, :]
        patch_features = patch_tokens.unflatten(1, (num_patches_h, num_patches_w))
        
        #第 0 位：[CLS] token（分类特征）
        #第 1-256 位：图像块 token（按光栅扫描顺序排列）
        return {
            "cls_features": cls_token,       # 分类特征 [1, 768]
            "seg_features": patch_features   # 分割特征 [1, 16, 16, 768]
        }

# 使用示例
if __name__ == "__main__":
    extractor = DinoV2FeatureExtractor()
    
    # 支持多种输入类型
    from PIL import Image
    import requests
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    features = extractor.extract_features(image)
    
    print("分类特征形状:", features["cls_features"].shape)  # torch.Size([1, 768])
    print("分割特征形状:", features["seg_features"].shape)  # torch.Size([1, 16, 16, 768])