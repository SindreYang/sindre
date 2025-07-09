import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import torchvision.transforms as T
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class SimilarityDINOV2:
    """
    参考:https://github.com/huggingface/notebooks/blob/main/examples/image_similarity.ipynb
    """
    def __init__(self, model_name='facebook/dinov2-base'):
        # 初始化模型和处理器
        self.transformation_chain = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # 数据存储结构
        self.candidate_images = []
        self.candidate_labels = []
        self.candidate_embeddings = []
        self.candidate_ids = []
        
        # 预处理管道
        # self.transformation_chain = T.Compose([
        #     T.Resize(int((256/224)*self.extractor.crop_size["height"])),
        #     T.CenterCrop(self.extractor.crop_size["height"]),
        #     T.ToTensor(),
        #     T.Normalize(mean=self.extractor.image_mean, std=self.extractor.image_std),
        # ])
        
        # 标签映射
        self.id2label = {}
        self.label2id = {}
    
    def set_label_mapping(self, id2label):
        """设置标签映射关系"""
        self.id2label = id2label
        self.label2id = {v: k for k, v in id2label.items()}
    
    def _compute_embedding(self, image):
        """计算单个图像的嵌入向量"""
        image_tensor = self.transformation_chain(image,return_tensors="pt").to(self.device)#self.transformation_chain(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            #embedding = self.model(pixel_values=image_tensor).last_hidden_state[:, 0].cpu()
            embedding = self.model(**image_tensor).last_hidden_state[:, 0].cpu()
        return embedding
    
    def add(self, image, label=None):
        """
        添加图像到候选集
        :param image: PIL Image 或图像路径
        :param label: 图像标签（可选）
        """
        # 如果是路径则加载图像
        if isinstance(image, str):
            image = Image.open(image)
        
        # 计算嵌入向量
        embedding = self._compute_embedding(image)
        
        # 存储数据
        idx = len(self.candidate_images)
        self.candidate_images.append(image)
        self.candidate_labels.append(label)
        self.candidate_embeddings.append(embedding)
        
        # 生成唯一ID：index_label
        entry_id = f"{idx}_{label}" if label is not None else str(idx)
        self.candidate_ids.append(entry_id)
        
        return idx
    
    def query(self, image, top_k=5):
        """
        查询相似图像
        :param image: PIL Image 或图像路径
        :param top_k: 返回最相似的数量
        :return: (索引列表, 标签列表, 相似度列表)
        """
        if isinstance(image, str):
            image = Image.open(image)
        
        # 计算查询图像的嵌入向量
        query_embedding = self._compute_embedding(image)
        
        # 计算相似度（余弦相似度）
        all_embeddings = torch.cat(self.candidate_embeddings, dim=0)
        scores = torch.nn.functional.cosine_similarity(
            all_embeddings, 
            query_embedding.expand_as(all_embeddings)
        ).numpy()
        
        # 获取top_k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_labels = [self.candidate_labels[i] for i in top_indices]
        top_scores = [scores[i] for i in top_indices]
        
        return top_indices, top_labels, top_scores
    
    def show(self, query_image, top_k=5, query_label=None):
        """
        可视化查询结果
        :param query_image: 查询图像（路径或PIL Image）
        :param top_k: 显示的最相似数量
        :param query_label: 查询图像的标签（可选）
        """
        if isinstance(query_image, str):
            query_image = Image.open(query_image)
        
        # 获取查询结果
        sim_ids, sim_labels, sim_scores = self.query(query_image, top_k)
        
        # 准备展示数据
        images = [query_image]
        labels = [query_label]
        titles = ["Query Image"]
        
        # 添加相似图像
        for i, (idx, label, score) in enumerate(zip(sim_ids, sim_labels, sim_scores)):
            images.append(self.candidate_images[idx])
            labels.append(label)
            titles.append(f"Similar #{i+1}\nScore: {score:.4f}")
        
        # 绘制结果
        plt.figure(figsize=(15, 8))
        columns = min(6, top_k + 1)
        
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = plt.subplot((len(images) + columns - 1) // columns, columns, i + 1)
            
            # 添加标签信息（如果可用）
            if labels[i] is not None and self.id2label:
                label_str = self.id2label.get(labels[i], f"Label {labels[i]}")
                title += f"\nLabel: {label_str}"
            
            ax.set_title(title)
            plt.imshow(img)
            plt.axis("off")
        
        plt.tight_layout()
        plt.show()

# ====================== 使用示例 ====================== #
if __name__ == "__main__":
    from datasets import load_dataset
    # 初始化相似性搜索系统
    similarity_system = SimilarityDINOV2()
    
    # 加载数据集
    dataset = load_dataset("beans")
    labels = dataset["train"].features["labels"].names
    
    # 设置标签映射
    id2label = {i: label for i, label in enumerate(labels)}
    similarity_system.set_label_mapping(id2label)
    
    # 添加候选图像 (使用训练集前100张)
    print("Adding candidate images...")
    for i in tqdm(range(100)):
        img = dataset["train"][i]["image"]
        label = dataset["train"][i]["labels"]
        similarity_system.add(img, label)
    
    # 随机选择测试图像进行查询
    test_idx = np.random.choice(len(dataset["test"]))
    test_sample = dataset["test"][test_idx]["image"]
    test_label = dataset["test"][test_idx]["labels"]
    
    print(f"\nQuerying similar images for test sample #{test_idx}...")
    print(f"True label: {id2label[test_label]}")
    
    # 查询并显示结果
    similarity_system.show(
        query_image=test_sample,
        top_k=5,
        query_label=test_label
    )