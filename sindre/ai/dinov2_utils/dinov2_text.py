import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from typing import List, Union, Tuple, Optional
import requests
from io import BytesIO
import gzip
import os
import ftfy
import regex as re
from functools import lru_cache

# 图像预处理转换 ==============================================================
class GaussianBlur(transforms.RandomApply):
    """应用高斯模糊到PIL图像，支持随机概率和模糊半径范围"""
    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=1-p)

class MaybeToTensor(transforms.ToTensor):
    """智能转换为Tensor，已为Tensor则保持不变"""
    def __call__(self, pic):
        return pic if isinstance(pic, torch.Tensor) else super().__call__(pic)

def make_normalize_transform(
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
) -> transforms.Normalize:
    """创建标准化转换"""
    return transforms.Normalize(mean=mean, std=std)

def make_eval_transform(
    resize_size: int = 256,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC
) -> transforms.Compose:
    """创建评估用预处理流水线"""
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform()
    ])

# 文本Tokenizer ==============================================================
@lru_cache()
def bytes_to_unicode():
    """UTF-8字节到Unicode映射"""
    bs = list(range(33, 126)) + list(range(161, 174)) + list(range(174, 256))
    cs = bs[:]
    return dict(zip(bs, [chr(n) for n in cs]))

class DINOV2Tokenizer:
    """DINOv2专用文本分词器"""
    def __init__(self, vocab_url: str = "https://dl.fbaipublicfiles.com/dinov2/thirdparty/bpe_simple_vocab_16e6.txt.gz"):
        self._load_vocab(vocab_url)
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
    
    def _load_vocab(self, url: str):
        """从URL加载词汇表"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            vocab_data = gzip.decompress(response.content).decode("utf-8").split("\n")
            self._build_vocab(vocab_data[1:49152-256-2+1])
        except Exception as e:
            raise RuntimeError(f"Vocabulary load failed: {e}")
    
    def _build_vocab(self, merges: list):
        """构建词汇表和BPE合并表"""
        vocab = list(bytes_to_unicode().values())
        self.bpe_ranks = dict(zip([tuple(m.split()) for m in merges], range(len(merges))))
        self.encoder = dict(zip(vocab + [v+"</w>" for v in vocab] + ["".join(m) for m in self.bpe_ranks] + ["<|startoftext|>", "<|endoftext|>"], range(len(vocab)*2+len(self.bpe_ranks)+2)))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
    
    def tokenize(self, texts: Union[str, List[str]], context_length: int = 77) -> torch.LongTensor:
        """分词处理，支持单个或多个文本"""
        texts = [texts] if isinstance(texts, str) else texts
        tokens = [[self.encoder["<|startoftext|>"]] + self._encode(text) + [self.encoder["<|endoftext|>"]] for text in texts]
        token_tensor = torch.zeros(len(tokens), context_length, dtype=torch.long)
        
        for i, t in enumerate(tokens):
            if len(t) > context_length:
                t = t[:context_length]
                t[-1] = self.encoder["<|endoftext|>"]
            token_tensor[i, :len(t)] = torch.tensor(t)
        
        return token_tensor

    def _encode(self, text: str) -> list:
        """文本编码核心逻辑"""
        text = re.sub(r'\s+', ' ', ftfy.fix_text(text).strip())
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token_bytes = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe] for bpe in self._bpe(token_bytes).split(' '))
        return bpe_tokens

    def _bpe(self, token: str) -> str:
        """BPE分词处理"""
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}
        
        while pairs:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            word = self._merge_bigram(word, bigram)
            pairs = self._get_pairs(word)
        
        self.cache[token] = ' '.join(word)
        return self.cache[token]

    def _merge_bigram(self, word: tuple, bigram: tuple) -> tuple:
        """合并二元组"""
        new_word, i = [], 0
        while i < len(word):
            try:
                j = word.index(bigram[0], i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break
            
            if i < len(word)-1 and word[i+1] == bigram[1]:
                new_word.append(bigram[0]+bigram[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    @staticmethod
    def _get_pairs(word: tuple) -> set:
        """获取符号对集合"""
        return {(word[i], word[i+1]) for i in range(len(word)-1)}

# DINOv2零样本分类器 ========================================================
class DINOV2ZeroShotClassifier:
    """DINOv2零样本图像分类器"""
    def __init__(self, model_type: str = 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l', device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('facebookresearch/dinov2', model_type).to(self.device).eval()
        self.tokenizer = DINOV2Tokenizer()
        self.transform = make_eval_transform()
    
    def preprocess_image(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """图像预处理"""
        images = [image] if isinstance(image, Image.Image) else image
        return torch.stack([self.transform(img) for img in images]).to(self.device)
    
    def extract_features(self, image_tensor: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取图像和文本特征"""
        text_tokens = self.tokenizer.tokenize(texts).to(self.device)
        
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float):
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)
        
        return F.normalize(image_features, dim=-1), F.normalize(text_features, dim=-1)
    
    def classify(self, image: Union[Image.Image, List[Image.Image]], class_descriptions: List[str]) -> dict:
        """零样本分类"""
        image_tensor = self.preprocess_image(image)
        img_feats, txt_feats = self.extract_features(image_tensor, class_descriptions)
        similarity = (img_feats @ txt_feats.T) * 100  # 缩放相似度
        probs = F.softmax(similarity, dim=-1).cpu()
        
        results = []
        for i in range(probs.shape[0]):
            top_idx = probs[i].argmax()
            predictions = [{"class": desc, "probability": float(prob)} 
                          for desc, prob in zip(class_descriptions, probs[i])]
            results.append({
                "predictions": predictions,
                "top_class": class_descriptions[top_idx],
                "top_probability": float(probs[i][top_idx])
            })
        
        return results[0] if isinstance(image, Image.Image) else results
    
    def get_patch_similarity(self, image: Image.Image, texts: List[str], output_size: Tuple[int, int] = (480, 640)) -> torch.Tensor:
        """获取图像块与文本的相似度"""
        image_tensor = self.preprocess_image(image)
        _, txt_feats = self.extract_features(image_tensor, texts)
        text_patch_feats = txt_feats[:, 1024:]  # 文本特征中与图像块对齐的部分
        
        with torch.no_grad(), torch.autocast(self.device, dtype=torch.float):
            _, patch_tokens = self.model.get_visual_class_and_patch_tokens(image_tensor)
        
        B, P, D = patch_tokens.shape
        H = W = int(P**0.5)
        patch_tokens = patch_tokens.movedim(2, 1).unflatten(2, (H, W)).float()
        patch_tokens = F.interpolate(patch_tokens, size=output_size, mode="bicubic", align_corners=False)
        patch_tokens = F.normalize(patch_tokens, p=2, dim=1)
        text_patch_feats = F.normalize(text_patch_feats.float(), p=2, dim=1)
        
        return torch.einsum("bdhw,cd->bchw", patch_tokens, text_patch_feats)

if __name__ == "__main__":
    import urllib
    from PIL import Image
    
    # 初始化分类器
    classifier = DINOV2ZeroShotClassifier()

    # 加载图像
    with urllib.request.urlopen("https://dl.fbaipublicfiles.com/dinov2/images/example.jpg") as f:
        img = Image.open(f).convert("RGB")
    #img = Image.open("image.jpg").convert("RGB")

    # 零样本分类
    classes = ["photo of a dog", "photo of a cat", "photo of a car"]
    result = classifier.classify(img, classes)
    print(f"Top class: {result['top_class']} ({result['top_probability']:.2%})")

    # 获取图像块相似度
    patch_sim = classifier.get_patch_similarity(img, ["wheel", "window", "headlight"])
    print(patch_sim)