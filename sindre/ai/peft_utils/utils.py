from peft import PeftConfig, LoraConfig, get_peft_model, PeftModel,TaskType
import torch
from typing import Optional, Union, Dict, Any
import os

class LoRAManage:
    """
    通用LoRA注入管理工具
    核心功能：LoRA配置/注入、加载、保存、合并、信息展示
    适配PEFT库，支持任意PyTorch模型（重点适配点云形状编码器）
    
    Note:
        1. TaskType（任务类型）仅支持以下合法值：
           - SEQ_CLS: 文本分类任务
           - SEQ_2_SEQ_LM: 序列到序列语言建模（如翻译、摘要）
           - CAUSAL_LM: 因果语言建模（如GPT类生成任务）
           - TOKEN_CLS: 词元分类（如命名实体识别）
           - QUESTION_ANS: 问答任务
           - FEATURE_EXTRACTION: 特征提取（无分类/回归头，仅输出特征）
        
        
    
    """
    def __init__(
        self,
        model: torch.nn.Module,
        lora_config: Optional[Union[LoraConfig, Dict[str, Any]]] = None,
        freeze_original: bool = True,  # 是否冻结原模型参数（仅训LoRA）
    ):
        """
        初始化LoRA管理器
        Args:
            model: 原始PyTorch模型（如你的VecSetAutoEncoder）
            lora_config: LoRA配置，可传LoraConfig对象或字典（优先级更高）
            freeze_original: 是否冻结原模型所有参数（仅训练LoRA适配器）
            **kwargs: 若未传lora_config，可通过kwargs传LoRA参数（r/lora_alpha/target_modules等）
        """
        # 核心属性初始化
        self.original_model = model  # 原始模型（未注入LoRA）
        self.lora_model: Optional[PeftModel] = None  # 注入LoRA后的模型
        self.lora_config: Optional[LoraConfig] = None  # LoRA配置
        self.device = next(model.parameters()).device if next(model.parameters()).is_cuda else torch.device("cpu")
        
        # 1. 处理LoRA配置（支持传Config对象/字典/kwargs）
        if isinstance(lora_config, LoraConfig):
            self.lora_config = lora_config
        elif isinstance(lora_config, dict):
            self.lora_config = LoraConfig(**lora_config)
        else:
            print("未传入lora_config,停止注入LoRA")
        
        # 冻结原模型参数（可选，适配小数据训练）
        if freeze_original:
            self._freeze_original_model()

        if self.lora_config is not None:
            # 注入LoRA适配器到模型
            self.lora_model = get_peft_model(self.original_model, self.lora_config)

    def get_lora_model(self):
        """获取LoRA适配器"""
        if self.lora_model is None:
            print("LoRA适配器为空")
        return self.lora_model

        
    def _freeze_original_model(self):
        """冻结原模型所有参数（仅LoRA适配器可训练）"""
        for param in self.original_model.parameters():
            param.requires_grad = False
        print(f"已冻结原模型所有参数，仅LoRA适配器可训练")

    def load(self, lora_path: str, strict: bool = True,train=False) -> PeftModel:
        """
        加载预训练的LoRA适配器
        Args:
            lora_path: LoRA适配器保存路径（PEFT格式）
            strict: 是否严格匹配权重（默认True）
            train: 是否开启继续训练模式(默认Fasle)
        Returns:
            加载后的LoRA模型
        """
        if not lora_path:
            raise ValueError("lora_path不能为空")
        if not os.path.exists(lora_path):
            print(f"{lora_path}不存在,加载失败")
            return
        
        try:
            # 加载LoRA配置
            peft_config = PeftConfig.from_pretrained(lora_path)
            # 加载LoRA权重到模型
            self.lora_model = PeftModel.from_pretrained(
                self.original_model,
                lora_path,
                config=peft_config,
                strict=strict,
                device_map=self.device
            )
            if train:
                for name, param in self.lora_model.named_parameters():
                    if "lora" in name.lower():
                        param.requires_grad = True



            print(f"成功从 {lora_path} 加载LoRA适配器,训练模式：{train}")
            return self.lora_model
        except Exception as e:
            raise RuntimeError(f"加载LoRA失败：{str(e)}")

    def save(self, save_path: str, safe_serialization: bool = True) -> None:
        """
        保存LoRA适配器（仅保存LoRA权重，不保存原模型）
        Args:
            save_path: 保存路径
            safe_serialization: 是否安全序列化（推荐True）
        """
        if self.lora_model is None:
            raise ValueError("LoRA模型未初始化，请先完成__init__或load")
        
        try:
            self.lora_model.save_pretrained(
                save_path,
                safe_serialization=safe_serialization
            )
            print(f"成功保存LoRA适配器到 {save_path}")
        except Exception as e:
            raise RuntimeError(f"保存LoRA失败：{str(e)}")

    def merge(self, unload: bool = True, safe_merge: bool = True) -> torch.nn.Module:
        """
        合并LoRA权重到原模型（用于部署，移除PEFT依赖）
        Args:
            unload: 合并后是否卸载LoRA适配器（释放显存）
            safe_merge: 是否安全合并（检查梯度/训练状态）
        Returns:
            合并LoRA权重后的原始模型
        """
        if self.lora_model is None:
            raise ValueError("LoRA模型未初始化，无法合并")
        
        if safe_merge:
            # 安全检查：确保模型处于eval模式，无梯度
            self.lora_model.eval()
            for param in self.lora_model.parameters():
                param.grad = None
        
        try:
            # 合并LoRA权重到原模型
            merged_model = self.lora_model.merge_and_unload()
            if unload:
                self.lora_model = None  # 释放原LoRA模型显存
            print("成功合并LoRA权重到原模型（可直接部署）")
            return merged_model
        except Exception as e:
            raise RuntimeError(f"合并LoRA失败：{str(e)}")

    def __repr__(self) -> str:
        """展示LoRA关键信息（方便调试）"""
        if self.lora_model is None:
            base_info = f"LoRAManage(未初始化LoRA模型, 原模型: {type(self.original_model).__name__})"
        else:
            # 获取可训练参数信息
            trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.lora_model.parameters())
            trainable_ratio = (trainable_params / total_params) * 100
            
            base_info = (
                f"LoRAManage(\n"
                f"  原模型: {type(self.original_model).__name__},\n"
                f"  LoRA配置: r={self.lora_config.r}, alpha={self.lora_config.lora_alpha},\n"
                f"  目标结构={self.lora_config.target_modules},\n"
                f"  可训练参数: {trainable_params:,} ({trainable_ratio:.4f}%),\n"
                f"  总参数: {total_params:,},\n"
                f"  设备: {self.device}\n"
                f")"
            )
        return base_info

    # 获取可训练参数
    def get_trainable_params(self) -> Dict[str, Any]:
        """获取可训练参数统计信息"""
        if self.lora_model is None:
            return {"trainable_params": 0, "total_params": 0, "ratio": 0.0}
        
        trainable_params = sum(p.numel() for p in self.lora_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.lora_model.parameters())
        ratio = (trainable_params / total_params) * 100
        
        return {
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_ratio(%)": ratio
        }