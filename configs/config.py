# 配置相关文件路径以及检索参数情况
from dataclasses import dataclass
import torch
import os
from pathlib import Path

@dataclass
class ModelConfig:
    """模型配置"""
    embed_model_path: str = "models/embedding/bge-base-zh-v1.5"
    clip_model_name: str = "ViT-B-32"
    clip_weight_path: str = "models/clip/clip_vitb32.pt"
    dino_model_name: str = "vit_base_patch16_224"
    dino_weight_path: str = "models/dino/dino_vitbase16_pretrain.pth"
    llm_model_path: str = "models/llm/qwen2.5-7b-instruct-q3_k_m.gguf"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从yaml文件加载配置"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get('model', {}))

@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 3
    persona_weight: float = 0.2
    knowledge_weight: float = 0.3
    image_weight: float = 0.5
    clip_alpha: float = 0.6
    
@dataclass
class PathConfig:
    """路径配置"""
    persona_index: str = "vector_store/persona.index"
    knowledge_index: str = "vector_store/knowledge.index"
    image_index: str = "vector_store/image.index"
    dino_index: str = "vector_store/dino.index"
    image_meta: str = "vector_store/image_meta.json"
    persona_dir: str = "data/persona"
    knowledge_dir: str = "data/knowledge"