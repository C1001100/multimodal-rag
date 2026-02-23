# src/models/dino_wrapper.py
import timm
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Union, List, Optional
from pathlib import Path

class DINOProcessor:
    """DINO图像处理器"""
    
    def __init__(self, model_name: str = "vit_base_patch16_224", 
                 checkpoint_path: Optional[str] = None,
                 device: str = "cuda"):
        self.device = device
        
        # 创建模型
        self.model = timm.create_model(model_name, pretrained=False)
        
        # 加载权重
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"Loading DINO checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # DINO权重处理
            if "teacher" in checkpoint:
                state_dict = checkpoint["teacher"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            
            # 移除多余前缀
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                k = k.replace("backbone.", "")
                new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            print("Checkpoint loaded successfully")
        else:
            print("No checkpoint provided, using random initialization")
        
        self.model.eval().to(device)
        
        # 创建transform（使用timm的标准transform）
        self.transform = timm.data.create_transform(
            input_size=224,
            is_training=False,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    
    @torch.no_grad()
    def encode_image(self, image: Union[str, Path, Image.Image, List[Image.Image]]) -> np.ndarray:
        """
        提取图像的DINO特征
        
        Args:
            image: 单张或多张图片
            
        Returns:
            numpy array of features, shape (n_images, feat_dim)
        """
        # 统一处理为列表
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
            images = [image]
        elif isinstance(image, Image.Image):
            images = [image]
        else:
            images = image
        
        batch = []
        for img in images:
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert('RGB')
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # 预处理
            img_tensor = self.transform(img)
            batch.append(img_tensor)
        
        # 堆叠成batch
        batch = torch.stack(batch).to(self.device)
        
        # 前向传播
        features = self.model(batch)  # shape: (batch_size, feat_dim)
        
        # L2归一化
        features = F.normalize(features, dim=-1)
        
        return features.cpu().numpy().astype('float32')
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算两个特征的余弦相似度"""
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)
        return float(np.dot(feat1.flatten(), feat2.flatten()))


class DINOModelManager:
    """DINO模型管理器（作为ModelManager的组件）"""
    
    def __init__(self, model_name: str,checkpoint_path: str, device: str = "cuda"):
        self.processor = DINOProcessor(
            model_name,
            checkpoint_path=checkpoint_path,
            device=device
        )
    
    def encode(self, images):
        return self.processor.encode_image(images)