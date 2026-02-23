# src/model_manager.py
import torch
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import timm
from src.models.dino_wrapper import DINOModelManager
import os
import open_clip
from pathlib import Path
from typing import Union, List

from src.utils.rag_logger import setup_logger
from configs.config import ModelConfig

logger = setup_logger(__name__)

class ModelManager:
    """统一管理所有模型"""
    
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.device = model_config.device
        self._load_models()
    
    def _load_models(self):
        """加载所有模型"""
        try:
            # 1. 文本embedding模型
            logger.info("Loading text embedding model...")
            self.text_embedder = SentenceTransformer(self.config.embed_model_path)
            
            # 2. CLIP模型
            logger.info("Loading CLIP model...")
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                self.config.clip_model_name,
                pretrained=self.config.clip_weight_path
            )
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            self.clip_tokenizer = open_clip.get_tokenizer(self.config.clip_model_name)
            
            # 3. LLM
            logger.info("Loading LLM model...")
            self.llm = Llama(
                model_path=self.config.llm_model_path,
                n_ctx=4096,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                verbose=False
            )
            
            # 4. DINO模型
            logger.info("Loading DINO model...")
            self._load_dino_model()
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_dino_model(self):
        """加载DINO模型"""
        try: 
            self.dino_model = DINOModelManager(
                model_name=self.config.dino_model_name,
                checkpoint_path=self.config.dino_weight_path,
                device=self.device
            )
            logger.info("DINO model loaded successfully from checkpoint")

        except Exception as e:
            logger.error(f"Failed to load DINO model: {e}")
            self.dino_model = None

    def encode_image_dino(self, image):
        """DINO图像编码"""
        if self.dino_model is None:
            logger.warning("DINO model not available")
            return None
        return self.dino_model.encode(image)
    
    def encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """CLIP文本编码"""
        if isinstance(text, str):
            text = [text]
        
        with torch.no_grad():
            text_tokens = self.clip_tokenizer(text).to(self.device)
            features = self.clip_model.encode_text(text_tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype('float32')
    
    def encode_image_clip(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """CLIP图像编码"""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype('float32')
    