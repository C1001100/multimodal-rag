# src/retrieval/retriever.py
import numpy as np
from typing import List, Dict, Any
from functools import lru_cache
from PIL import Image
from typing import Optional

from ..utils.rag_logger import setup_logger
logger = setup_logger(__name__)

class Retriever:
    """多模态检索器"""
    
    def __init__(self, model_manager, vector_store):
        self.models = model_manager
        self.store = vector_store
    
    @lru_cache(maxsize=128)
    def _cached_text_embed(self, query: str):
        """缓存文本embedding结果"""
        return self.models.text_embedder.encode(
            [query], 
            normalize_embeddings=True,
            show_progress_bar=False
        )
    
    def retrieve_text(self, query: str, index_name: str, top_k: int = 3) -> List[Dict]:
        """文本检索"""
        q_emb = self._cached_text_embed(query)
        scores, indices = self.store.search(index_name, q_emb, top_k)
        
        docs = self.store.docs.get(f"{index_name}_docs", [])
        results = []
        for idx, score in zip(indices, scores):
            if idx < len(docs):
                results.append({
                    'content': docs[idx],
                    'score': float(score),
                    'type': 'text',
                    'index': int(idx)
                })
        return results
    
    def retrieve_image_by_text(self, query: str, top_k: int = 3) -> List[Dict]:
        """文本检索图像"""
        text_features = self.models.encode_text(query)
        scores, indices = self.store.search("image", text_features, top_k)
        
        results = []
        meta = self.store.metadata.get('image_meta', {})
        for idx, score in zip(indices, scores):
            idx_str = str(idx)
            if idx_str in meta:
                item = meta[idx_str].copy()
                item['score'] = float(score)
                item['type'] = 'image'
                results.append(item)
        return results
    

    def retrieve_multimodal_fusion(self, query: str, query_image: Optional[Image.Image] = None, 
                                top_k: int = 3, clip_alpha: float = 0.6) -> List[Dict]:
        """多模态融合检索"""
        
        meta = self.store.metadata.get('image_meta', {})
        
        # 情况1：有查询图像 -> 使用DINO进行图像到图像的检索
        if query_image is not None:
            # 提取查询图像的DINO特征
            query_dino_feat = self.models.encode_image_dino(query_image)
            
            # 在DINO索引中检索相似图像
            dino_scores, dino_ids = self.store.search("dino", query_dino_feat, top_k * 2)
            
            # 构建结果
            results = []
            for idx, dino_score in zip(dino_ids, dino_scores):
                idx_str = str(idx)
                if idx_str in meta:
                    item = meta[idx_str].copy()
                    item.update({
                        'dino_score': float(dino_score),
                        'final_score': float(dino_score),  # 纯DINO分数
                        'type': 'image_by_image',
                        'retrieval_method': 'dino_only'
                    })
                    results.append(item)
            
            # 按分数排序
            results.sort(key=lambda x: x['final_score'], reverse=True)
            return results[:top_k]
        
        # 情况2：只有文本 -> 使用CLIP文本检索 
        # TODO：DINO特征的辅助排序
        else:
            # CLIP文本检索
            text_features = self.models.encode_text(query)
            clip_scores, clip_ids = self.store.search("image", text_features, top_k * 2)
            
            results = []
            
            for idx, clip_score in zip(clip_ids, clip_scores):
                idx = int(idx)
                idx_str = str(idx)
                
                if idx_str not in meta:
                    continue
                
                # TODO:获取该图像的DINO特征（考虑进一步dino重排序）
                # dino_vector = self.store.get_dino_vector(idx)
                dino_score = clip_score
                
                # 融合分数（以CLIP为主）
                final_score = clip_alpha * clip_score + (1 - clip_alpha) * dino_score
                
                item = meta[idx_str].copy()
                item.update({
                    'clip_score': float(clip_score),
                    'dino_score': float(dino_score),
                    'final_score': final_score,
                    'type': 'image_by_text',
                    'retrieval_method': 'clip_with_dino_rerank'
                })
                results.append(item)
            
            results.sort(key=lambda x: x['final_score'], reverse=True)
            return results[:top_k]