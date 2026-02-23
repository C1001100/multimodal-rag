# src/retrieval/vector_store.py
import os
import json
import faiss
import numpy as np
from typing import Dict, List, Optional, Any

from ..utils.rag_logger import setup_logger
logger = setup_logger(__name__)

class VectorStore:
    """管理所有FAISS索引和数据"""
    
    def __init__(self, path_config):
        self.config = path_config
        self.indexes = {}
        self.metadata = {}
        self.docs = {}
        self._load_all()
    
    def _load_index(self, path: str, name: str):
        """加载单个索引"""
        try:
            if os.path.exists(path):
                self.indexes[name] = faiss.read_index(path)
                logger.info(f"Loaded {name} index with {self.indexes[name].ntotal} vectors")
            else:
                logger.warning(f"Index {path} not found, skipping")
        except Exception as e:
            logger.error(f"Failed to load {name} index: {e}")
    
    def _load_metadata(self, path: str, name: str):
        """加载元数据"""
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.metadata[name] = json.load(f)
                logger.info(f"Loaded {name} metadata")
        except Exception as e:
            logger.error(f"Failed to load {name} metadata: {e}")
    
    def _load_docs(self, dir_path: str, name: str):
        """加载文档"""
        try:
            if os.path.exists(dir_path):
                docs = []
                for f in sorted(os.listdir(dir_path)):
                    path = os.path.join(dir_path, f)
                    with open(path, 'r', encoding='utf-8') as fp:
                        docs.append(fp.read())
                self.docs[name] = docs
                logger.info(f"Loaded {len(docs)} docs from {dir_path}")
        except Exception as e:
            logger.error(f"Failed to load docs from {dir_path}: {e}")
    
    def _load_all(self):
        """加载所有数据"""
        self._load_index(self.config.persona_index, "persona")
        self._load_index(self.config.knowledge_index, "knowledge")
        self._load_index(self.config.image_index, "image")
        self._load_index(self.config.dino_index, "dino")
        self._load_metadata(self.config.image_meta, "image_meta")
        self._load_docs(self.config.persona_dir, "persona_docs")
        self._load_docs(self.config.knowledge_dir, "knowledge_docs")
    
    def search(self, index_name: str, query_vector: np.ndarray, top_k: int = 3):
        """统一搜索接口"""
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} not found")
        
        scores, indices = self.indexes[index_name].search(query_vector, top_k)
        return scores[0], indices[0]
    
    def get_dino_vector(self, idx: int) -> np.ndarray:
        """获取指定索引的DINO向量"""
        if 'dino' not in self.indexes:
            return None
        return self.indexes['dino'].reconstruct(idx).reshape(1, -1)