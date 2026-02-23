# for dino (Image) 每张图片提取DINO特征存入FAISS
import os
import sys
sys.path.append('.')

import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
import torch

from src.models.dino_wrapper import DINOProcessor

def build_dino_index(image_dir: str, output_index: str, checkpoint_path: str):
    """构建DINO索引"""
    
    # 初始化DINO
    dino = DINOProcessor(
        checkpoint_path=checkpoint_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 获取所有图片
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    features = []
    meta = {}
    
    for idx, filename in enumerate(tqdm(image_files, desc="Processing images")):
        path = os.path.join(image_dir, filename)
        
        # 提取特征
        feat = dino.encode_image(path)  # shape: (1, feat_dim)
        features.append(feat[0])
        
        # 保存元数据
        meta[str(idx)] = {
            'path': path,
            'filename': filename
        }
    
    # 创建FAISS索引
    features = np.array(features).astype('float32')
    index = faiss.IndexFlatIP(features.shape[1])  # 内积相似度（等价于余弦相似度，因为已归一化）
    index.add(features)
    
    # 保存
    faiss.write_index(index, output_index)
    
    # 保存元数据
    import json
    with open(output_index.replace('.index', '_meta.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"Index built: {index.ntotal} images")
    print(f"Index saved to: {output_index}")

if __name__ == "__main__":
    build_dino_index(
        image_dir="data/images",
        output_index="vector_store/dino.index",
        checkpoint_path="models/dino/dino_vitbase16_pretrain.pth"
    )