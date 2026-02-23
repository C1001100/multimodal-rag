# for clip (Text & Image)
import os
import json
import faiss
import torch
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm

# ==========================
# 配置
# ==========================
IMAGE_DIR = "data/images"
OUTPUT_INDEX = "vector_store/image.index"
OUTPUT_META = "vector_store/image_meta.json"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE="cpu"
MODEL_NAME = "ViT-B-32"
CLIP_WEIGHT_PATH = "models/clip/clip_vitb32.pt"

# ==========================
# 加载模型
# ==========================
print("[INFO] Loading open_clip model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME,
    pretrained=CLIP_WEIGHT_PATH
)


model = model.to(DEVICE)
model.eval()

# ==========================
# 收集图片
# ==========================
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"[INFO] Found {len(image_files)} images.")

embeddings = []
meta = {}

# ==========================
# 编码图片
# ==========================
for idx, filename in enumerate(tqdm(image_files)):
    path = os.path.join(IMAGE_DIR, filename)

    try:
        image = Image.open(path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features = image_features.cpu().numpy()[0]

        embeddings.append(image_features)

        meta[str(idx)] = {
            "path": path,
            "filename": filename
        }

    except Exception as e:
        print(f"[WARN] Failed on {filename}: {e}")

# ==========================
# 构建 FAISS（Cosine → 内积）
# ==========================
embeddings = np.array(embeddings).astype("float32")
dim = embeddings.shape[1]

print("[INFO] Building FAISS index...")
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

# ==========================
# 保存
# ==========================
os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, OUTPUT_INDEX)

with open(OUTPUT_META, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("[INFO] Image index built successfully.")
