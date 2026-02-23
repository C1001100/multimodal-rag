# BGE (Text)；处理文本文件
from sentence_transformers import SentenceTransformer
import faiss, os

model = SentenceTransformer("models/embedding/bge-base-zh-v1.5")

def build_index(texts, index_path):
    emb = model.encode(texts, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, index_path)

def load_docs(dir):
    docs = []
    for f in os.listdir(dir):
        with open(os.path.join(dir, f), "r", encoding="utf-8") as fp:
            docs.append(fp.read())
    return docs

build_index(load_docs("data/persona"), "vector_store/persona.index")
build_index(load_docs("data/knowledge"), "vector_store/knowledge.index")
