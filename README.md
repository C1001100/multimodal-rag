# MultiModal RAG

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个支持文本和图像的多模态检索增强生成（RAG）系统。能够基于用户问题，从文本知识库和图像库中检索相关信息，并生成回答。

## 📋 目录

- [特性](#特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [待办事项](#待办事项)
- [许可证](#许可证)

## ✨ 特性

- **多模态检索**：支持文本检索文本、文本检索图像、多模态融合检索
- **混合检索策略**：CLIP + DINO 特征融合，提升图像检索准确率
- **模块化设计**：模型管理、向量存储、检索、生成完全解耦
- **缓存机制**：文本embedding缓存，避免重复计算
- **交互式**：目前支持命令行交互模式
- **易于扩展**：可轻松替换或添加新的模型/检索策略

## 🏗 系统架构

``` bash
User Input (Text / Image)
        ↓
Embedding Layer
  - BGE (Text)
  - CLIP (Text & Image)
  - DINO (Image)
        ↓
Vector Store (FAISS)
        ↓
Fusion Ranking
        ↓
Prompt Builder
        ↓
LLM Generator
        ↓
Final Response

``` 

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA 11.0+ (推荐，但CPU也可运行)

### 安装步骤

1. 克隆仓库
    ```bash
    git clone https://github.com/C1001100/multimodal-rag.git
    cd multimodal-rag
    ```

2. 安装依赖


    ``` bash
    pip install -r requirements.txt
    ``` 

3. 下载模型文件

    本项目需要以下模型文件，请自行下载并放置在对应目录：

    - BGE-base-zh-v1.5	
        - 用途：文本embedding	
        - 下载：HuggingFace	
        - 存放路径：models/embedding/bge-base-zh-v1.5/
    - CLIP ViT-B-32	
        - 用途：图像-文本对齐	
        - 下载：HuggingFace		
        - 存放路径：models/clip/clip_vitb32.pt
    - Dino ViT-B-16	
        - 用途：图像-图片	
        - 下载：HuggingFace		
        - 存放路径：models/dino/dino_vitbase16_pretrain.pth
    - Qwen2.5-7B-Instruct (GGUF)	
        - 用途：文本生成	
        - 下载：HuggingFace	
        - 存放路径：models/llm/qwen2.5-7b-instruct-q3_k_m.gguf


4. 准备数据

    将知识文档放在 data/knowledge/ 目录；

    persona文档放在 data/persona/ 目录；

    图像文件信息在 vector_store/image_meta.json 中配置。

    运行脚本预先处理知识库进行本地存储，便于后续读取检索。
    ``` bash
    python scripts/build_index_bge.py
    python scripts/build_index_clip.py
    python scripts/build_index_dino.py
    ``` 

    

5. 运行系统

    路径配置在configs/config.yaml，按本地模型及数据文件情况进行修改。

    ``` bash
    python main.py
    ``` 

## 📌 待办事项

- 添加Web界面 (Gradio/Streamlit)
- 优化缓存策略
- 优化检索策略
- 添加更多检索算法 (BM25, 混合检索)
- 支持批量处理
- 优化推理速度

## 📄 许可证
MIT License

## 🙏 致谢

- sentence-transformers
- FAISS
- OpenCLIP
- llama.cpp
- DINO

如果这个项目对你有帮助，欢迎⭐ Star！