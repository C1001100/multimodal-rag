# RAG主程序
import argparse
import yaml
from typing import Dict, Any
from PIL import Image
from typing import Optional
import os

from configs.config import ModelConfig, RetrievalConfig, PathConfig
from src.models.model_manager import ModelManager
from src.retrieval.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.prompt_builder import PromptBuilder
from src.utils.rag_logger import setup_logger

logger = setup_logger(__name__)

class MultiModalRAGApp:
    """主应用类"""
    
    def __init__(self, model_config: ModelConfig, path_config: PathConfig, 
                 retrieval_config: RetrievalConfig):
        self.model_config = model_config
        self.path_config = path_config
        self.retrieval_config = retrieval_config
        
        logger.info("Initializing MultiModal RAG System...")
        self.models = ModelManager(model_config)
        self.vector_store = VectorStore(path_config)
        self.retriever = Retriever(self.models, self.vector_store)
        self.prompt_builder = PromptBuilder()
    
    def process_query(self, query: str, query_image_path: Optional[str] = None) -> Dict[str, Any]:
        """处理单个查询（支持文本+图像）"""
        try:
            # 如果有查询图像，先加载
            query_image = None
            if query_image_path and os.path.exists(query_image_path):
                query_image = Image.open(query_image_path).convert('RGB')
                logger.info(f"Loaded query image: {query_image_path}")
            
            # 1. 文本检索
            persona_results = self.retriever.retrieve_text(
                query, "persona", self.retrieval_config.top_k
            )
            knowledge_results = self.retriever.retrieve_text(
                query, "knowledge", self.retrieval_config.top_k
            )
            
            # 2. 图像检索
            image_results = self.retriever.retrieve_multimodal_fusion(
                query=query,
                query_image=query_image,
                top_k=self.retrieval_config.top_k,
                clip_alpha=self.retrieval_config.clip_alpha
            )
            
            # 3. 构建prompt
            persona_text = "\n".join([r['content'] for r in persona_results])
            knowledge_text = "\n".join([r['content'] for r in knowledge_results])
            
            prompt = self.prompt_builder.build(
                persona_text=persona_text,
                knowledge_text=knowledge_text,
                images_info=image_results,
                query=query
            )
            logger.info(f"prompt最终为： {prompt}")
            
            # 4. LLM生成
            response = self.models.llm(
                prompt, 
                max_tokens=512, 
                temperature=0.7,
                stop=["</s>", "用户问题："]
            )
            
            answer = response["choices"][0]["text"].strip()
            
            return {
                'answer': answer,
                'persona_used': persona_results,
                'knowledge_used': knowledge_results,
                'images_used': image_results,
                'query_image_used': query_image_path if query_image_path else None,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'answer': f"抱歉，处理您的请求时出现错误：{str(e)}",
                'error': str(e)
            }
    
    def run_interactive(self):
        """交互式运行 - 两步式输入"""
        print("\n" + "="*60)
        print("多模态RAG系统 - 交互模式")
        print("支持图片上传：先输入图片路径，再输入问题")
        print("如果不需要图片，直接输入问题即可")
        print("输入 'quit' 退出")
        print("="*60)
        
        while True:
            # 第一步：是否上传图片
            image_input = input("\n图片路径（直接回车跳过）: ").strip()
            
            query_image_path = None
            if image_input and image_input.lower() not in ['quit', 'exit', 'q']:
                if os.path.exists(image_input) and image_input.lower().endswith(('.png', '.jpg', '.jpeg')):
                    query_image_path = image_input
                    print(f"已加载图片: {os.path.basename(query_image_path)}")
                else:
                    print("图片路径无效或文件不存在，将跳过图片")
            
            # 第二步：输入问题
            query = input("请输入问题: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                print("问题不能为空")
                continue
            
            print("\n正在处理...")
            result = self.process_query(query, query_image_path)
            
            print(f"\n回答: {result['answer']}")
            
            if 'images_used' in result and result['images_used']:
                print(f"\n参考图像: {len(result['images_used'])}张")

def main():
    parser = argparse.ArgumentParser(description="MultiModal RAG System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", 
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    model_config = ModelConfig(**config_dict.get('model', {}))
    retrieval_config = RetrievalConfig(**config_dict.get('retrieval', {}))
    path_config = PathConfig(**config_dict.get('path', {}))
    
    app = MultiModalRAGApp(model_config, path_config, retrieval_config)
    # 进入交互模式
    app.run_interactive()

if __name__ == "__main__":
    main()