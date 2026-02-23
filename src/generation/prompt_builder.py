# src/generation/prompt_builder.py
from typing import List, Dict

class PromptBuilder:
    """构建多模态prompt"""
    
    DEFAULT_PERSONA = "你是一个偏技术风格的 AI 助手，擅长多模态理解。"
    
    @classmethod
    def build(cls, persona_text: str, knowledge_text: str, 
              images_info: List[Dict], query: str) -> str:
        """构建完整prompt"""
        
        sections = []
        
        # 1. 系统设定
        sections.append(f"【系统设定】\n{persona_text if persona_text else cls.DEFAULT_PERSONA}\n")
        
        # 2. 知识文本
        if knowledge_text:
            sections.append(f"【参考知识】\n{knowledge_text}\n")
        
        # 3. 图像信息
        if images_info:
            img_desc = []
            for i, img in enumerate(images_info, 1):
                desc = f"图像{i}: {img.get('filename', '未命名')}"
                if 'description' in img:
                    desc += f" - {img['description']}"
                if 'score' in img:
                    desc += f" (相关度: {img['score']:.2f})"
                img_desc.append(desc)
            
            sections.append(f"【相关图像】\n" + "\n".join(img_desc) + "\n")
        
        # 4. 用户查询
        sections.append(f"【用户问题】\n{query}\n")
        
        # 5. 指令
        sections.append("【回答要求】\n请基于以上信息回答问题。")
        
        return "\n".join(sections)