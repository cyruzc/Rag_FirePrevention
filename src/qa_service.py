import logging
import time
from typing import List, Dict, Any
from .models import DocumentResponse
from .cache_manager import QACacheManager
import requests
import json

logger = logging.getLogger(__name__)

class QAService:
    """问答服务 - 使用外部LLM API进行答案生成，集成缓存机制"""
    
    def __init__(self, llm_api_url: str = None, api_key: str = None, enable_cache: bool = True):
        self.llm_api_url = llm_api_url
        self.api_key = api_key
        self.enable_cache = enable_cache
        
        # 初始化缓存管理器
        if enable_cache:
            self.cache_manager = QACacheManager(cache_dir="./cache", ttl=3600)  # 1小时缓存
        else:
            self.cache_manager = None
            
        self._setup_prompt_template()
    
    def _setup_prompt_template(self):
        """设置提示词模板"""
        self.prompt_template = """你是一个专业的火灾预防安全专家。

{context_section}

用户问题：{question}

{answer_guidance}

请给出专业、简洁的回答，使用中文：
"""
    
    def _get_answer_strategy(self, relevant_docs: List[Dict[str, Any]]) -> tuple:
        """根据相关文档确定回答策略"""
        if not relevant_docs:
            return "general", "知识库中没有找到相关文档。", "请基于您的专业知识回答这个问题。"
        
        # 获取最高相似度分数
        max_score = max([doc["score"] for doc in relevant_docs])
        
        if max_score > 0.6:
            # 高置信度：主要基于文档
            context_section = "基于以下相关文档内容：\n" + "\n".join([
                f"- {doc['content'][:300]}..." for doc in relevant_docs[:2]  # 只取前2个最相关文档
            ])
            guidance = "请主要基于提供的文档内容给出专业回答。"
            return "document_based", context_section, guidance
            
        elif max_score > 0.3:
            # 中等置信度：结合文档和通用知识
            context_section = "以下文档可能与问题相关：\n" + "\n".join([
                f"- {doc['content'][:200]}..." for doc in relevant_docs[:1]  # 只取最相关文档
            ])
            guidance = "请结合文档内容和您的专业知识回答。"
            return "hybrid", context_section, guidance
            
        else:
            # 低置信度：主要使用通用知识
            return "general", "知识库中没有找到高度相关的文档。", "请基于您的专业知识回答这个问题。"
    
    def generate_answer(self, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """生成答案 - 使用分级回答策略，集成缓存机制"""
        try:
            # 检查缓存
            if self.enable_cache and self.cache_manager:
                cached_answer = self.cache_manager.get_answer(question, relevant_docs)
                if cached_answer:
                    logger.info(f"从缓存获取答案: {question[:50]}...")
                    return cached_answer
            
            start_time = time.time()
            
            # 确定回答策略
            strategy, context_section, guidance = self._get_answer_strategy(relevant_docs)
            
            # 构建完整提示词
            prompt = self.prompt_template.format(
                context_section=context_section,
                question=question,
                answer_guidance=guidance
            )
            
            # 如果有外部LLM API，使用API生成答案
            if self.llm_api_url and self.api_key:
                answer = self._call_external_llm(prompt)
            else:
                # 如果没有外部API，使用简单的规则生成答案
                answer = self._generate_simple_answer(question, relevant_docs)
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 缓存答案（仅当使用LLM API时才缓存）
            if self.enable_cache and self.cache_manager and self.llm_api_url and self.api_key:
                self.cache_manager.set_answer(question, relevant_docs, answer)
                logger.info(f"答案已缓存: {question[:50]}... (处理时间: {processing_time:.2f}s)")
            
            return answer
                
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return "抱歉，暂时无法回答这个问题。请稍后再试."
    
    def _call_external_llm(self, prompt: str) -> str:
        """调用外部LLM API - 支持DeepSeek和其他兼容OpenAI的API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # DeepSeek模型配置
            data = {
                "model": "deepseek-chat",  # DeepSeek主要模型
                "messages": [
                    {"role": "system", "content": "你是一个专业的火灾预防安全专家。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.3,
                "stream": False
            }
            
            response = requests.post(self.llm_api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            # 兼容不同API响应格式
            if "choices" in result and len(result["choices"]) > 0:
                # OpenAI兼容格式
                return result["choices"][0]["message"]["content"]
            elif "output" in result:
                # 其他API格式
                return result["output"]
            else:
                logger.warning(f"未知的API响应格式: {result}")
                return self._generate_simple_answer_from_prompt(prompt)
            
        except Exception as e:
            logger.error(f"调用外部LLM API失败: {e}")
            return self._generate_simple_answer_from_prompt(prompt)
    
    def _generate_simple_answer(self, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """生成简单答案（备用方案）"""
        if not relevant_docs:
            return "抱歉，在当前的火灾预防知识库中没有找到相关信息。"
        
        # 基于相关文档生成简单回答
        doc_content = relevant_docs[0]["content"]
        
        # 简单的关键词匹配回答
        if "灭火器" in question:
            return "根据文档内容，灭火器应放置在易于取用的位置，定期检查压力表，确保在有效期内使用。"
        elif "逃生" in question or "疏散" in question:
            return "火灾逃生时应保持冷静，用湿毛巾捂住口鼻，低姿前进，按照疏散指示标志撤离。"
        elif "预防" in question:
            return "火灾预防包括定期检查电器线路、不乱扔烟头、不堵塞消防通道、配备灭火器材等措施。"
        elif "报警" in question:
            return "发现火灾应立即拨打119报警，说明详细地址、火势情况和人员被困情况。"
        else:
            # 返回第一个相关文档的摘要
            return f"根据相关文档：{doc_content[:200]}..."
    
    def _generate_simple_answer_from_prompt(self, prompt: str) -> str:
        """从提示词生成简单答案"""
        # 提取问题部分
        if "用户问题：" in prompt:
            question_part = prompt.split("用户问题：")[1].split("\n\n回答要求：")[0].strip()
            
            if "灭火器" in question_part:
                return "灭火器应定期检查，确保压力正常，放置在明显易取的位置。"
            elif "逃生" in question_part or "疏散" in question_part:
                return "火灾逃生时要保持冷静，低姿前进，用湿毛巾捂住口鼻。"
            elif "预防" in question_part:
                return "定期检查电器、不乱扔烟头、保持通道畅通是重要的火灾预防措施。"
            else:
                return "根据相关文档，建议您关注火灾预防的基本知识，包括电器安全、用火管理等。"
        
        return "基于提供的文档内容，建议您遵循标准的火灾预防和安全操作规程。"
