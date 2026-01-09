from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import List
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from .models import (
    QueryRequest, QueryResponse, DocumentResponse, 
    DocumentUploadRequest, HealthResponse
)
from .vector_store import VectorStore
from .qa_service import QAService

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="火灾预防RAG服务",
    description="基于RAG技术的火灾预防知识问答服务，为Unity数字人提供LLM增强文本服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局服务实例
vector_store = None
qa_service = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化服务"""
    global vector_store, qa_service
    
    try:
        # 验证配置并显示环境状态
        Config.validate_config()
        Config.print_env_status()
        
        # 初始化向量存储
        vector_store = VectorStore()
        
        # 获取DeepSeek配置
        deepseek_config = Config.get_llm_config("deepseek")

        # 初始化问答服务
        qa_service = QAService(
            llm_api_url=deepseek_config["api_url"],
            api_key=deepseek_config["api_key"],
            enable_rag=Config.ENABLE_RAG
        )
        
        logger.info("RAG服务启动成功")
        if deepseek_config["api_key"]:
            logger.info(f"已配置DeepSeek模型: {deepseek_config['model']}")
        else:
            logger.info("未配置LLM API密钥，使用内置规则引擎")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "火灾预防RAG服务运行中",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    try:
        vector_db_healthy = vector_store.health_check() if vector_store else False
        model_healthy = qa_service is not None
        
        status = "healthy" if vector_db_healthy and model_healthy else "unhealthy"
        
        return HealthResponse(
            status=status,
            vector_db_status="healthy" if vector_db_healthy else "unhealthy",
            model_status="healthy" if model_healthy else "unhealthy"
        )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return HealthResponse(
            status="unhealthy",
            vector_db_status="error",
            model_status="error"
        )

@app.post("/query", response_model=QueryResponse)
async def query_knowledge(request: QueryRequest):
    """查询火灾预防知识"""
    try:
        if not qa_service:
            raise HTTPException(status_code=503, detail="服务未就绪")

        relevant_docs = []

        # 只有在RAG启用时才进行向量检索
        if Config.ENABLE_RAG:
            if not vector_store:
                raise HTTPException(status_code=503, detail="向量存储服务未就绪")
            # 搜索相关文档
            relevant_docs = vector_store.search(request.question, request.top_k)

        # 生成答案
        answer = qa_service.generate_answer(request.question, relevant_docs)

        # 格式化文档响应
        doc_responses = [
            DocumentResponse(
                content=doc["content"],
                score=doc["score"],
                metadata=doc.get("metadata", {})
            )
            for doc in relevant_docs
        ]

        return QueryResponse(
            answer=answer,
            relevant_documents=doc_responses,
            question=request.question
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@app.post("/documents")
async def upload_document(request: DocumentUploadRequest):
    """上传火灾预防文档"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="服务未就绪")
        
        # 准备文档数据
        document = {
            "content": request.content,
            "metadata": {
                "title": request.title,
                "category": request.category,
                **(request.metadata or {})
            }
        }
        
        # 添加到向量数据库
        vector_store.add_documents([document])
        
        return {
            "message": "文档上传成功",
            "title": request.title,
            "category": request.category
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文档上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档上传失败: {str(e)}")

@app.get("/documents/info")
async def get_documents_info():
    """获取文档集合信息"""
    try:
        if not vector_store:
            raise HTTPException(status_code=503, detail="服务未就绪")
        
        info = vector_store.get_collection_info()
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取文档信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取文档信息失败: {str(e)}")

@app.get("/examples")
async def get_example_questions():
    """获取示例问题"""
    return {
        "example_questions": [
            "如何使用灭火器？",
            "火灾发生时如何正确逃生？",
            "家庭火灾预防措施有哪些？",
            "办公室火灾应急处理流程是什么？",
            "电器火灾如何预防？"
        ]
    }
