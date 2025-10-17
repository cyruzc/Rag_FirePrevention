from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    """查询请求模型"""
    question: str
    top_k: int = 3

class DocumentResponse(BaseModel):
    """文档响应模型"""
    content: str
    score: float
    metadata: Optional[dict] = None

class QueryResponse(BaseModel):
    """查询响应模型"""
    answer: str
    relevant_documents: List[DocumentResponse]
    question: str

class DocumentUploadRequest(BaseModel):
    """文档上传请求模型"""
    content: str
    title: str
    category: str = "fire_prevention"
    metadata: Optional[dict] = None

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    vector_db_status: str
    model_status: str
