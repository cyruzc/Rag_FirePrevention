#!/usr/bin/env python3
"""
火灾预防RAG服务主程序
为Unity数字人提供LLM增强文本服务
使用整理后的高质量问答知识库
"""

import uvicorn
import logging
from src.api import app
from data.qa_processor import rebuild_vector_db_with_qa

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_knowledge_base():
    """初始化知识库，使用问答数据重建向量数据库"""
    try:
        logger.info("正在初始化火灾预防问答知识库...")
        
        # 使用问答数据重建向量数据库
        success = rebuild_vector_db_with_qa()
        
        if not success:
            raise Exception("问答知识库初始化失败")
            
        logger.info("问答知识库初始化成功")
        return True
        
    except Exception as e:
        logger.error(f"知识库初始化失败: {e}")
        raise

def main():
    """主函数"""
    try:
        # 初始化知识库
        success = initialize_knowledge_base()
        
        if not success:
            raise Exception("知识库初始化失败")
        
        # 启动FastAPI服务
        logger.info("启动火灾预防RAG服务...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            reload=False  # 生产环境建议关闭热重载
        )
        
    except KeyboardInterrupt:
        logger.info("服务被用户中断")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

if __name__ == "__main__":
    main()
