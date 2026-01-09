"""
RAG系统配置文件
支持DeepSeek和其他LLM API配置
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

class Config:
    """系统配置类"""

    # DeepSeek API配置
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")

    # 其他LLM配置（备用）
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # RAG功能开关
    ENABLE_RAG: bool = os.getenv("ENABLE_RAG", "false").lower() == "true"

    # 系统配置
    VECTOR_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "fire_prevention_docs"
    
    # API配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # 模型参数
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.3
    TOP_K: int = 3
    
    @classmethod
    def get_llm_config(cls, provider: str = "deepseek") -> dict:
        """获取LLM配置"""
        if provider.lower() == "deepseek":
            return {
                "api_url": cls.DEEPSEEK_API_URL,
                "api_key": cls.DEEPSEEK_API_KEY,
                "model": "deepseek-chat"
            }
        elif provider.lower() == "openai":
            return {
                "api_url": cls.OPENAI_API_URL,
                "api_key": cls.OPENAI_API_KEY,
                "model": "gpt-3.5-turbo"
            }
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")
    
    @classmethod
    def validate_config(cls):
        """验证配置"""
        # 重新加载环境变量以确保获取最新值
        load_dotenv(override=True)
        
        # 更新配置值
        cls.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        cls.ENABLE_RAG = os.getenv("ENABLE_RAG", "false").lower() == "true"

        if not cls.DEEPSEEK_API_KEY and not cls.OPENAI_API_KEY:
            print("⚠️  警告: 未配置任何LLM API密钥，系统将使用内置规则引擎")
            print("💡 提示: 在.env文件中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 来启用LLM功能")
        else:
            if cls.DEEPSEEK_API_KEY:
                print(f"✅ DeepSeek API配置就绪 (密钥长度: {len(cls.DEEPSEEK_API_KEY)})")
            if cls.OPENAI_API_KEY:
                print(f"✅ OpenAI API配置就绪 (密钥长度: {len(cls.OPENAI_API_KEY)})")

        # 显示RAG功能状态
        if cls.ENABLE_RAG:
            print("✅ RAG功能已启用 (使用向量检索增强)")
        else:
            print("ℹ️  RAG功能已禁用 (直接使用LLM)")
    
    @classmethod
    def print_env_status(cls):
        """打印环境变量状态"""
        print("\n🔍 环境变量状态:")
        print(f"   .env文件路径: {os.path.abspath('.env')}")
        print(f"   DEEPSEEK_API_KEY: {'已设置' if cls.DEEPSEEK_API_KEY else '未设置'}")
        print(f"   OPENAI_API_KEY: {'已设置' if cls.OPENAI_API_KEY else '未设置'}")
        print(f"   ENABLE_RAG: {'已启用' if cls.ENABLE_RAG else '已禁用'}")
        if cls.DEEPSEEK_API_KEY:
            print(f"   DeepSeek密钥前10位: {cls.DEEPSEEK_API_KEY[:10]}...")
