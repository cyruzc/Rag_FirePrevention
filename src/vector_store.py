import chromadb
import logging
import time
from typing import List, Dict, Any
import chromadb.utils.embedding_functions as embedding_functions
from .cache_manager import VectorCacheManager

logger = logging.getLogger(__name__)

class VectorStore:
    """向量存储管理器 - 使用EmbeddingGemma-300M最佳性能模型，集成缓存机制"""
    
    def __init__(self, persist_directory: str = "./chroma_db_gemma", enable_cache: bool = True):
        self.persist_directory = persist_directory
        self.enable_cache = enable_cache
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        # 初始化缓存管理器
        if enable_cache:
            self.cache_manager = VectorCacheManager(cache_dir="./cache", ttl=7200)  # 2小时缓存
        else:
            self.cache_manager = None
            
        self._initialize()
    
    def _initialize(self):
        """初始化向量数据库和EmbeddingGemma-300M嵌入模型（使用ModelScope）"""
        try:
            # 初始化ChromaDB客户端
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # 使用ModelScope下载并加载EmbeddingGemma-300M模型
            print("🔄 使用ModelScope加载EmbeddingGemma-300M嵌入模型...")
            from modelscope import snapshot_download
            from sentence_transformers import SentenceTransformer
            
            # 下载模型到本地缓存
            model_dir = snapshot_download('google/embeddinggemma-300m')
            print(f"✅ 模型下载完成: {model_dir}")
            
            # 从本地路径加载模型
            self.embedding_model = SentenceTransformer(model_dir)
            
            # 创建自定义嵌入函数
            def gemma_embedding_function(texts):
                return self.embedding_model.encode(texts).tolist()
            
            # 使用自定义嵌入函数创建集合
            embedding_func = embedding_functions.DefaultEmbeddingFunction()
            embedding_func.__call__ = gemma_embedding_function
            
            self.collection = self.client.get_or_create_collection(
                name="fire_prevention_docs",
                embedding_function=embedding_func,
                metadata={
                    "description": "火灾预防知识文档集合",
                    "embedding_model": "EmbeddingGemma-300M (ModelScope)",
                    "dimensions": 1024,
                    "model_source": "modelscope"
                }
            )
            
            logger.info("向量存储初始化成功（使用ModelScope EmbeddingGemma-300M嵌入）")
            logger.info(f"嵌入模型: EmbeddingGemma-300M (1024维) - ModelScope")
            
        except Exception as e:
            logger.error(f"向量存储初始化失败: {e}")
            # 如果Gemma模型加载失败，回退到默认嵌入
            print(f"⚠️ EmbeddingGemma-300M模型加载失败，回退到默认嵌入: {e}")
            self._fallback_to_default()
    
    def _fallback_to_default(self):
        """回退到默认嵌入模型"""
        try:
            print("🔄 回退到默认嵌入模型...")
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(
                name="fire_prevention_docs",
                metadata={
                    "description": "火灾预防知识文档集合",
                    "embedding_model": "ChromaDB Default",
                    "dimensions": "unknown"
                }
            )
            logger.info("向量存储回退到默认嵌入模型成功")
            logger.info("嵌入模型: ChromaDB Default")
            
        except Exception as e:
            logger.error(f"回退到默认嵌入失败: {e}")
            raise
    
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """添加文档到向量数据库"""
        try:
            if not documents:
                return
            
            # 提取文档内容
            contents = [doc["content"] for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            ids = [f"doc_{i}" for i in range(len(documents))]
            
            # 使用ChromaDB自动处理嵌入
            self.collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """搜索相关文档 - 集成缓存机制"""
        try:
            # 检查缓存
            if self.enable_cache and self.cache_manager:
                cached_results = self.cache_manager.get_search_results(query, top_k)
                if cached_results:
                    logger.info(f"从缓存获取检索结果: {query[:50]}...")
                    return cached_results
            
            start_time = time.time()
            
            # 使用ChromaDB内置查询功能
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            # 格式化结果
            documents = []
            if results['documents']:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0], 
                    results['metadatas'][0], 
                    results['distances'][0]
                )):
                    documents.append({
                        "content": doc,
                        "metadata": metadata or {},
                        "score": 1 - distance  # 转换为相似度分数
                    })
            
            # 计算检索时间
            search_time = time.time() - start_time
            
            # 缓存结果
            if self.enable_cache and self.cache_manager:
                self.cache_manager.set_search_results(query, top_k, documents)
                logger.info(f"检索结果已缓存: {query[:50]}... (检索时间: {search_time:.3f}s)")
            
            return documents
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        try:
            count = self.collection.count()
            return {
                "collection_name": "fire_prevention_docs",
                "document_count": count,
                "status": "healthy"
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {"status": "error", "error": str(e)}
    
    def health_check(self) -> bool:
        """健康检查"""
        try:
            self.collection.count()
            return True
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False
