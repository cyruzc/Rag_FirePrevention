import logging
import time
import json
import hashlib
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheManager:
    """缓存管理器 - 用于加速API访问和减少LLM API调用"""
    
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
            ttl: 缓存生存时间（秒），默认1小时
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        self.memory_cache = {}  # 内存缓存
        self._ensure_cache_dir()
        
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        self.cache_dir.mkdir(exist_ok=True)
        
    def _generate_key(self, data: Any) -> str:
        """生成缓存键"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _get_cache_file_path(self, key: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"{key}.json"
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """检查缓存是否有效"""
        if not cache_file.exists():
            return False
            
        # 检查文件修改时间
        file_mtime = cache_file.stat().st_mtime
        current_time = time.time()
        
        return (current_time - file_mtime) < self.ttl
    
    def set(self, key: str, value: Any, use_disk: bool = True):
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            use_disk: 是否使用磁盘缓存
        """
        # 设置内存缓存
        self.memory_cache[key] = {
            'value': value,
            'timestamp': time.time()
        }
        
        # 设置磁盘缓存
        if use_disk:
            cache_file = self._get_cache_file_path(key)
            try:
                cache_data = {
                    'value': value,
                    'timestamp': time.time()
                }
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"写入磁盘缓存失败: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或过期则返回None
        """
        # 首先检查内存缓存
        if key in self.memory_cache:
            cache_data = self.memory_cache[key]
            if time.time() - cache_data['timestamp'] < self.ttl:
                return cache_data['value']
            else:
                # 内存缓存过期，删除
                del self.memory_cache[key]
        
        # 检查磁盘缓存
        cache_file = self._get_cache_file_path(key)
        if self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # 同时更新内存缓存
                self.memory_cache[key] = cache_data
                return cache_data['value']
            except Exception as e:
                logger.warning(f"读取磁盘缓存失败: {e}")
        
        return None
    
    def delete(self, key: str):
        """删除缓存"""
        # 删除内存缓存
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # 删除磁盘缓存
        cache_file = self._get_cache_file_path(key)
        if cache_file.exists():
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"删除磁盘缓存失败: {e}")
    
    def clear(self):
        """清空所有缓存"""
        # 清空内存缓存
        self.memory_cache.clear()
        
        # 清空磁盘缓存
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"清空磁盘缓存失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        memory_count = len(self.memory_cache)
        
        try:
            disk_count = len(list(self.cache_dir.glob("*.json")))
        except:
            disk_count = 0
            
        return {
            'memory_cache_count': memory_count,
            'disk_cache_count': disk_count,
            'ttl': self.ttl,
            'cache_dir': str(self.cache_dir)
        }


class QACacheManager:
    """问答缓存管理器 - 专门用于问答服务的缓存"""
    
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        self.cache_manager = CacheManager(cache_dir, ttl)
        
    def _generate_qa_key(self, question: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """生成问答缓存键"""
        # 基于问题和相关文档生成唯一键
        doc_keys = []
        for doc in relevant_docs:
            doc_content = doc.get('content', '')[:100]  # 只取前100个字符
            doc_score = doc.get('score', 0)
            doc_keys.append(f"{doc_content}:{doc_score:.3f}")
        
        cache_data = {
            'question': question,
            'docs': doc_keys
        }
        
        return self.cache_manager._generate_key(cache_data)
    
    def get_answer(self, question: str, relevant_docs: List[Dict[str, Any]]) -> Optional[str]:
        """获取缓存的答案"""
        key = self._generate_qa_key(question, relevant_docs)
        return self.cache_manager.get(key)
    
    def set_answer(self, question: str, relevant_docs: List[Dict[str, Any]], answer: str):
        """设置答案缓存"""
        key = self._generate_qa_key(question, relevant_docs)
        self.cache_manager.set(key, answer)
    
    def clear_qa_cache(self):
        """清空问答缓存"""
        self.cache_manager.clear()


class VectorCacheManager:
    """向量检索缓存管理器"""
    
    def __init__(self, cache_dir: str = "./cache", ttl: int = 3600):
        self.cache_manager = CacheManager(cache_dir, ttl)
        
    def _generate_vector_key(self, query: str, top_k: int) -> str:
        """生成向量检索缓存键"""
        cache_data = {
            'query': query,
            'top_k': top_k
        }
        return self.cache_manager._generate_key(cache_data)
    
    def get_search_results(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """获取缓存的检索结果"""
        key = self._generate_vector_key(query, top_k)
        return self.cache_manager.get(key)
    
    def set_search_results(self, query: str, top_k: int, results: List[Dict[str, Any]]):
        """设置检索结果缓存"""
        key = self._generate_vector_key(query, top_k)
        self.cache_manager.set(key, results)
    
    def clear_vector_cache(self):
        """清空向量检索缓存"""
        self.cache_manager.clear()
