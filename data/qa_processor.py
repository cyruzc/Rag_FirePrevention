import json
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class QAProcessor:
    """问答知识库处理器 - 专门处理整理后的高质量问答数据"""
    
    def __init__(self, qa_file_path: str = "data/_docs/docs.json"):
        self.qa_file_path = Path(qa_file_path)
        self.qa_data = []
        
    def load_qa_data(self) -> List[Dict[str, Any]]:
        """加载问答数据"""
        try:
            with open(self.qa_file_path, 'r', encoding='utf-8') as f:
                self.qa_data = json.load(f)
            logger.info(f"成功加载 {len(self.qa_data)} 个问答数据")
            return self.qa_data
        except Exception as e:
            logger.error(f"加载问答数据失败: {e}")
            return []
    
    def convert_to_vector_documents(self) -> List[Dict[str, Any]]:
        """将问答数据转换为向量数据库文档格式"""
        documents = []
        
        for i, qa in enumerate(self.qa_data):
            # 创建向量数据库文档
            document = {
                "content": f"问题：{qa['question']}\n答案：{qa['answer']}",
                "metadata": {
                    "question": qa['question'],
                    "answer": qa['answer'],
                    "category": qa.get('category', '未知'),
                    "type": "qa",
                    "source": "docs.json",
                    "doc_id": f"qa_{i}"
                }
            }
            documents.append(document)
        
        logger.info(f"转换完成：{len(documents)} 个向量文档")
        return documents
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        if not self.qa_data:
            self.load_qa_data()
            
        categories = {}
        for qa in self.qa_data:
            category = qa.get('category', '未知')
            categories[category] = categories.get(category, 0) + 1
            
        return {
            "total_qa": len(self.qa_data),
            "categories": categories,
            "avg_answer_length": sum(len(qa['answer']) for qa in self.qa_data) / len(self.qa_data) if self.qa_data else 0
        }
    
    def search_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按分类搜索问答"""
        if not self.qa_data:
            self.load_qa_data()
            
        return [qa for qa in self.qa_data if qa.get('category') == category]
    
    def get_all_categories(self) -> List[str]:
        """获取所有分类"""
        if not self.qa_data:
            self.load_qa_data()
            
        return list(set(qa.get('category', '未知') for qa in self.qa_data))


def rebuild_vector_db_with_qa():
    """使用问答数据重建向量数据库"""
    try:
        from src.vector_store import VectorStore
        
        # 初始化处理器
        processor = QAProcessor()
        qa_data = processor.load_qa_data()
        
        if not qa_data:
            logger.error("问答数据为空，无法重建向量数据库")
            return False
        
        # 转换为向量文档
        documents = processor.convert_to_vector_documents()
        
        # 重建向量数据库
        vector_store = VectorStore(persist_directory="./chroma_db_gemma")
        
        # 清空现有集合
        try:
            vector_store.client.delete_collection("fire_prevention_docs")
        except:
            pass  # 集合可能不存在
        
        # 重新初始化向量存储
        vector_store._initialize()
        
        # 添加新文档
        vector_store.add_documents(documents)
        
        stats = processor.get_statistics()
        logger.info(f"向量数据库重建成功！包含 {stats['total_qa']} 个问答")
        logger.info(f"分类分布: {stats['categories']}")
        
        return True
        
    except Exception as e:
        logger.error(f"重建向量数据库失败: {e}")
        return False


if __name__ == "__main__":
    # 测试处理器
    processor = QAProcessor()
    stats = processor.get_statistics()
    print(f"问答数据统计: {stats}")
