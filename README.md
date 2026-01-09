# 火灾预防RAG服务

基于RAG（Retrieval-Augmented Generation）技术的火灾预防知识问答服务，为Unity数字人提供LLM增强文本服务。

## 🎯 系统特性

### 核心功能

- 🔍 **智能检索**: 基于EmbeddingGemma-300M的向量相似度检索
- 🤖 **智能问答**: 结合DeepSeek LLM的答案生成
- 💾 **缓存加速**: 内存+磁盘二级缓存机制
- 📚 **高质量知识库**: 56个专业火灾预防问答
- 🌐 **RESTful API**: 标准化的API接口
- 🔧 **易于集成**: 专为Unity数字人设计

### 技术优势

- ✅ **专业准确**: 基于整理后的高质量问答知识库
- ✅ **快速响应**: 缓存机制确保重复查询瞬时响应
- ✅ **成本优化**: 显著减少LLM API调用次数
- ✅ **稳定可靠**: ModelScope模型管理，自动回退机制

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 内存: 2GB+ (推荐)
- 存储: 500MB+ (包含模型文件)

### 安装依赖

```bash
uv sync
```

### 配置API密钥

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/chat/completions

# RAG功能开关 (设置为 "true" 启用RAG，不设置或设置为其他值则直接使用LLM)
ENABLE_RAG=false
```

### 启动服务

```bash
python main.py
```

服务将在 `http://localhost:8000` 启动，API文档可在 `http://localhost:8000/docs` 查看。

## 📚 知识库内容

系统包含 **56个高质量火灾预防问答**，涵盖：

- **火灾预防** (25个): 电气安全、用火管理、设备维护
- **火灾防护** (7个): 消防设施、报警系统、防护装备  
- **火灾逃生** (9个): 疏散技巧、自救方法
- **火灾救援** (7个): 灭火原理、应急处理
- **法律法规** (8个): 消防法规定、责任义务

## 🔌 API接口

### 问答接口

```http
POST /api/ask
Content-Type: application/json

{
  "question": "如何使用灭火器？"
}
```

响应：

```json
{
  "answer": "灭火器的使用方法包括...",
  "relevant_docs": [...],
  "processing_time": 1.23
}
```

### 健康检查

```http
GET /api/health
```

### 缓存统计

```http
GET /api/cache/stats
```

## 🏗️ 项目结构

```
rag/
├── src/                    # 源代码目录
│   ├── __init__.py        # 包初始化
│   ├── models.py          # 数据模型
│   ├── vector_store.py    # 向量存储管理（带缓存）
│   ├── qa_service.py      # 问答服务（带缓存）
│   ├── api.py            # API路由
│   └── cache_manager.py   # 缓存管理
├── data/                  # 数据目录
│   └── _docs/
│       └── docs.json     # 问答知识库（56个问答）
├── main.py               # 主程序入口
├── config.py             # 配置文件
├── pyproject.toml        # 项目配置
├── DEPLOYMENT_GUIDE.md   # 详细部署指南
└── README.md            # 项目说明
```

## ⚙️ 系统配置

### 向量存储配置

- **嵌入模型**: EmbeddingGemma-300M (通过ModelScope下载)
- **向量数据库**: ChromaDB (`./chroma_db_gemma`)
- **缓存机制**: 内存+磁盘二级缓存

### 缓存配置

- **向量检索缓存**: 2小时TTL
- **问答结果缓存**: 1小时TTL
- **缓存目录**: `./cache/`

## 🎮 Unity集成示例

```csharp
// Unity C# 示例
public async Task<string> AskQuestion(string question)
{
    using var client = new HttpClient();
    var request = new 
    {
        question = question
    };
    
    var response = await client.PostAsJsonAsync("http://localhost:8000/api/ask", request);
    var result = await response.Content.ReadFromJsonAsync<AnswerResponse>();
    
    return result.answer;
}

// 使用示例
string answer = await AskQuestion("在空调出风口晾晒衣物有何火灾风险？");
```

## 📊 性能指标

### 预期性能

- **检索时间**: <0.15s (使用EmbeddingGemma-300M)
- **缓存命中响应**: <0.01s
- **答案生成**: <2.0s (使用DeepSeek API)
- **内存使用**: ~150MB

### 缓存效果

- **LLM API调用减少**: 80%+ (针对重复问题)
- **向量检索加速**: 95%+ (针对相同查询)

## 🔧 维护管理

### 知识库更新

系统使用预处理的问答知识库，位于 `data/_docs/docs.json`。

### 缓存管理

- 缓存自动过期管理
- 支持手动清除缓存
- 缓存统计监控

### 模型管理

- 首次启动自动下载EmbeddingGemma-300M
- ModelScope国内镜像，下载稳定
- 失败时自动回退到默认嵌入

## 🛠️ 故障排除

### 常见问题

1. **模型下载失败**: 检查网络连接，确认ModelScope服务可用
2. **API密钥错误**: 检查 `.env` 文件配置
3. **内存不足**: 系统自动回退到默认嵌入模型

### 日志监控

系统日志位于控制台输出，包含：

- 模型加载状态
- 检索性能指标
- 缓存使用情况
- 错误信息

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目。

---

**为您的Unity数字人提供专业、准确、快速的中文火灾预防问答服务！**
