"""
Agentic RAG 核心包 — src 包初始化。

对外暴露主要类和函数，方便外部 import。
"""

from .agent import AgentFactory
from .config import Config, config
from .document_loader import DocumentLoader
from .embedder import (
    BaseEmbedder,
    EmbedderError,
    EmbeddingModelError,
    OllamaConnectionError,
    OllamaEmbedder,
)
from .knowledge_base import KnowledgeBaseManager
from .reranker import (
    BaseReranker,
    CrossEncoderReranker,
    RerankerError,
    RerankerModelError,
    create_reranker,
)
from .vector_store import (
    BaseVectorStore,
    LanceDBStore,
    VectorStoreConnectionError,
    VectorStoreError,
)

__all__ = [
    # 配置
    "config",
    "Config",
    # Embedder
    "BaseEmbedder",
    "OllamaEmbedder",
    "EmbedderError",
    "OllamaConnectionError",
    "EmbeddingModelError",
    # 向量数据库
    "BaseVectorStore",
    "LanceDBStore",
    "VectorStoreError",
    "VectorStoreConnectionError",
    # 文档加载
    "DocumentLoader",
    # 知识库
    "KnowledgeBaseManager",
    # Agent
    "AgentFactory",
    # Reranker
    "BaseReranker",
    "CrossEncoderReranker",
    "create_reranker",
    "RerankerError",
    "RerankerModelError",
]
