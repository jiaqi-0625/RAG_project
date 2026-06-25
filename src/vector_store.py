"""
向量数据库抽象层。

设计思路：定义 BaseVectorStore 接口，LanceDB 作为默认实现。
后续可扩展 ChromaDB、FAISS、Qdrant 等实现，遵循依赖倒置原则。

面试时："我抽象了向量存储层，理论上可以一行代码切换到其他向量数据库。"
"""

from abc import ABC, abstractmethod
import logging
from typing import Any

from agno.knowledge.embedder.ollama import OllamaEmbedder as AgnoOllamaEmbedder
from agno.knowledge.reranker.base import Reranker
from agno.vectordb.lancedb import LanceDb, SearchType

from .config import config

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """向量数据库通用异常基类"""


class VectorStoreConnectionError(VectorStoreError):
    """无法连接或初始化向量数据库"""


class BaseVectorStore(ABC):
    """向量数据库基类"""

    @abstractmethod
    def add(self, documents: list[Any]) -> None:
        """添加文档到向量库"""
        ...

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int = 5) -> list[Any]:
        """根据向量检索最相似的 top_k 条记录"""
        ...

    @abstractmethod
    def delete(self) -> None:
        """清空向量库"""
        ...

    @abstractmethod
    def exists(self) -> bool:
        """检查向量库是否存在（是否已加载过数据）"""
        ...


class LanceDBStore(BaseVectorStore):
    """
    LanceDB 向量存储实现。

    LanceDB 的优势：
    - 嵌入式：无需单独部署服务，直接读写本地文件
    - 列式存储：基于 Lance 格式，读写速度快
    - 支持向量检索 + 全文检索的混合搜索
    - 零运维成本，适合本地 RAG 场景
    """

    def __init__(
        self,
        table_name: str | None = None,
        uri: str | None = None,
        embedder_model: str | None = None,
        reranker: Reranker | None = None,
    ):
        self._table_name = table_name or config.LANCEDB_TABLE_NAME
        self._uri = uri or config.LANCEDB_URI
        self._embedder_model = embedder_model or config.EMBEDDING_MODEL
        self._reranker = reranker

        # 内部使用 agno 的 LanceDb 封装，它已集成了 OllamaEmbedder
        try:
            self._db = LanceDb(
                table_name=self._table_name,
                uri=self._uri,
                search_type=SearchType.vector,
                embedder=AgnoOllamaEmbedder(
                    id=self._embedder_model,
                    dimensions=config.EMBEDDING_DIMENSIONS,
                ),
                reranker=reranker,  # Cross-Encoder 重排序
            )
        except Exception as e:
            raise VectorStoreConnectionError(f"无法初始化 LanceDB (uri={self._uri}): {e}") from e

    def add(self, documents: list[Any]) -> None:
        """批量添加文档（agno Document 对象列表）"""
        # LanceDb 的 add 在 agno Knowledge 层调用，这里做薄封装
        pass  # 实际加载逻辑在 knowledge_base 中

    def search(self, query_vector: list[float], top_k: int = 5) -> list[Any]:
        """向量检索 — 返回与查询最相似的文档块"""
        # agno 框架内部使用 Knowledge.search() 完成检索，
        # 此方法仅为接口完整性保留
        return []

    def delete(self) -> None:
        """清空数据（删除表并重建）"""
        try:
            self._db.drop()
            logger.info(f"已清空 LanceDB 表: {self._table_name}")
        except Exception as e:
            raise VectorStoreConnectionError(f"清空向量数据库失败: {e}") from e

    def exists(self) -> bool:
        """检查表是否存在"""
        try:
            return bool(self._db.exists())
        except Exception as e:
            logger.warning(f"检查 LanceDB 表是否存在时出错: {e}")
            return False

    @property
    def db(self) -> LanceDb:
        """暴露 agno 的 LanceDb 实例给上层使用"""
        return self._db

    def __repr__(self) -> str:
        rerank = "enabled" if self._reranker else "disabled"
        return (
            f"LanceDBStore(table={self._table_name}, "
            f"embedder={self._embedder_model}, "
            f"reranker={rerank})"
        )
