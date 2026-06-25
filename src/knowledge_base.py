"""
知识库管理模块。

负责：
1. 初始化向量数据库（LanceDB）
2. 加载文档内容到知识库
3. 对外提供统一的知识库实例

它是 document_loader + vector_store + embedder 三者的协调层。
"""

import logging
from pathlib import Path
from typing import List, Optional, Set

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reranker.base import Reranker

from .config import config
from .document_loader import DocumentLoader
from .vector_store import LanceDBStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeBaseManager:
    """
    知识库管理器 — 封装文档加载、嵌入、存储的全流程。

    底层委托给 agno Knowledge，它内部自动处理：
    - PDF 解析（通过 pypdf 或 pdf_reader）
    - 文本分块
    - 向量嵌入（通过 OllamaEmbedder）
    - 写入 LanceDB

    Usage:
        kb_manager = KnowledgeBaseManager()
        kb_manager.add_source("https://example.com/paper.pdf")
        kb_manager.add_source("/local/doc.txt")
        results = kb_manager.knowledge.search("query")
    """

    def __init__(
        self,
        vector_store: Optional[LanceDBStore] = None,
        loader: Optional[DocumentLoader] = None,
        reranker: Optional[Reranker] = None,
    ):
        self._vector_store = vector_store or LanceDBStore(reranker=reranker)
        self._loader = loader or DocumentLoader()
        self._reranker = reranker

        # 确保上传目录存在
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # 跟踪已加载的来源（避免重复加载）
        self._loaded_sources: Set[str] = set()

        # 初始化 agno Knowledge 实例
        self._knowledge = Knowledge(
            vector_db=self._vector_store.db,
        )

    @property
    def knowledge(self) -> Knowledge:
        """返回 agno Knowledge 实例，供 Agent 使用"""
        return self._knowledge

    @property
    def loaded_sources(self) -> Set[str]:
        """已加载的来源集合"""
        return self._loaded_sources

    def add_source(self, source: str) -> None:
        """
        添加一个文档来源（URL 或本地文件）。

        底层使用 agno Knowledge.add_content()，它会自动：
        1. 判断来源类型（URL / 本地文件）
        2. 选择合适的 Reader 解析文档
        3. 分块 → 嵌入 → 写入 LanceDB

        Args:
            source: URL 或本地文件路径。

        Raises:
            ValueError: 来源已加载过或格式不支持。
            RuntimeError: 文档下载、解析或嵌入失败。
        """
        if source in self._loaded_sources:
            raise ValueError(f"来源已加载过: {source}")

        if not self._loader.validate_source(source):
            raise ValueError(
                f"不支持的来源: {source}。"
                f"支持的类型: 网页 URL、{sorted(self._loader.SUPPORTED_SUFFIXES)}"
            )

        logger.info(f"正在加载文档: {source}")
        try:
            # 区分来源类型，使用正确的 agno API
            # - URL → add_content(url=...)
            # - 本地文件 → add_content(path=...)
            if self._loader._is_url(source):
                self._knowledge.add_content(url=source)
            else:
                self._knowledge.add_content(path=source)
        except Exception as e:
            error_msg = str(e)
            if "connect" in error_msg.lower() or "timeout" in error_msg.lower():
                raise RuntimeError(
                    f"无法下载文档 (网络不可达或超时): {source}"
                ) from e
            elif "ollama" in error_msg.lower():
                raise RuntimeError(
                    f"嵌入失败 (Ollama 服务异常): {source}。"
                    f"请确认 Ollama 正在运行且模型已下载。"
                ) from e
            else:
                raise RuntimeError(
                    f"文档加载失败: {source} — {e}"
                ) from e

        self._loaded_sources.add(source)
        logger.info(f"成功加载来源: {source}")

    def add_sources(self, sources: List[str]) -> int:
        """批量添加文档来源，返回成功添加的数量"""
        count = 0
        for source in sources:
            try:
                self.add_source(source)
                count += 1
            except Exception as e:
                logger.error(f"Failed to add source '{source}': {e}")
        return count

    def clear(self) -> None:
        """清空知识库并重建"""
        self._vector_store.delete()
        self._loaded_sources.clear()
        self._knowledge = Knowledge(vector_db=self._vector_store.db)
        logger.info("Knowledge base cleared.")

    def exists(self) -> bool:
        """检查知识库是否已有数据"""
        return self._vector_store.exists()

    def __repr__(self) -> str:
        rerank = "enabled" if self._reranker else "disabled"
        return (
            f"KnowledgeBaseManager("
            f"sources={len(self._loaded_sources)}, "
            f"store={self._vector_store}, "
            f"reranker={rerank})"
        )
