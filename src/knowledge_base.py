"""
知识库管理模块。

负责：
1. 初始化向量数据库（LanceDB）
2. 加载文档内容到知识库
3. 对外提供统一的知识库实例
4. 持久化已加载来源列表（JSON），重启后自动恢复

它是 document_loader + vector_store + embedder 三者的协调层。
"""

import json
import logging
from pathlib import Path

from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reranker.base import Reranker

from .config import config
from .document_loader import DocumentLoader
from .vector_store import LanceDBStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 已加载来源的持久化文件
_SOURCES_STATE_FILE = config.DATA_DIR / "loaded_sources.json"


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
        vector_store: LanceDBStore | None = None,
        loader: DocumentLoader | None = None,
        reranker: Reranker | None = None,
    ):
        self._vector_store = vector_store or LanceDBStore(reranker=reranker)
        self._loader = loader or DocumentLoader()
        self._reranker = reranker

        # 确保上传目录存在
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 从持久化文件恢复已加载来源列表
        self._loaded_sources: set[str] = set()
        self._restore_sources()

        # 初始化 agno Knowledge 实例
        self._knowledge = Knowledge(
            vector_db=self._vector_store.db,
        )

    @property
    def knowledge(self) -> Knowledge:
        """返回 agno Knowledge 实例，供 Agent 使用"""
        return self._knowledge

    @property
    def loaded_sources(self) -> set[str]:
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
                raise RuntimeError(f"无法下载文档 (网络不可达或超时): {source}") from e
            elif "ollama" in error_msg.lower():
                raise RuntimeError(
                    f"嵌入失败 (Ollama 服务异常): {source}。"
                    f"请确认 Ollama 正在运行且模型已下载。"
                ) from e
            else:
                raise RuntimeError(f"文档加载失败: {source} — {e}") from e

        self._loaded_sources.add(source)
        self._save_sources()
        logger.info(f"成功加载来源: {source}")

    # ── 持久化: loaded_sources ↔ JSON ──

    def _save_sources(self) -> None:
        """将已加载来源列表持久化到 JSON 文件"""
        urls: list[str] = []
        files: list[dict[str, str]] = []
        for src in self._loaded_sources:
            if self._loader._is_url(src):
                urls.append(src)
            else:
                # 本地文件: 存路径 + 从路径推断显示名
                p = Path(src)
                files.append({"name": p.name, "path": src})

        state = {"urls": urls, "files": files}
        try:
            _SOURCES_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
            logger.debug(f"已保存来源状态到 {_SOURCES_STATE_FILE}")
        except OSError as e:
            logger.warning(f"无法保存来源状态: {e}")

    def _restore_sources(self) -> None:
        """从 JSON 文件恢复已加载来源列表。

        仅在 LanceDB 有数据时才恢复（防止 JSON 残留导致状态不一致）。
        """
        if not self._vector_store.exists():
            logger.info("LanceDB 为空，跳过来源恢复。")
            return
        if not _SOURCES_STATE_FILE.exists():
            logger.info("来源状态文件不存在，跳过恢复。")
            return

        try:
            state = json.loads(_SOURCES_STATE_FILE.read_text(encoding="utf-8"))
            urls: list[str] = state.get("urls", [])
            files: list[dict[str, str]] = state.get("files", [])

            for url in urls:
                self._loaded_sources.add(url)
            for f in files:
                path = f.get("path", "")
                if path and Path(path).exists():
                    self._loaded_sources.add(path)
                elif path:
                    # 文件已被移动或删除，但仍记录在案（LanceDB 中数据还在）
                    self._loaded_sources.add(path)
                    logger.debug(f"已加载文件不存在于磁盘但仍保留在来源列表: {path}")

            logger.info(
                f"已从 {_SOURCES_STATE_FILE} 恢复 {len(urls)} 个 URL、" f"{len(files)} 个文件"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"恢复来源状态失败: {e}")

    def get_loaded_state(self) -> dict:
        """返回已加载来源的状态信息，供 UI 层恢复 session state。

        Returns:
            {"urls": [...], "files": [{"name": ..., "path": ...}, ...]}
        """
        urls: list[str] = []
        files: list[dict[str, str]] = []
        for src in self._loaded_sources:
            if self._loader._is_url(src):
                urls.append(src)
            else:
                p = Path(src)
                files.append({"name": p.name, "path": src})
        return {"urls": urls, "files": files}

    def add_sources(self, sources: list[str]) -> int:
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
        # 清除持久化文件
        try:
            if _SOURCES_STATE_FILE.exists():
                _SOURCES_STATE_FILE.unlink()
        except OSError as e:
            logger.warning(f"无法删除来源状态文件: {e}")
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
