"""
Reranker（重排序）模块 — 对向量检索结果进行精排。

设计思路：定义一个 BaseReranker 接口，CrossEncoderReranker 作为默认实现。
Reranker 在向量检索之后、LLM 生成之前介入：
  1. 向量检索召回 Top-K 候选（如 20 个）
  2. Cross-Encoder 对每个 (query, chunk) 对打分
  3. 按分数降序排列，取 Top-N（如 5 个）送入 LLM

面试时可以说：
  "我在检索和生成之间加入了一个 Cross-Encoder Reranker，
   用精排模型对粗排结果二次排序，显著提升了答案的准确性。"

当前支持的 Reranker：
  - cross-encoder/ms-marco-MiniLM-L-6-v2  (英文，~87MB，速度快)
  - BAAI/bge-reranker-v2-m3               (多语言，~2.3GB，更强大)
  - mixedbread-ai/mxbai-rerank-xsmall-v1  (多语言，~280MB，平衡之选)
"""

from abc import ABC, abstractmethod
import logging

from agno.knowledge.document import Document
from agno.knowledge.reranker.sentence_transformer import SentenceTransformerReranker

from .config import config

logger = logging.getLogger(__name__)


class RerankerError(Exception):
    """Reranker 通用异常基类"""


class RerankerModelError(RerankerError):
    """Reranker 模型加载或推理失败"""


class BaseReranker(ABC):
    """Reranker 基类 — 所有 reranker 实现必须遵循此接口"""

    @abstractmethod
    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        对文档列表进行重排序。

        Args:
            query: 用户查询。
            documents: 待重排序的文档列表（通常来自向量检索的 Top-K 候选）。

        Returns:
            按相关性从高到低排列的文档列表。
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回 reranker 模型名称"""
        ...


class CrossEncoderReranker(BaseReranker):
    """
    基于 Sentence-Transformers Cross-Encoder 的 Reranker。

    底层封装 agno 的 SentenceTransformerReranker，
    自动处理模型下载、推理和排序。

    工作原理：
      - 对每个 (query, document) 对，Cross-Encoder 输出一个相关性分数
      - 分数是 raw logits，越高表示越相关
      - 按分数降序排列所有文档

    Usage:
        reranker = CrossEncoderReranker(top_n=5)
        reranked = reranker.rerank("What is RAG?", documents)
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_n: int | None = None,
    ):
        """
        Args:
            model_name: Cross-Encoder 模型名称，默认使用配置中的 RERANKER_MODEL。
            top_n: 重排序后保留的文档数，默认使用配置中的 RERANKER_TOP_N。
        """
        self._model_name = model_name or config.RERANKER_MODEL
        self._top_n = top_n if top_n is not None else config.RERANKER_TOP_N

        logger.info(
            f"正在初始化 CrossEncoderReranker: model={self._model_name}, top_n={self._top_n}"
        )

        try:
            # 使用 agno 内置的 SentenceTransformerReranker
            self._reranker = SentenceTransformerReranker(
                model=self._model_name,
                top_n=self._top_n,
            )
        except Exception as e:
            raise RerankerModelError(
                f"无法加载 Reranker 模型 '{self._model_name}': {e}。"
                f"请确认模型名称正确且网络可访问（首次加载需下载模型）。"
            ) from e

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        对文档列表进行重排序。

        委托给 agno SentenceTransformerReranker，它内部会：
        1. 构建 (query, doc.content) 对
        2. 用 CrossEncoder 打分
        3. 将分数写入 doc.reranking_score
        4. 按分数降序排列
        5. 截断至 top_n
        """
        if not documents:
            return documents

        logger.debug(f"正在重排序 {len(documents)} 个文档 (query: {query[:80]}...)")
        try:
            reranked = self._reranker.rerank(query=query, documents=documents)
            logger.debug(f"重排序完成，保留 {len(reranked)} 个文档")
            return reranked  # type: ignore[no-any-return]
        except Exception as e:
            logger.error(f"重排序失败: {e}。回退到原始检索结果。")
            # 降级策略：重排序失败时返回原始结果（保留最好的 top_n 个）
            return documents[: self._top_n]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def top_n(self) -> int:
        return self._top_n

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self._model_name}, top_n={self._top_n})"


def create_reranker(
    model_name: str | None = None,
    top_n: int | None = None,
    enabled: bool | None = None,
) -> CrossEncoderReranker | None:
    """
    Reranker 工厂函数 — 根据配置创建 reranker 实例。

    Args:
        model_name: Cross-Encoder 模型名称，默认从 config 读取。
        top_n: 重排序后保留的文档数，默认从 config 读取。
        enabled: 是否启用 reranker，默认从 config 读取。设为 False 时返回 None。

    Returns:
        CrossEncoderReranker 实例，或 None（如果禁用）。

    Usage:
        reranker = create_reranker()
        if reranker:
            results = reranker.rerank(query, docs)

        # 禁用 reranker
        reranker = create_reranker(enabled=False)  # → None

        # 使用多语言模型
        reranker = create_reranker(
            model_name="BAAI/bge-reranker-v2-m3",
            top_n=10,
        )
    """
    _enabled = enabled if enabled is not None else config.RERANKER_ENABLED

    if not _enabled:
        logger.info("Reranker is disabled.")
        return None

    return CrossEncoderReranker(
        model_name=model_name,
        top_n=top_n,
    )
