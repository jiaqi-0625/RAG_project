"""
嵌入模型抽象层。

设计思路：定义一个 BaseEmbedder 接口，所有嵌入模型实现此接口。
这样后续可以无缝切换 embeddinggemma / nomic-embed-text / bge-m3 等不同模型，
而无需修改上层代码。

面试时可以说："我设计了一个可替换的嵌入模型抽象层，遵循开闭原则。"
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import ollama

from .config import config

logger = logging.getLogger(__name__)


class EmbedderError(Exception):
    """嵌入模型通用异常基类"""


class OllamaConnectionError(EmbedderError):
    """无法连接到 Ollama 服务"""


class EmbeddingModelError(EmbedderError):
    """嵌入模型推理失败（模型不存在、显存不足等）"""


class BaseEmbedder(ABC):
    """嵌入模型基类"""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """将单段文本转为向量"""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量将多段文本转为向量"""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """返回嵌入向量的维度"""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称"""
        ...


class OllamaEmbedder(BaseEmbedder):
    """
    基于 Ollama 的嵌入模型封装。

    支持 embeddinggemma、nomic-embed-text、bge-m3 等任何 Ollama 拉取的嵌入模型。
    目前默认使用 Google EmbeddingGemma (768 维)。
    """

    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
    ):
        self._model_name = model_name or config.EMBEDDING_MODEL
        self._dimensions = config.EMBEDDING_DIMENSIONS
        self._host = host or config.OLLAMA_HOST

        if self._host != "http://localhost:11434":
            self._client = ollama.Client(host=self._host)
        else:
            self._client = ollama

    def embed(self, text: str) -> List[float]:
        """对单段文本做 embedding

        Raises:
            OllamaConnectionError: Ollama 服务不可达
            EmbeddingModelError: 模型未找到或推理失败
        """
        try:
            response = self._client.embed(
                model=self._model_name,
                input=text,
            )
        except (ConnectionError, ConnectionRefusedError) as e:
            raise OllamaConnectionError(
                f"无法连接到 Ollama 服务 ({self._host})。"
                f"请确认已执行 'ollama serve' 启动服务。"
            ) from e
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "model" in error_msg:
                raise EmbeddingModelError(
                    f"嵌入模型 '{self._model_name}' 未找到。"
                    f"请执行 'ollama pull {self._model_name}' 下载模型。"
                ) from e
            raise EmbeddingModelError(
                f"嵌入失败: {e}"
            ) from e

        if "embeddings" not in response or not response["embeddings"]:
            raise EmbeddingModelError(
                f"嵌入模型 '{self._model_name}' 返回了空结果，请检查模型是否正常加载。"
            )
        return response["embeddings"][0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量 embedding（一次 HTTP 请求处理多段文本）

        Raises:
            OllamaConnectionError: Ollama 服务不可达
            EmbeddingModelError: 模型未找到或推理失败
        """
        if not texts:
            return []

        try:
            response = self._client.embed(
                model=self._model_name,
                input=texts,
            )
        except (ConnectionError, ConnectionRefusedError) as e:
            raise OllamaConnectionError(
                f"无法连接到 Ollama 服务 ({self._host})。"
                f"请确认已执行 'ollama serve' 启动服务。"
            ) from e
        except Exception as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "model" in error_msg:
                raise EmbeddingModelError(
                    f"嵌入模型 '{self._model_name}' 未找到。"
                    f"请执行 'ollama pull {self._model_name}' 下载模型。"
                ) from e
            raise EmbeddingModelError(
                f"批量嵌入失败: {e}"
            ) from e

        if "embeddings" not in response or not response["embeddings"]:
            raise EmbeddingModelError(
                f"嵌入模型 '{self._model_name}' 返回了空结果，请检查模型是否正常加载。"
            )

        # 验证返回的向量数量与输入文本数量一致
        if len(response["embeddings"]) != len(texts):
            logger.warning(
                f"批量嵌入返回数量不匹配: 期望 {len(texts)}, 实际 {len(response['embeddings'])}"
            )

        return response["embeddings"]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def __repr__(self) -> str:
        return f"OllamaEmbedder(model={self._model_name}, dim={self._dimensions})"
