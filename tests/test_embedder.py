"""
测试 embedder 模块。

运行: pytest tests/test_embedder.py -v
"""

import pytest
from unittest.mock import patch, MagicMock
from src.embedder import (
    OllamaEmbedder,
    BaseEmbedder,
    OllamaConnectionError,
    EmbeddingModelError,
)


class TestOllamaEmbedder:
    """测试 OllamaEmbedder"""

    def test_embedder_implements_base(self):
        """验证 OllamaEmbedder 实现了 BaseEmbedder 接口"""
        embedder = OllamaEmbedder()
        assert isinstance(embedder, BaseEmbedder)

    def test_dimensions(self):
        """验证维度属性"""
        embedder = OllamaEmbedder()
        assert embedder.dimensions == 768

    def test_model_name(self):
        """验证模型名属性"""
        embedder = OllamaEmbedder()
        assert "embeddinggemma" in embedder.model_name

    def test_repr(self):
        """验证字符串表示"""
        embedder = OllamaEmbedder(model_name="test-model")
        assert "test-model" in repr(embedder)

    # --- 集成测试（需要 Ollama 运行）---
    @pytest.mark.integration
    def test_embed_single_text(self):
        """测试单段文本 embedding（需要 Ollama 服务运行）"""
        embedder = OllamaEmbedder()
        vector = embedder.embed("Hello world")
        assert len(vector) == 768
        assert all(isinstance(v, float) for v in vector)

    @pytest.mark.integration
    def test_embed_batch(self):
        """测试批量 embedding"""
        embedder = OllamaEmbedder()
        vectors = embedder.embed_batch(["First text", "Second text"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 768
        assert len(vectors[1]) == 768

    # --- 异常处理测试 ---

    def test_embed_connection_error(self):
        """测试 Ollama 连接失败时抛出 OllamaConnectionError"""
        embedder = OllamaEmbedder()
        with patch.object(
            embedder._client, "embed",
            side_effect=ConnectionError("Connection refused"),
        ):
            with pytest.raises(OllamaConnectionError, match="无法连接到"):
                embedder.embed("test")

    def test_embed_model_not_found(self):
        """测试模型未找到时抛出 EmbeddingModelError"""
        embedder = OllamaEmbedder()
        with patch.object(
            embedder._client, "embed",
            side_effect=Exception("model 'bad-model' not found"),
        ):
            with pytest.raises(EmbeddingModelError, match="未找到"):
                embedder.embed("test")

    def test_embed_empty_response(self):
        """测试空响应时抛出 EmbeddingModelError"""
        embedder = OllamaEmbedder()
        with patch.object(
            embedder._client, "embed",
            return_value={"embeddings": []},
        ):
            with pytest.raises(EmbeddingModelError, match="空结果"):
                embedder.embed("test")

    def test_embed_batch_empty_list(self):
        """测试空列表直接返回空列表"""
        embedder = OllamaEmbedder()
        result = embedder.embed_batch([])
        assert result == []
