"""
测试 knowledge_base 模块。

运行: pytest tests/test_knowledge_base.py -v
"""

import pytest
from src.knowledge_base import KnowledgeBaseManager
from src.vector_store import LanceDBStore


class TestKnowledgeBaseManager:
    """测试 KnowledgeBaseManager"""

    def test_initialization(self):
        """验证初始化"""
        kb = KnowledgeBaseManager()
        assert kb.knowledge is not None
        assert len(kb.loaded_sources) == 0

    def test_repr(self):
        """验证字符串表示"""
        kb = KnowledgeBaseManager()
        assert "KnowledgeBaseManager" in repr(kb)

    def test_add_duplicate_source_raises(self):
        """验证重复添加同一个来源会报错"""
        kb = KnowledgeBaseManager()
        kb._loaded_sources.add("https://already-loaded.com/doc.pdf")
        with pytest.raises(ValueError, match="已加载过"):
            kb.add_source("https://already-loaded.com/doc.pdf")
