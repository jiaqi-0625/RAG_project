"""
测试 document_loader 模块。

运行: pytest tests/test_document_loader.py -v
"""

import pytest
from pathlib import Path
from src.document_loader import DocumentLoader


class TestDocumentLoader:
    """测试 DocumentLoader"""

    def test_is_url(self):
        """验证 URL 判断逻辑"""
        assert DocumentLoader._is_url("https://example.com/doc.pdf") is True
        assert DocumentLoader._is_url("http://arxiv.org/paper.pdf") is True
        assert DocumentLoader._is_url("/local/path/file.pdf") is False
        assert DocumentLoader._is_url("C:\\Users\\file.pdf") is False

    def test_supported_suffixes(self):
        """验证支持的文件类型"""
        loader = DocumentLoader()
        assert ".pdf" in loader.SUPPORTED_SUFFIXES
        assert ".txt" in loader.SUPPORTED_SUFFIXES
        assert ".md" in loader.SUPPORTED_SUFFIXES

    def test_validate_nonexistent_file(self):
        """验证不存在的文件路径返回 False（现在会检查文件存在性）"""
        loader = DocumentLoader()
        # validate_source 现在会检查文件是否存在
        assert loader.validate_source("/nonexistent/path/file.pdf") is False

    def test_validate_unsupported_format(self):
        """验证不支持的格式返回 False"""
        loader = DocumentLoader()
        assert loader.validate_source("test.xyz") is False
        assert loader.validate_source("test.abc") is False

    def test_get_source_type_docx(self):
        """验证 .docx 文件类型识别"""
        loader = DocumentLoader()
        assert loader.get_source_type("test.docx") == "local_docx"

    def test_get_source_type_html(self):
        """验证 .html 文件类型识别"""
        loader = DocumentLoader()
        assert loader.get_source_type("test.html") == "local_html"
