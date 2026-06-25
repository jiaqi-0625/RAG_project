"""
文档加载器 — 支持多种文档格式和来源。

负责判断文档来源类型（URL vs 本地文件）、格式验证，
实际文档解析委托给 agno Knowledge 内置的各种 Reader。

当前支持：
- PDF URL / 本地 PDF（通过 agno pdf_reader）
- 网页 URL / 本地 HTML（通过 agno website_reader）
- 本地 TXT / Markdown（通过 agno 内置 Reader）
- 本地 Word 文档 .docx（通过 agno docx_reader）

后续扩展方向：
- YouTube — agno 已内置 youtube_reader
- CSV / Excel — agno 已内置 csv_reader / excel_reader
"""

import logging
from pathlib import Path
from typing import Set
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    统一文档加载器 — 判断来源类型，委托 agno Knowledge 加载。

    Usage:
        loader = DocumentLoader()
        loader.validate_source("https://example.com/sample.pdf")  # True
        loader.validate_source("/local/file.pdf")                  # True
        loader.validate_source("invalid.xyz")                      # False
    """

    # 支持的文件扩展名
    SUPPORTED_SUFFIXES: Set[str] = {
        ".pdf", ".txt", ".md", ".markdown",
        ".docx", ".html", ".htm",
    }

    def __init__(self):
        pass

    def validate_source(self, source: str) -> bool:
        """
        验证来源是否有效（URL 或支持的文件格式 + 本地文件存在性）。

        Returns:
            True 如果来源有效，False 如果格式不支持或本地文件不存在。
        """
        if not source or not source.strip():
            return False

        if self._is_url(source):
            return True

        # 本地文件：检查格式
        suffix = Path(source).suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            return False

        # 本地文件：检查是否存在
        if not Path(source).exists():
            logger.warning(f"本地文件不存在: {source}")
            return False

        return True

    def get_source_type(self, source: str) -> str:
        """
        返回来源类型。

        Returns:
            'url' | 'local_pdf' | 'local_txt' | 'local_md' | 'unknown'
        """
        if self._is_url(source):
            return "url"

        suffix = Path(source).suffix.lower()
        type_map = {
            ".pdf": "local_pdf",
            ".txt": "local_txt",
            ".md": "local_md",
            ".markdown": "local_md",
            ".docx": "local_docx",
            ".html": "local_html",
            ".htm": "local_html",
        }
        return type_map.get(suffix, "unknown")

    @staticmethod
    def _is_url(source: str) -> bool:
        """判断 source 是 URL 还是本地路径"""
        parsed = urlparse(source)
        return parsed.scheme in ("http", "https")

    def __repr__(self) -> str:
        return f"DocumentLoader(supported={sorted(self.SUPPORTED_SUFFIXES)})"
