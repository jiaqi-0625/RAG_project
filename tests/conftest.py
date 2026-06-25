"""
pytest 配置 — 注册自定义 markers。
"""

import pytest


def pytest_configure(config):
    """注册项目自定义 pytest 标记"""
    config.addinivalue_line(
        "markers", "integration: 需要 Ollama 服务运行才能通过的集成测试"
    )
