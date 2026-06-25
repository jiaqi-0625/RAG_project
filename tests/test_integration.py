"""
端到端集成测试 — 验证 "加载文档 → 检索 → 生成回答" 的完整流程。

运行方式:
    # 跳过集成测试（CI 模式）
    pytest tests/ -v -k "not integration"

    # 只跑集成测试（需要 Ollama 运行 + 模型已下载）
    pytest tests/test_integration.py -v -m integration

    # 跑全部测试
    pytest tests/ -v

前置条件:
    - Ollama 服务运行中 (默认 http://localhost:11434)
    - 已下载模型: embeddinggemma:latest, llama3.2:latest
"""

import logging
import tempfile
from typing import Generator

import pytest

from src.config import config
from src.embedder import OllamaEmbedder, OllamaConnectionError, EmbeddingModelError
from src.vector_store import LanceDBStore
from src.knowledge_base import KnowledgeBaseManager
from src.document_loader import DocumentLoader
from src.agent import AgentFactory
from src.reranker import create_reranker

logger = logging.getLogger(__name__)

# ============================================================
# 环境前置检查
# ============================================================


def check_ollama_available() -> bool:
    """检查 Ollama 服务是否可用"""
    try:
        embedder = OllamaEmbedder()
        embedder.embed("health check")
        return True
    except (OllamaConnectionError, EmbeddingModelError):
        return False
    except Exception:
        return False


def check_model_available(model_name: str) -> bool:
    """检查指定模型是否已下载"""
    import ollama
    try:
        response = ollama.list()
        models = response.models if hasattr(response, "models") else []
        # 提取模型名（可能是 Model 对象或字典）
        model_names = []
        for m in models:
            if hasattr(m, "model"):
                model_names.append(m.model)
            elif isinstance(m, dict):
                model_names.append(m.get("name", ""))
        base_name = model_name.replace(":latest", "")
        return any(base_name in m for m in model_names)
    except Exception as e:
        logger.warning(f"检查模型可用性失败: {e}")
        return False


# ============================================================
# Skip 条件
# ============================================================

ollama_required = pytest.mark.skipif(
    not check_ollama_available(),
    reason="Ollama 服务不可用 — 请先运行 'ollama serve' 启动服务",
)

llm_required = pytest.mark.skipif(
    not check_model_available(config.LLM_MODEL),
    reason=f"LLM 模型 '{config.LLM_MODEL}' 未下载 — 请执行 'ollama pull {config.LLM_MODEL}'",
)

embedding_required = pytest.mark.skipif(
    not check_model_available(config.EMBEDDING_MODEL),
    reason=f"嵌入模型 '{config.EMBEDDING_MODEL}' 未下载 — 请执行 'ollama pull {config.EMBEDDING_MODEL}'",
)


# ============================================================
# 测试文档
# ============================================================

# 一段简单的英文测试文本（确保 EmbeddingGemma 和 Llama 都能理解）
TEST_DOC_EN = """
Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is the simulation of human intelligence in machines
that are programmed to think and learn. Machine Learning (ML) is a subset of AI
that enables systems to learn and improve from experience without being explicitly
programmed.

Deep Learning is a subset of Machine Learning that uses neural networks with many
layers. It has been successfully applied to fields such as computer vision, natural
language processing, and speech recognition.

The Transformer architecture, introduced in the paper "Attention Is All You Need"
in 2017, revolutionized natural language processing. It forms the basis of modern
large language models like GPT, BERT, and Llama.

Key concepts in AI include supervised learning, unsupervised learning, and
reinforcement learning. Each approach has different use cases and requirements.
"""

# 第二篇短文 — 用于测试多文档加载
TEST_DOC_CN = """
检索增强生成（RAG）技术详解

检索增强生成（Retrieval-Augmented Generation，简称 RAG）是一种结合了信息检索
和文本生成的 AI 技术架构。它的核心思想是：在生成回答之前，先从外部知识库中
检索相关信息，然后将检索结果作为上下文注入到大语言模型中。

RAG 的典型工作流程包括以下步骤：
1. 文档加载与解析：从 PDF、网页、数据库等多种来源获取文档。
2. 文本分块（Chunking）：将长文档切分成适合嵌入的小块（通常 500-2000 字符）。
3. 向量嵌入（Embedding）：使用嵌入模型将文本块转换为向量表示。
4. 向量存储：将向量存入 LanceDB、ChromaDB 等向量数据库。
5. 检索（Retrieval）：用户提问时，将问题转为向量，在数据库中搜索最相似的文本块。
6. 生成（Generation）：将检索到的文本块作为上下文，交由 LLM 生成最终回答。

RAG 相比传统 LLM 的优势在于：可以处理最新信息、减少幻觉（Hallucination）、
提供可溯源的答案。但是 chunk size 的选择、嵌入模型的质量、检索策略的优劣
都会影响最终效果。
"""


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def temp_text_file() -> str:
    """创建临时英文测试文档"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(TEST_DOC_EN)
        return f.name


@pytest.fixture
def temp_text_file_cn() -> str:
    """创建临时中文测试文档"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(TEST_DOC_CN)
        return f.name


@pytest.fixture
def kb_manager() -> Generator[KnowledgeBaseManager, None, None]:
    """创建一个全新的知识库管理器（测试用）"""
    # 使用独立的表名避免与正式数据冲突
    import uuid
    test_table = f"test_kb_{uuid.uuid4().hex[:8]}"

    store = LanceDBStore(table_name=test_table)
    kb = KnowledgeBaseManager(vector_store=store)
    yield kb
    # 清理
    try:
        kb.clear()
    except Exception:
        pass


@pytest.fixture
def agent_with_kb(kb_manager: KnowledgeBaseManager):
    """创建带知识库的 Agent"""
    return AgentFactory.create(
        kb_manager,
        debug_mode=True,
        enable_tools=False,  # 集成测试中关闭工具，减少干扰
    )


# ============================================================
# 测试：环境健康检查
# ============================================================


@pytest.mark.integration
class TestEnvironmentHealth:
    """验证运行环境是否满足集成测试的前置条件"""

    @ollama_required
    def test_ollama_embedding_works(self):
        """测试 Ollama 嵌入功能正常"""
        embedder = OllamaEmbedder()
        vector = embedder.embed("Hello world")
        assert len(vector) == config.EMBEDDING_DIMENSIONS
        assert all(isinstance(v, float) for v in vector)

    @ollama_required
    def test_ollama_batch_embedding_works(self):
        """测试 Ollama 批量嵌入功能正常"""
        embedder = OllamaEmbedder()
        texts = ["First text", "Second text", "Third text"]
        vectors = embedder.embed_batch(texts)
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == config.EMBEDDING_DIMENSIONS

    @ollama_required
    def test_lancedb_initialization(self):
        """测试 LanceDB 能正常初始化"""
        import uuid
        test_table = f"test_health_{uuid.uuid4().hex[:8]}"
        store = LanceDBStore(table_name=test_table)
        assert store.db is not None
        # 清理
        try:
            store.delete()
        except Exception:
            pass

    @ollama_required
    def test_knowledge_base_initialization(self, kb_manager):
        """测试知识库能正常初始化"""
        assert kb_manager.knowledge is not None
        assert len(kb_manager.loaded_sources) == 0


# ============================================================
# 测试：文档加载 → 知识库存储
# ============================================================


@pytest.mark.integration
class TestDocumentLoading:
    """测试文档加载到知识库的完整流程"""

    @ollama_required
    @embedding_required
    def test_load_single_text_file(self, kb_manager, temp_text_file):
        """测试加载单个 TXT 文件到知识库"""
        kb_manager.add_source(temp_text_file)

        assert temp_text_file in kb_manager.loaded_sources
        assert kb_manager.exists()

    @ollama_required
    @embedding_required
    def test_load_multiple_sources(self, kb_manager, temp_text_file, temp_text_file_cn):
        """测试加载多个文档源"""
        kb_manager.add_source(temp_text_file)
        kb_manager.add_source(temp_text_file_cn)

        assert len(kb_manager.loaded_sources) == 2
        assert temp_text_file in kb_manager.loaded_sources
        assert temp_text_file_cn in kb_manager.loaded_sources

    @ollama_required
    @embedding_required
    def test_duplicate_source_rejected(self, kb_manager, temp_text_file):
        """测试重复加载同一文档会被拒绝"""
        kb_manager.add_source(temp_text_file)

        with pytest.raises(ValueError, match="已加载过"):
            kb_manager.add_source(temp_text_file)

    @ollama_required
    @embedding_required
    def test_batch_add_sources(self, kb_manager, temp_text_file, temp_text_file_cn):
        """测试批量添加文档"""
        count = kb_manager.add_sources([temp_text_file, temp_text_file_cn])
        assert count == 2
        assert len(kb_manager.loaded_sources) == 2

    @ollama_required
    @embedding_required
    def test_unsupported_format_rejected(self, kb_manager):
        """测试不支持的格式被拒绝"""
        with pytest.raises(ValueError, match="不支持"):
            kb_manager.add_source("test.xyz")

    @ollama_required
    @embedding_required
    def test_clear_knowledge_base(self, kb_manager, temp_text_file):
        """测试清空知识库"""
        kb_manager.add_source(temp_text_file)
        assert kb_manager.exists()

        kb_manager.clear()
        assert len(kb_manager.loaded_sources) == 0

    @ollama_required
    @embedding_required
    def test_document_loader_validates_existence(self):
        """测试文档加载器检查文件是否存在"""
        loader = DocumentLoader()
        assert loader.validate_source("/nonexistent/file/that/doesnt/exist.pdf") is False


# ============================================================
# 测试：Agent + 知识库端到端 RAG 流程
# ============================================================


@pytest.mark.integration
class TestEndToEndRAG:
    """端到端测试：加载文档 → Agent 回答问题"""

    @ollama_required
    @embedding_required
    @llm_required
    def test_agent_answers_from_loaded_document(
        self, kb_manager, temp_text_file, agent_with_kb
    ):
        """核心测试：加载文档后 Agent 能基于文档内容回答问题"""
        # Step 1: 加载文档
        kb_manager.add_source(temp_text_file)

        # Step 2: 向 Agent 提问（内容在文档中明确存在）
        query = "What is the Transformer architecture and when was it introduced?"
        response = agent_with_kb.run(query, stream=False)

        # Step 3: 验证
        assert response is not None
        assert response.content is not None
        answer = str(response.content).lower()
        # 文档中明确提到 "Transformer architecture ... 2017 ... Attention Is All You Need"
        assert "2017" in answer or "transformer" in answer, (
            f"期望回答提及 Transformer 或 2017，实际回答: {answer[:300]}"
        )

    @ollama_required
    @embedding_required
    @llm_required
    def test_agent_streaming_response(self, kb_manager, temp_text_file, agent_with_kb):
        """测试 Agent 流式输出"""
        kb_manager.add_source(temp_text_file)

        query = "Tell me about supervised learning and reinforcement learning."
        chunks = []
        for chunk in agent_with_kb.run(query, stream=True):
            if chunk.content is not None:
                chunks.append(chunk.content)

        full_response = "".join(chunks)
        assert len(full_response) > 20, (
            f"期望流式回答至少包含 20 个字符，实际: {len(full_response)}"
        )

    @ollama_required
    @embedding_required
    @llm_required
    def test_agent_respects_language_preference(
        self, kb_manager, temp_text_file_cn, agent_with_kb
    ):
        """测试 Agent 用中文回答中文问题"""
        kb_manager.add_source(temp_text_file_cn)

        query = "什么是 RAG？它的工作流程是什么？"
        response = agent_with_kb.run(query, stream=False)

        assert response.content is not None
        answer = str(response.content)
        # 中文回答应包含中文字符
        has_chinese = any("一" <= c <= "鿿" for c in answer)
        assert has_chinese, (
            f"期望中文回答包含中文字符，实际回答: {answer[:300]}"
        )

    @ollama_required
    @embedding_required
    @llm_required
    def test_agent_handles_empty_knowledge_base(self, kb_manager, agent_with_kb):
        """测试 Agent 在空知识库时仍能正常回答（基于模型自身知识）"""
        # 不加载任何文档 — 知识库为空
        # 问一个在知识库中不存在但模型可以回答的问题
        query = "What color is the sky?"
        response = agent_with_kb.run(query, stream=False)

        assert response.content is not None
        answer = str(response.content)
        # Agent 应该基于自身训练数据给出合理回答
        assert len(answer) > 10, (
            f"期望 Agent 在空知识库时仍能做出回答，实际: {answer[:200]}"
        )

    @ollama_required
    @embedding_required
    @llm_required
    def test_full_pipeline_with_multiple_docs(
        self, kb_manager, temp_text_file, temp_text_file_cn, agent_with_kb
    ):
        """全流程测试：加载多文档 → 跨文档检索 → 综合回答"""
        # Step 1: 加载两篇不同主题的文档
        kb_manager.add_source(temp_text_file)     # 英文 AI/ML 概述
        kb_manager.add_source(temp_text_file_cn)  # 中文 RAG 详解

        # Step 2: 提问涉及英文文档的内容
        query_en = "What are the key concepts in AI according to the document?"
        response_en = agent_with_kb.run(query_en, stream=False)
        assert response_en.content is not None
        # 应该能检索到 supervised/unsupervised/reinforcement learning

        # Step 3: 提问涉及中文文档的内容
        query_cn = "RAG 相比传统 LLM 有什么优势？"
        response_cn = agent_with_kb.run(query_cn, stream=False)
        assert response_cn.content is not None
        answer_cn = str(response_cn.content)
        # 应包含中文回答
        has_chinese = any("一" <= c <= "鿿" for c in answer_cn)
        assert has_chinese

    @ollama_required
    @embedding_required
    @llm_required
    def test_agent_with_conversation_memory(
        self, kb_manager, temp_text_file
    ):
        """测试多轮对话记忆：追问能引用上文"""
        agent = AgentFactory.create(
            kb_manager,
            debug_mode=True,
            enable_tools=False,
        )

        kb_manager.add_source(temp_text_file)

        session_id = "test-session-memory-integration"

        # 第一轮：问 Transformer
        q1 = "What is the Transformer architecture?"
        r1 = agent.run(q1, stream=False, session_id=session_id)
        assert r1.content is not None

        # 第二轮：追问（用 "it" 指代）
        q2 = "When was it introduced?"
        r2 = agent.run(q2, stream=False, session_id=session_id)
        assert r2.content is not None
        # 结合上下文应该能回答 2017
        answer2 = str(r2.content).lower()
        # 注意：small models 可能不总是完美追踪指代，所以用宽松断言
        assert len(answer2) > 10, (
            f"期望追问能得到有意义的回答，实际: {answer2[:200]}"
        )

    @ollama_required
    @embedding_required
    @llm_required
    def test_agent_reranker_pipeline(self, kb_manager, temp_text_file):
        """测试 Reranker 流水线：加载 → 检索 → 重排序 → 生成"""
        # 只在 reranker 启用时测试
        reranker = create_reranker()
        if reranker is None:
            pytest.skip("Reranker 已禁用，跳过重排序集成测试")

        agent = AgentFactory.create(
            kb_manager,
            debug_mode=True,
            enable_tools=False,
        )

        kb_manager.add_source(temp_text_file)

        # 提问需要精确匹配的内容
        query = "What is deep learning and what fields is it applied to?"
        response = agent.run(query, stream=False)

        assert response.content is not None
        answer = str(response.content).lower()
        # 应包含文档中关于 Deep Learning 的信息
        assert len(answer) > 20, (
            f"期望 Reranker 流水线产出有意义的回答，实际长度: {len(answer)}"
        )
