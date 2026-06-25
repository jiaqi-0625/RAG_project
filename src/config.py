"""
集中管理全局配置，支持环境变量覆盖。

所有硬编码的参数（模型名、路径、端口等）统一通过此模块读取，
方便在不同环境间切换，也方便后续 Docker 化。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")


class Config:
    """应用全局配置"""

    # --- 项目路径 ---
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # --- Ollama 服务 ---
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # --- Embedding 模型 ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma:latest")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

    # --- 生成模型 (LLM) ---
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:latest")

    # --- 向量数据库 (LanceDB) ---
    LANCEDB_URI: str = os.getenv("LANCEDB_URI", str(PROJECT_ROOT / "data" / "lancedb"))
    LANCEDB_TABLE_NAME: str = os.getenv("LANCEDB_TABLE_NAME", "knowledge_base")

    # --- Streamlit ---
    STREAMLIT_PORT: int = int(os.getenv("STREAMLIT_PORT", "8501"))
    STREAMLIT_TITLE: str = os.getenv(
        "STREAMLIT_TITLE", "Agentic RAG — 100% 本地知识库问答系统"
    )

    # --- 文档处理 ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # --- Agent ---
    AGENT_DEBUG_MODE: bool = os.getenv("AGENT_DEBUG_MODE", "false").lower() == "true"

    # --- 模型 keep_alive ---
    KEEP_ALIVE: str = os.getenv("KEEP_ALIVE", "5m")

    # --- Reranker ---
    # Cross-Encoder 模型名称，默认使用轻量英文模型 (~87MB)
    # 多语言场景可切换为: BAAI/bge-reranker-v2-m3
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    # 重排序后保留的文档数
    RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "5"))
    # 初始向量检索数量（重排序前的候选集大小）
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))
    # 是否启用 reranker
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"

    # --- 文件上传 ---
    UPLOAD_DIR: Path = Path(
        os.getenv("UPLOAD_DIR", str(PROJECT_ROOT / "data" / "uploads"))
    )

    # --- 工具调用 (Function Calling) ---
    # 是否启用 Agent 工具调用能力（搜索网页、计算等）
    ENABLE_TOOLS: bool = os.getenv("ENABLE_TOOLS", "true").lower() == "true"
    # 排除的工具列表（逗号分隔），例如不想让 Agent 联网可设为 "search_web"
    TOOLS_EXCLUDE: str = os.getenv("TOOLS_EXCLUDE", "")

    # --- ReAct 推理可视化 ---
    # 是否在 UI 中展示 Agent 的 "思考→行动→观察→回答" 推理过程
    SHOW_REACT_PROCESS: bool = (
        os.getenv("SHOW_REACT_PROCESS", "true").lower() == "true"
    )

    # --- 对话记忆 ---
    # 是否启用多轮对话记忆（通过 session 持久化）
    ENABLE_CONVERSATION_MEMORY: bool = (
        os.getenv("ENABLE_CONVERSATION_MEMORY", "true").lower() == "true"
    )
    # 每次请求携带的历史轮次数（0 = 所有历史）
    NUM_HISTORY_RUNS: int = int(os.getenv("NUM_HISTORY_RUNS", "10"))
    # SQLite 数据库路径（用于持久化 session 和对话历史）
    DB_PATH: str = os.getenv(
        "DB_PATH", str(PROJECT_ROOT / "data" / "sessions.db")
    )


# 单例实例
config = Config()
