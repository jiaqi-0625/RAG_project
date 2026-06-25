"""
Agent 工厂 — 创建和配置 RAG Agent。

封装 Agno Agent 的初始化逻辑，
让上层（Streamlit / FastAPI）无需关心 Agent 的配置细节。

支持：
- 多轮对话记忆（session + add_history_to_context）
- SQLite 持久化会话历史
- 可切换 LLM 模型
- 工具调用 (Function Calling)：搜索网页、计算、获取时间等
- ReAct 推理循环：思考→行动→观察→回答 的可视化
"""

from collections.abc import Callable

from agno.agent import Agent
from agno.db.sqlite.sqlite import SqliteDb
from agno.models.ollama import Ollama
from agno.tools.function import Function

from .config import config
from .knowledge_base import KnowledgeBaseManager
from .tools import DEFAULT_TOOLS, get_tools


class AgentFactory:
    """
    Agent 工厂类。

    Usage:
        kb_manager = KnowledgeBaseManager()
        agent = AgentFactory.create(kb_manager)
        response = agent.run("什么是 RAG？", session_id="my-session")
    """

    # 默认系统指令
    DEFAULT_INSTRUCTIONS = [
        # 语言指令
        "You MUST respond in the same language as the user's question. "
        "If the user writes in Chinese (中文), respond in Chinese. "
        "If the user writes in English, respond in English. "
        "This is the most important rule — never respond in English to a Chinese question.",
        # RAG 核心
        "Search the knowledge base for relevant information and base your answers on it.",
        "Be clear, and generate well-structured answers.",
        "Use clear headings, bullet points, or numbered lists where appropriate.",
        "If the answer is not found in the knowledge base, say so honestly.",
        # 对话记忆相关指令
        "When the user asks a follow-up question, use the conversation history "
        "to understand the context and provide coherent answers.",
        "If the user refers to previous answers (e.g., '刚才提到的...', '你之前说的...'), "
        "reference the conversation history to understand what they mean.",
        # 工具调用相关指令
        "You have access to tools (web search, calculator, clock). "
        "Use them silently when needed — do NOT describe the tool, "
        "do NOT output JSON, do NOT say 'I will now use the X tool'. "
        "Just call the tool, get the result, and weave it naturally into your answer. "
        "The user should not notice you used a tool at all.",
    ]

    @classmethod
    def create(
        cls,
        kb_manager: KnowledgeBaseManager,
        model_name: str | None = None,
        instructions: list[str] | None = None,
        debug_mode: bool | None = None,
        tools: list[Callable | Function | dict] | None = None,
        enable_tools: bool | None = None,
        stream_events: bool | None = None,
        exclude_tools: list[str] | None = None,
    ) -> Agent:
        """
        创建一个支持工具调用和对话记忆的 Agentic RAG Agent。

        Args:
            kb_manager: 知识库管理器实例。
            model_name: LLM 模型名称，默认使用配置中的 LLM_MODEL。
            instructions: 自定义系统指令，不传则使用默认指令。
            debug_mode: 是否开启调试模式。
            tools: 自定义工具列表，不传则使用 DEFAULT_TOOLS。
                   传空列表 [] 可完全禁用工具。
            enable_tools: 是否启用工具调用，默认使用配置中的 ENABLE_TOOLS。
            stream_events: 是否在流式响应中发送事件（用于 ReAct 可视化）。
                           默认使用配置中的 SHOW_REACT_PROCESS。
            exclude_tools: 要排除的工具名称列表，例如 ["search_web"]。

        Returns:
            配置好的 agno Agent 实例。
        """
        # 数据库：用于持久化 session 和对话历史
        db = None
        if config.ENABLE_CONVERSATION_MEMORY:
            try:
                db = SqliteDb(db_file=config.DB_PATH)
            except Exception:
                # SQLite 不可用时降级为纯内存模式（对话历史仍可在单次 session 内工作）
                db = None

        # --- 工具配置 ---
        should_enable_tools = enable_tools if enable_tools is not None else config.ENABLE_TOOLS
        resolved_tools: list | None = None

        if should_enable_tools:
            if tools is not None:
                # 用户显式传入了工具列表
                resolved_tools = list(tools)
            else:
                # 使用默认工具，应用排除列表
                exclude = exclude_tools or _parse_exclude_list(config.TOOLS_EXCLUDE)
                resolved_tools = get_tools(exclude=exclude) if exclude else list(DEFAULT_TOOLS)

        # --- 流式事件配置 ---
        # stream_events=True 时 Agno 会在常规内容流之外额外发送工具调用
        # 和推理事件。不覆盖 events_to_skip，使用 Agno 默认值（跳过
        # RunContent 事件，因为内容已通过常规 stream=True 通道输出）。
        should_stream_events = (
            stream_events if stream_events is not None else config.SHOW_REACT_PROCESS
        )

        return Agent(
            model=Ollama(
                id=model_name or config.LLM_MODEL,
            ),
            knowledge=kb_manager.knowledge,
            instructions=instructions or cls.DEFAULT_INSTRUCTIONS,
            search_knowledge=True,
            debug_mode=(debug_mode if debug_mode is not None else config.AGENT_DEBUG_MODE),
            markdown=True,
            # --- 工具调用 ---
            tools=resolved_tools,
            # --- ReAct 推理流式事件（思考→行动→观察→回答）---
            stream_events=should_stream_events,
            # events_to_skip 不传，让 Agno 使用默认值 [RunEvent.run_content]
            # 这样内容走常规 stream 通道，工具/推理事件走事件通道，互不干扰
            # --- 对话记忆 ---
            db=db,
            add_history_to_context=config.ENABLE_CONVERSATION_MEMORY,
            num_history_runs=config.NUM_HISTORY_RUNS,
        )

    @classmethod
    def create_with_custom_model(
        cls,
        kb_manager: KnowledgeBaseManager,
        model_id: str,
        instructions: list[str] | None = None,
        **kwargs,
    ) -> Agent:
        """
        使用自定义模型创建 Agent。

        方便切换不同 LLM 做对比实验，比如：
          - "llama3.2:latest"
          - "qwen2.5:latest"
          - "deepseek-r1:latest"

        **kwargs 透传给 create()，支持 tools / enable_tools / stream_events 等参数。
        """
        return cls.create(
            kb_manager=kb_manager,
            model_name=model_id,
            instructions=instructions,
            **kwargs,
        )


def _parse_exclude_list(raw: str) -> list[str]:
    """解析逗号分隔的排除列表字符串（来自环境变量）"""
    if not raw or not raw.strip():
        return []
    return [name.strip() for name in raw.split(",") if name.strip()]
