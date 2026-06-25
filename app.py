"""
app.py — Streamlit 入口（纯 UI 层）。

业务逻辑全部委托给 src/ 下的模块：
- config.py       → 配置管理
- embedder.py     → 嵌入模型
- vector_store.py → 向量数据库
- document_loader.py → 文档加载
- knowledge_base.py → 知识库管理
- agent.py        → Agent 工厂
- reranker.py     → 重排序模块
- memory.py       → 对话记忆（session + 聊天历史）

UI 负责：页面布局 + 用户交互 + 流式渲染 + 多轮对话管理。
"""

from datetime import UTC, datetime
import hashlib
import json
import logging
from pathlib import Path

import streamlit as st

from src.agent import AgentFactory
from src.config import config
from src.embedder import EmbeddingModelError, OllamaConnectionError
from src.knowledge_base import KnowledgeBaseManager
from src.memory import ChatHistory, SessionManager
from src.reranker import RerankerError, create_reranker
from src.tools import get_tools
from src.vector_store import VectorStoreConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Agentic RAG — 100% 本地知识库问答系统",
    page_icon="🔥",
    layout="wide",
)


# ============================================================
# 初始化全局状态（cached — 只创建一次）
# ============================================================
@st.cache_resource
def init_knowledge_base() -> KnowledgeBaseManager:
    """惰性初始化知识库（Streamlit cache_resource 确保只加载一次）"""
    try:
        reranker = create_reranker()
        if reranker:
            logger.info(f"已启用 Reranker: {reranker.model_name}")
        else:
            logger.info("Reranker 已禁用。")
        return KnowledgeBaseManager(reranker=reranker)
    except RerankerError as e:
        st.warning(f"⚠️ Reranker 模型加载失败，回退到无重排序模式: {e}")
        return KnowledgeBaseManager(reranker=None)
    except VectorStoreConnectionError as e:
        st.error(f"❌ 向量数据库初始化失败: {e}")
        raise


# --- 持久化 session 状态 ---
if "urls" not in st.session_state:
    st.session_state.urls = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []  # list of (display_name, saved_path)

# --- 对话记忆状态 ---
if "session_manager" not in st.session_state:
    st.session_state.session_manager = SessionManager()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatHistory()

# --- 工具选择状态（默认全部启用，用户可在侧边栏切换）---
if "tool_selections" not in st.session_state:
    st.session_state.tool_selections = {
        "search_web": True,
        "calculate": True,
        "get_current_time": True,
    }

kb_manager = init_knowledge_base()

# 加载尚未处理过的 URL（防止 rerun 时重复加载）
for url in st.session_state.urls:
    if url not in kb_manager.loaded_sources:
        try:
            kb_manager.add_source(url)
        except Exception as e:
            st.warning(f"加载失败 {url}: {e}")

# 加载尚未处理过的上传文件
for display_name, saved_path in st.session_state.uploaded_files:
    if saved_path not in kb_manager.loaded_sources:
        try:
            kb_manager.add_source(saved_path)
        except Exception as e:
            st.warning(f"加载失败 {display_name}: {e}")

# 创建 Agent：根据用户在侧边栏的工具选择动态配置
selected_tools = [name for name, enabled in st.session_state.tool_selections.items() if enabled]
agent = AgentFactory.create(
    kb_manager,
    tools=get_tools(include=selected_tools) if selected_tools else [],
)


# ============================================================
# Convenience helpers
# ============================================================
def get_session_id() -> str:
    """获取当前对话的 session_id"""
    return str(st.session_state.session_manager.current_session_id)


def start_new_session() -> None:
    """开始全新对话"""
    st.session_state.session_manager.new_session()
    st.session_state.chat_history.clear()


# ============================================================
# ReAct 推理可视化 — 将 Agno 流式事件渲染为可视化步骤
# ============================================================
def _render_react_steps(steps: list) -> str:
    """
    将 ReAct 步骤列表渲染为精简的单行 Markdown 列表。
    不展示原始 JSON 参数和冗长返回值，只给一句话摘要。
    """
    if not steps:
        return ""

    lines = ["#### 🧠 推理过程", ""]

    for i, step in enumerate(steps):
        icon = step.get("icon", "➡️")
        label = step.get("label", "")
        detail = step.get("detail", "").strip()

        # 只保留一行摘要
        if detail:
            # 截断到第一行 / 80 字符
            summary = detail.split("\n")[0][:80]
            lines.append(f"{i + 1}. {icon} {label} — {summary}")
        else:
            lines.append(f"{i + 1}. {icon} {label}")

    return "\n".join(lines)


# ============================================================
# Sidebar — 知识来源管理 + 对话控制
# ============================================================
with st.sidebar:
    # Logo 区
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            "https://www.gstatic.com/lamda/images/gemma_sparkle_v2_c00210c455.svg",
            width=64,
        )
    with col2:
        st.image(
            "https://ollama.com/public/ollama.png",
            width=64,
        )
    with col3:
        st.image(
            "https://raw.githubusercontent.com/agno-agi/agno/main/docs/assets/agno-logo.png",
            width=64,
        )

    # --- URL 添加 ---
    st.header("🌐 添加网页来源")
    new_url = st.text_input(
        "输入文档 URL",
        placeholder="https://example.com/paper.pdf",
        help="支持 PDF 链接和网页链接，系统会自动下载并解析。",
    )

    if st.button("➕ 添加 URL", type="primary"):
        if new_url:
            if new_url not in st.session_state.urls:
                st.session_state.urls.append(new_url)
                with st.spinner("📥 正在处理文档……"):
                    try:
                        kb_manager.add_source(new_url)
                        st.success(f"✅ 已添加: {new_url}")
                    except Exception as e:
                        st.error(f"添加失败: {e}")
                st.rerun()
            else:
                st.warning("该 URL 已经添加过了。")
        else:
            st.error("请输入一个 URL。")

    st.divider()

    # --- 本地文件上传 ---
    st.header("📁 上传本地文件")
    uploaded_file = st.file_uploader(
        "选择文件（拖拽或点击）",
        type=["pdf", "txt", "md", "markdown", "docx", "html", "htm"],
        help="支持格式：PDF、Word (.docx)、HTML、TXT、Markdown",
    )

    if uploaded_file is not None:
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        saved_path = config.UPLOAD_DIR / uploaded_file.name

        if saved_path.exists():
            file_hash = hashlib.md5(uploaded_file.getbuffer()).hexdigest()[:8]
            stem = Path(uploaded_file.name).stem
            suffix = Path(uploaded_file.name).suffix
            saved_path = config.UPLOAD_DIR / f"{stem}_{file_hash}{suffix}"

        str_path = str(saved_path)
        already_added = any(str_path == sp for _, sp in st.session_state.uploaded_files)

        if not already_added:
            if not saved_path.exists():
                with open(saved_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            st.session_state.uploaded_files.append((uploaded_file.name, str_path))
            with st.spinner(f"📥 正在解析 {uploaded_file.name}……"):
                try:
                    kb_manager.add_source(str_path)
                    st.success(f"✅ 已加载: {uploaded_file.name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"加载失败: {e}")
                    st.session_state.uploaded_files.pop()
        else:
            st.info(f"📁 {uploaded_file.name} 已经在知识库中。")

    # --- 已加载来源列表 ---
    has_urls = bool(st.session_state.urls)
    has_files = bool(st.session_state.uploaded_files)

    if has_urls or has_files:
        st.subheader("📚 当前知识来源")
        idx = 1

        if has_urls:
            for url in st.session_state.urls:
                loaded = "✅" if url in kb_manager.loaded_sources else "⏳"
                st.markdown(f"{idx}. {loaded} 🌐 {url}")
                idx += 1

        if has_files:
            for display_name, saved_path in st.session_state.uploaded_files:
                loaded = "✅" if saved_path in kb_manager.loaded_sources else "⏳"
                st.markdown(f"{idx}. {loaded} 📁 {display_name}")
                idx += 1

    st.divider()

    # --- 工具配置 (Function Calling) ---
    st.header("🔧 工具配置")
    st.caption("选择 Agent 可使用的工具能力")

    tool_changed = False
    new_selections = dict(st.session_state.tool_selections)

    # search_web
    web_enabled = st.checkbox(
        "🌐 网页搜索",
        value=st.session_state.tool_selections.get("search_web", True),
        help="通过 DuckDuckGo 搜索实时信息（免费，无需 API Key）",
    )
    if web_enabled != st.session_state.tool_selections.get("search_web"):
        new_selections["search_web"] = web_enabled
        tool_changed = True

    # calculate
    calc_enabled = st.checkbox(
        "🧮 数学计算",
        value=st.session_state.tool_selections.get("calculate", True),
        help="安全求值数学表达式，支持 sqrt/sin/cos/log 等函数",
    )
    if calc_enabled != st.session_state.tool_selections.get("calculate"):
        new_selections["calculate"] = calc_enabled
        tool_changed = True

    # get_current_time
    time_enabled = st.checkbox(
        "🕐 获取时间",
        value=st.session_state.tool_selections.get("get_current_time", True),
        help="查询当前日期和时间",
    )
    if time_enabled != st.session_state.tool_selections.get("get_current_time"):
        new_selections["get_current_time"] = time_enabled
        tool_changed = True

    if tool_changed:
        st.session_state.tool_selections = new_selections
        st.rerun()

    # 显示当前启用的工具数量
    enabled_count = sum(1 for v in st.session_state.tool_selections.values() if v)
    total_count = len(st.session_state.tool_selections)
    if enabled_count == 0:
        st.warning("⚠️ 未启用任何工具，Agent 将仅使用知识库检索。")
    elif enabled_count == total_count:
        st.success(f"✅ 全部 {total_count} 个工具已启用")
    else:
        st.info(f"🔧 已启用 {enabled_count}/{total_count} 个工具")

    st.divider()

    # --- 对话控制 ---
    st.header("💬 对话控制")
    st.caption(f"当前 Session：`{get_session_id()}`")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🆕 新对话", use_container_width=True):
            start_new_session()
            st.rerun()
    with col_b:
        if st.button("🗑️ 清空历史", use_container_width=True):
            st.session_state.chat_history.clear()
            st.rerun()

    # 记忆状态指示器
    mem_enabled_global = getattr(config, "ENABLE_CONVERSATION_MEMORY", True)
    if mem_enabled_global:
        try:
            mem_enabled = agent.db is not None
        except Exception:
            mem_enabled = False
        if mem_enabled:
            st.success("🧠 对话记忆：已启用（SQLite 持久化）")
        else:
            st.warning("🧠 对话记忆：内存模式（重启丢失）")
    else:
        st.info("🧠 对话记忆：已禁用")

    # 对话轮次统计
    if not st.session_state.chat_history.empty:
        turns = len([m for m in st.session_state.chat_history.messages if m.role == "user"])
        st.caption(f"📊 当前对话已进行 {turns} 轮")

    st.divider()

    # --- 历史会话 ---
    st.header("📜 历史会话")

    # 刷新按钮
    if "session_list" not in st.session_state:
        st.session_state.session_list = []

    if st.button("🔄 刷新列表", use_container_width=True):
        st.session_state.session_list = st.session_state.session_manager.list_sessions(agent)
        st.rerun()

    # 首次加载时自动刷新
    if not st.session_state.session_list:
        st.session_state.session_list = st.session_state.session_manager.list_sessions(agent)

    sessions = st.session_state.session_list

    if not sessions:
        st.caption("暂无历史会话。开始对话后会自动保存。")
    else:
        sm = st.session_state.session_manager
        for i, s in enumerate(sessions):
            # 格式化时间
            ts = s.updated_at or s.created_at
            if ts:
                dt = datetime.fromtimestamp(ts, tz=UTC)
                time_str = dt.strftime("%m/%d %H:%M")
            else:
                time_str = "未知时间"

            # 当前会话标记
            marker = "🟢" if s.is_current else "  "
            sid_short = s.session_id[-12:]

            # 行：标记 + 名称 + 时间
            st.markdown(f"{marker} **{s.session_name}**")
            st.caption(f"  🕒 {time_str} · {s.message_count} 条消息 · `{sid_short}`")

            col_sw, col_rn, col_dl = st.columns([1, 1, 1])
            with col_sw:
                if not s.is_current:
                    if st.button("📂 切换", key=f"sw_{i}", use_container_width=True):
                        sm.load_session(agent, s.session_id, st.session_state.chat_history)
                        st.session_state.session_list = sm.list_sessions(agent)
                        st.rerun()
                else:
                    st.button("✅ 当前", key=f"cur_{i}", disabled=True, use_container_width=True)

            with col_rn:
                new_name = st.text_input(
                    "重命名",
                    value=s.session_name,
                    key=f"rn_{s.session_id}",
                    label_visibility="collapsed",
                )
                if new_name and new_name != s.session_name:
                    sm.rename_session(agent, s.session_id, new_name)
                    st.session_state.session_list = sm.list_sessions(agent)
                    st.rerun()

            with col_dl:
                if st.button("🗑️ 删除", key=f"dl_{i}", use_container_width=True):
                    sm.delete_session(agent, s.session_id)
                    if s.is_current:
                        st.session_state.chat_history.clear()
                    st.session_state.session_list = sm.list_sessions(agent)
                    st.rerun()


# ============================================================
# Main Area — 对话式问答界面
# ============================================================
st.title("🔥 Agentic RAG — 100% 本地知识库问答")
st.caption("完全本地运行 · 无需 API Key · 无需联网 · 数据不出本机")

# 欢迎信息（仅在无对话历史时显示）
if st.session_state.chat_history.empty:
    st.markdown(
        """
    本系统演示了一个**完全本地化**的 Agentic RAG（检索增强生成）系统：

    - **EmbeddingGemma** — Google 轻量级嵌入模型（768 维），将文本转为向量
    - **LanceDB** — 嵌入式向量数据库，零配置、零运维
    - **Cross-Encoder Reranker** — 对检索结果进行精排，大幅提升答案准确性
    - **Llama 3.2** — 本地大语言模型，负责生成自然语言回答
    - **Agno 框架** — Agent 编排引擎，协调检索与生成全流程
    - **🧠 对话记忆** — 多轮对话上下文保持，支持追问和深入讨论

    在左侧边栏添加文档 URL 或上传本地文件，即可开始构建知识库并提问。
        """
    )

# --- 渲染聊天历史 ---
for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.role):
        st.markdown(msg.content)

# --- 输入区 ---
query = st.chat_input(
    placeholder="输入你的问题，支持多轮追问……",
)

if query:
    # 添加用户消息到历史
    st.session_state.chat_history.add_user_message(query)

    # 渲染用户消息
    with st.chat_message("user"):
        st.markdown(query)

    # --- 调用 Agent 并流式渲染回答（含 ReAct 推理可视化）---
    with st.chat_message("assistant"):
        response = ""
        react_steps: list = []  # 收集 ReAct 步骤
        react_placeholder = st.empty()  # ReAct 可视化容器
        resp_container = st.empty()  # 回答内容容器

        try:
            for chunk in agent.run(
                query,
                stream=True,
                session_id=get_session_id(),
            ):
                event = getattr(chunk, "event", "")

                # ── ReAct: 工具调用开始 ──
                if event == "ToolCallStarted" and chunk.tools:
                    for tool in chunk.tools:
                        if tool.tool_name:
                            args_str = (
                                json.dumps(tool.tool_args, ensure_ascii=False, indent=2)
                                if tool.tool_args
                                else ""
                            )
                            react_steps.append(
                                {
                                    "phase": "act",
                                    "icon": "🔧",
                                    "label": f"调用工具: `{tool.tool_name}`",
                                    "detail": args_str,
                                }
                            )

                # ── ReAct: 工具调用完成 → 观察结果 ──
                elif event == "ToolCallCompleted" and chunk.tools:
                    for tool in chunk.tools:
                        if tool.tool_name:
                            result_text = tool.result or ""
                            error = tool.tool_call_error
                            phase = "error" if error else "observe"
                            icon = "❌" if error else "👁️"
                            react_steps.append(
                                {
                                    "phase": phase,
                                    "icon": icon,
                                    "label": f"工具返回: `{tool.tool_name}`",
                                    "detail": result_text,
                                }
                            )

                # ── ReAct: 推理步骤 ──
                elif event in ("ReasoningStep", "ReasoningStarted"):
                    reasoning_content = getattr(chunk, "content", None)
                    react_steps.append(
                        {
                            "phase": "think",
                            "icon": "💭",
                            "label": "推理思考",
                            "detail": str(reasoning_content or "分析中..."),
                        }
                    )

                # ── ReAct: 推理完成 ──
                elif event == "ReasoningCompleted":
                    react_steps.append(
                        {
                            "phase": "think",
                            "icon": "✅",
                            "label": "推理完成",
                            "detail": "",
                        }
                    )

                # ── 更新 ReAct 可视化 ──
                if config.SHOW_REACT_PROCESS and react_steps:
                    react_placeholder.markdown(_render_react_steps(react_steps))

                # ── 累积回答内容 ──
                # 只收集常规内容块，跳过工具/推理等事件块（它们可能也带 content，
                # 但那些是事件元数据，不是给用户看的回答文本）
                if chunk.content is not None and event in ("", "RunContent"):
                    response += chunk.content
                    resp_container.markdown(response)

        except OllamaConnectionError as e:
            st.error(f"❌ Ollama 服务连接失败: {e}")
            logger.exception("Ollama 连接失败")
            response = f"⚠️ Ollama 服务不可用：{e}"
        except EmbeddingModelError as e:
            st.error(f"❌ 嵌入模型错误: {e}")
            logger.exception("嵌入模型错误")
            response = f"⚠️ 嵌入模型出错：{e}"
        except Exception as e:
            st.error(f"生成回答时出错: {e}")
            logger.exception("Agent 运行失败")
            response = f"⚠️ 出错了：{e}"

    # 将助手回答保存到历史
    if response:
        st.session_state.chat_history.add_assistant_message(response)


# ============================================================
# Footer — 系统原理说明
# ============================================================
with st.expander("📖 系统工作原理"):
    st.markdown(
        """
### 架构数据流

```
用户提问
  │
  ├─→ EmbeddingGemma → 查询向量
  │         │
  │         └─→ LanceDB（向量检索）→ Top-20 候选文档块
  │                                           │
  │                                           └─→ CrossEncoder Reranker（精排）→ Top-5 最佳块
  │                                                                                   │
  └─→ Llama 3.2 ← 上下文（精排后的文档块 + 对话历史 + 用户问题）────────────────────┘
          │
          └─→ 流式输出回答
```

### 核心组件

| 组件 | 作用 |
|------|------|
| **EmbeddingGemma** | 将文本转换为 768 维向量表示 |
| **LanceDB** | 存储和检索文档向量（嵌入式、零运维） |
| **CrossEncoder Reranker** | 对候选文档块二次精排，提升检索精度 |
| **Llama 3.2** | 基于上下文生成自然语言回答 |
| **Agno Agent** | 智能编排检索与生成的全流程 |
| **🧠 对话记忆** | SQLite 持久化 session，支持多轮追问 |

### 对话记忆工作原理

1. 每次对话使用唯一的 `session_id` 标识
2. Agent 设置 `add_history_to_context=True`，自动将历史消息注入上下文
3. Session 通过 SQLite 持久化到 `data/sessions.db`，重启后仍可继续对话
4. 点击"新对话"生成新 session，历史对话独立保存

### 支持的文档格式

| 格式 | 来源 |
|------|------|
| PDF | URL 链接 / 本地上传 |
| Word (.docx) | 本地上传 |
| HTML | URL 链接 / 本地上传 |
| TXT / Markdown | 本地上传 |

### 为什么选择 100% 本地部署？

- 🔒 **数据隐私**：文档永远不会离开你的机器
- 💰 **零成本**：无需支付任何 API 调用费用
- ⚡ **低延迟**：无网络往返，响应更快
- 🌐 **离线可用**：不依赖互联网连接
- 🏢 **企业就绪**：适合内网部署、合规要求严格的场景
        """
    )
