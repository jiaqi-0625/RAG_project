"""
app.py — Streamlit 入口（纯 UI 层）。

DeepQuery — 100% 本地 Agentic RAG 知识库问答系统
ChatGPT 风格布局 · 浅色立体设计 · 品牌化 UI

业务逻辑全部委托给 src/ 下的模块。
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
    page_title="DeepQuery — 本地智能，深度问答",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局 CSS — 浅色立体设计系统
# ============================================================
st.markdown(
    """
<style>
    /* ============================================================
       DeepQuery Design System — Global CSS
       色彩: Indigo + Cyan, 浅色背景, 立体阴影层级
       ============================================================ */

    /* ── Root variables ── */
    :root {
        --dp-primary: #4F46E5;
        --dp-primary-light: #818CF8;
        --dp-primary-soft: #EEF2FF;
        --dp-accent: #06B6D4;
        --dp-accent-soft: #ECFEFF;
        --dp-success: #10B981;
        --dp-success-soft: #ECFDF5;
        --dp-warning: #F59E0B;
        --dp-warning-soft: #FFFBEB;
        --dp-error: #EF4444;
        --dp-error-soft: #FEF2F2;
        --dp-bg: #F1F5F9;
        --dp-surface: #FFFFFF;
        --dp-border: #E2E8F0;
        --dp-border-light: #F1F5F9;
        --dp-text: #1E293B;
        --dp-text-secondary: #64748B;
        --dp-text-muted: #94A3B8;
        --dp-shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --dp-shadow-md: 0 4px 6px -1px rgba(0,0,0,0.07), 0 2px 4px -2px rgba(0,0,0,0.05);
        --dp-shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -4px rgba(0,0,0,0.04);
        --dp-shadow-xl: 0 20px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.04);
        --dp-radius-sm: 8px;
        --dp-radius-md: 12px;
        --dp-radius-lg: 16px;
        --dp-radius-xl: 24px;
    }

    /* ── Global page background ── */
    .stApp {
        background-color: var(--dp-bg);
    }
    [data-testid="stAppViewContainer"] {
        background-color: var(--dp-bg);
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
        max-width: 860px;
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background-color: var(--dp-bg);
        border-right: 1px solid var(--dp-border);
    }
    [data-testid="stSidebar"] .block-container {
        padding: 1rem 1rem 1rem 1rem;
    }
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* ── Sidebar brand header ── */
    .dp-sidebar-brand {
        background: linear-gradient(135deg, var(--dp-primary), #6366F1);
        border-radius: var(--dp-radius-md);
        padding: 16px 16px 14px 16px;
        margin-bottom: 14px;
        color: #FFFFFF;
        box-shadow: var(--dp-shadow-md);
    }
    .dp-sidebar-brand .dp-brand-name {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .dp-sidebar-brand .dp-brand-tagline {
        font-size: 0.72rem;
        opacity: 0.85;
        margin-top: 2px;
    }
    .dp-sidebar-brand .dp-brand-status {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        margin-top: 8px;
        font-size: 0.7rem;
        background: rgba(255,255,255,0.2);
        padding: 2px 10px;
        border-radius: 20px;
    }
    .dp-status-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #10B981;
        display: inline-block;
        box-shadow: 0 0 6px rgba(16,185,129,0.6);
    }

    /* ── Sidebar section cards ── */
    .dp-sidebar-section {
        background: var(--dp-surface);
        border: 1px solid var(--dp-border);
        border-radius: var(--dp-radius-md);
        padding: 14px 14px 10px 14px;
        margin-bottom: 10px;
        box-shadow: var(--dp-shadow-sm);
    }
    .dp-sidebar-section h4 {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--dp-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0 0 8px 0;
    }

    /* ── Sidebar loaded source item ── */
    .dp-source-item {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 5px 0;
        font-size: 0.78rem;
        color: var(--dp-text-secondary);
        border-bottom: 1px solid var(--dp-border-light);
    }
    .dp-source-item:last-child { border-bottom: none; }

    /* ── Chat message bubbles (override Streamlit defaults) ── */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        padding: 4px 0 !important;
    }
    [data-testid="stChatMessage"] > div:first-child {
        display: none !important;
    }

    /* User message bubble — right-aligned feel, indigo tint */
    [data-testid="stChatMessage"][data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
        background: var(--dp-primary-soft) !important;
        border: 1px solid #C7D2FE !important;
        border-radius: var(--dp-radius-lg) var(--dp-radius-lg) 4px var(--dp-radius-lg) !important;
        padding: 12px 16px !important;
        margin: 4px 0 4px auto !important;
        max-width: 80% !important;
        box-shadow: var(--dp-shadow-sm) !important;
        color: var(--dp-text) !important;
    }

    /* AI message bubble — white card with shadow */
    div[data-testid="stChatMessage"]:has(> .stChatMessageAvatarWrapper) div[data-testid="stMarkdownContainer"],
    .stChatMessage.stChatMessage--ai div[data-testid="stMarkdownContainer"] {
        background: var(--dp-surface) !important;
        border: 1px solid var(--dp-border) !important;
        border-radius: var(--dp-radius-lg) var(--dp-radius-lg) var(--dp-radius-lg) 4px !important;
        padding: 12px 16px !important;
        margin-right: auto !important;
        max-width: 85% !important;
        box-shadow: var(--dp-shadow-md) !important;
        color: var(--dp-text) !important;
    }

    /* ── Chat input styling — pill shape ── */
    [data-testid="stChatInput"] {
        position: sticky;
        bottom: 0;
        background: linear-gradient(to top, var(--dp-bg) 80%, transparent);
        padding: 16px 0 8px 0;
    }
    [data-testid="stChatInput"] textarea {
        border: 2px solid var(--dp-border) !important;
        border-radius: var(--dp-radius-xl) !important;
        padding: 12px 20px !important;
        box-shadow: var(--dp-shadow-md) !important;
        transition: all 0.2s ease;
        background: var(--dp-surface) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--dp-primary-light) !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.15), var(--dp-shadow-md) !important;
    }

    /* ── Welcome hero ── */
    .dp-hero {
        text-align: center;
        padding: 3rem 1rem 2rem 1rem;
    }
    .dp-hero-logo {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 6px rgba(79,70,229,0.2));
    }
    .dp-hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--dp-text);
        letter-spacing: -0.03em;
        margin: 0;
    }
    .dp-hero-subtitle {
        font-size: 1rem;
        color: var(--dp-text-secondary);
        margin-top: 4px;
    }

    /* ── Feature cards (welcome page) ── */
    .dp-feature-grid {
        display: flex;
        gap: 12px;
        margin: 24px 0;
        flex-wrap: wrap;
        justify-content: center;
    }
    .dp-feature-card {
        flex: 1 1 150px;
        max-width: 200px;
        background: var(--dp-surface);
        border: 1px solid var(--dp-border);
        border-radius: var(--dp-radius-md);
        padding: 18px 14px;
        text-align: center;
        box-shadow: var(--dp-shadow-sm);
        transition: all 0.2s ease;
    }
    .dp-feature-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--dp-shadow-lg);
    }
    .dp-feature-card .dp-feature-icon {
        font-size: 2rem;
        margin-bottom: 6px;
    }
    .dp-feature-card .dp-feature-title {
        font-weight: 700;
        font-size: 0.85rem;
        color: var(--dp-text);
    }
    .dp-feature-card .dp-feature-desc {
        font-size: 0.72rem;
        color: var(--dp-text-muted);
        margin-top: 4px;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: var(--dp-radius-sm) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--dp-shadow-sm) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--dp-shadow-md) !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--dp-primary), #6366F1) !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #4338CA, #4F46E5) !important;
    }

    /* ── Expander (footer) ── */
    .streamlit-expanderHeader {
        border-radius: var(--dp-radius-sm) !important;
        font-weight: 600 !important;
        color: var(--dp-text-secondary) !important;
    }

    /* ── Dividers ── */
    hr {
        border-color: var(--dp-border) !important;
        margin: 0.5rem 0 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: var(--dp-border);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--dp-text-muted); }

    /* ── Sidebar buttons ── */
    [data-testid="stSidebar"] .stButton > button {
        font-size: 0.8rem !important;
    }

    /* ── ReAct visualization ── */
    .dp-react-container {
        background: var(--dp-surface);
        border: 1px solid var(--dp-border);
        border-radius: var(--dp-radius-md);
        padding: 12px 16px;
        margin-bottom: 12px;
        box-shadow: var(--dp-shadow-sm);
    }
    .dp-react-container h4 {
        margin: 0 0 8px 0;
        font-size: 0.8rem;
        color: var(--dp-text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .dp-react-step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 0;
        font-size: 0.8rem;
        color: var(--dp-text-secondary);
    }
    .dp-react-step .dp-step-dot {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        flex-shrink: 0;
    }

    /* ── Inline knowledge toolbar (main area, above chat input) ── */
    .dp-knowledge-bar {
        background: var(--dp-surface);
        border: 1px solid var(--dp-border);
        border-radius: var(--dp-radius-md);
        padding: 10px 14px;
        margin: 0 0 8px 0;
        box-shadow: var(--dp-shadow-sm);
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .dp-knowledge-bar .dp-kb-divider {
        width: 1px; height: 24px;
        background: var(--dp-border);
        margin: 0 4px;
    }
    .dp-knowledge-bar .dp-kb-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--dp-text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        white-space: nowrap;
    }
    .dp-knowledge-bar input {
        border-radius: var(--dp-radius-sm) !important;
        font-size: 0.82rem !important;
    }
    .dp-knowledge-bar .stButton > button {
        font-size: 0.78rem !important;
        padding: 4px 12px !important;
    }

    /* ── Sidebar checkbox styling ── */
    [data-testid="stSidebar"] .stCheckbox label {
        font-size: 0.82rem;
    }

    /* ── Toast/Success/Error message styling ── */
    [data-testid="stAlert"] {
        border-radius: var(--dp-radius-sm) !important;
        box-shadow: var(--dp-shadow-sm) !important;
    }

    /* ── Responsive: on small screens, reduce max-width ── */
    @media (max-width: 768px) {
        .main .block-container { max-width: 100%; padding: 1rem; }
        .dp-hero-title { font-size: 1.5rem; }
        .dp-feature-card { flex: 1 1 120px; }
    }
</style>
""",
    unsafe_allow_html=True,
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

# --- 工具选择状态 ---
if "tool_selections" not in st.session_state:
    st.session_state.tool_selections = {
        "search_web": True,
        "calculate": True,
        "get_current_time": True,
    }

# --- 侧边栏折叠状态 ---
if "sidebar_section_loaded" not in st.session_state:
    st.session_state.sidebar_section_loaded = True
if "sidebar_section_tools" not in st.session_state:
    st.session_state.sidebar_section_tools = True
if "sidebar_section_sessions" not in st.session_state:
    st.session_state.sidebar_section_sessions = False

kb_manager = init_knowledge_base()

# ── 从持久化文件恢复 session state（首次加载时）──
if "state_restored" not in st.session_state:
    st.session_state.state_restored = False

if not st.session_state.state_restored:
    # 直接从 JSON 文件读取（避免依赖可能被缓存的 kb_manager 方法）
    _state_file = Path("data/loaded_sources.json")
    if _state_file.exists():
        try:
            _state = json.loads(_state_file.read_text(encoding="utf-8"))
            if not st.session_state.urls and _state.get("urls"):
                st.session_state.urls = _state["urls"]
            if not st.session_state.uploaded_files and _state.get("files"):
                st.session_state.uploaded_files = [(f["name"], f["path"]) for f in _state["files"]]
        except Exception:
            pass  # JSON 损坏时静默跳过
    st.session_state.state_restored = True

# 加载尚未处理过的 URL
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

# 创建 Agent
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


def _render_react_steps(steps: list) -> str:
    """将 ReAct 步骤列表渲染为 HTML 可视化步骤条"""
    if not steps:
        return ""

    phase_colors = {
        "think": ("#818CF8", "#EEF2FF", "💭"),
        "act": ("#06B6D4", "#ECFEFF", "🔧"),
        "observe": ("#10B981", "#ECFDF5", "👁️"),
        "error": ("#EF4444", "#FEF2F2", "❌"),
    }

    parts = [
        '<div class="dp-react-container">',
        "<h4>🧠 推理过程</h4>",
    ]
    for step in steps:
        phase = step.get("phase", "")
        icon = step.get("icon", "➡️")
        label = step.get("label", "")
        detail = step.get("detail", "").strip()

        color_bg, color_bg_soft, _ = phase_colors.get(phase, ("#94A3B8", "#F8FAFC", "➡️"))

        summary = detail.split("\n")[0][:80] if detail else ""

        parts.append('<div class="dp-react-step">')
        parts.append(
            f'<span class="dp-step-dot" style="background:{color_bg_soft};color:{color_bg};">'
            f"{icon}</span>"
        )
        parts.append(f"<span><strong>{label}</strong>")
        if summary:
            parts.append(f'<br><span style="font-size:0.72rem;color:#94A3B8;">{summary}</span>')
        parts.append("</span></div>")

    parts.append("</div>")
    return "\n".join(parts)


# ============================================================
# Sidebar — ChatGPT 风格左侧面板
# ============================================================
with st.sidebar:
    # ── Brand Header Card ──
    st.markdown(
        """
    <div class="dp-sidebar-brand">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:1.6rem;">🔮</span>
            <div>
                <div class="dp-brand-name">DeepQuery</div>
                <div class="dp-brand-tagline">本地智能，深度问答</div>
            </div>
        </div>
        <div class="dp-brand-status">
            <span class="dp-status-dot"></span> Ollama · llama3.2
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Section 1: 已加载来源 ──
    has_urls = bool(st.session_state.urls)
    has_files = bool(st.session_state.uploaded_files)

    if has_urls or has_files:
        st.markdown(
            '<div class="dp-sidebar-section"><h4>📚 已加载文档</h4>',
            unsafe_allow_html=True,
        )
        idx = 1
        if has_urls:
            for url in st.session_state.urls:
                loaded = "✅" if url in kb_manager.loaded_sources else "⏳"
                st.markdown(
                    f'<div class="dp-source-item">'
                    f"<span>{loaded}</span> <span>🌐</span>"
                    f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                    f'{url.split("/")[-1][:30]}</span></div>',
                    unsafe_allow_html=True,
                )
                idx += 1
        if has_files:
            for display_name, saved_path in st.session_state.uploaded_files:
                loaded = "✅" if saved_path in kb_manager.loaded_sources else "⏳"
                st.markdown(
                    f'<div class="dp-source-item">'
                    f"<span>{loaded}</span> <span>📄</span>"
                    f'<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'
                    f"{display_name[:30]}</span></div>",
                    unsafe_allow_html=True,
                )
                idx += 1
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 3: 工具配置 ──
    st.markdown(
        '<div class="dp-sidebar-section"><h4>🔧 工具配置</h4>',
        unsafe_allow_html=True,
    )

    tool_changed = False
    new_selections = dict(st.session_state.tool_selections)

    web_enabled = st.checkbox(
        "🌐 网页搜索",
        value=st.session_state.tool_selections.get("search_web", True),
        help="通过 DuckDuckGo 搜索实时信息",
    )
    if web_enabled != st.session_state.tool_selections.get("search_web"):
        new_selections["search_web"] = web_enabled
        tool_changed = True

    calc_enabled = st.checkbox(
        "🧮 数学计算",
        value=st.session_state.tool_selections.get("calculate", True),
        help="安全求值数学表达式",
    )
    if calc_enabled != st.session_state.tool_selections.get("calculate"):
        new_selections["calculate"] = calc_enabled
        tool_changed = True

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

    enabled_count = sum(1 for v in st.session_state.tool_selections.values() if v)
    total_count = len(st.session_state.tool_selections)
    if enabled_count == 0:
        st.warning("⚠️ 未启用工具")
    elif enabled_count == total_count:
        st.success(f"✅ 全部 {total_count} 个已启用")
    else:
        st.info(f"🔧 {enabled_count}/{total_count} 已启用")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 4: 对话控制 ──
    st.markdown(
        '<div class="dp-sidebar-section"><h4>💬 对话</h4>',
        unsafe_allow_html=True,
    )

    mem_enabled_global = getattr(config, "ENABLE_CONVERSATION_MEMORY", True)
    if mem_enabled_global:
        try:
            mem_enabled = agent.db is not None
        except Exception:
            mem_enabled = False
        status_label = "🧠 记忆已启用" if mem_enabled else "🧠 内存模式"
        st.caption(status_label)

    if not st.session_state.chat_history.empty:
        turns = len([m for m in st.session_state.chat_history.messages if m.role == "user"])
        st.caption(f"📊 已对话 {turns} 轮")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🆕 新对话", use_container_width=True, key="btn_new_chat"):
            start_new_session()
            st.rerun()
    with col_b:
        if st.button("🗑️ 清空", use_container_width=True, key="btn_clear"):
            st.session_state.chat_history.clear()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Section 5: 历史会话 ──
    st.markdown(
        '<div class="dp-sidebar-section"><h4>📜 历史会话</h4>',
        unsafe_allow_html=True,
    )

    if "session_list" not in st.session_state:
        st.session_state.session_list = []

    if st.button("🔄 刷新列表", use_container_width=True, key="btn_refresh"):
        st.session_state.session_list = st.session_state.session_manager.list_sessions(agent)
        st.rerun()

    if not st.session_state.session_list:
        st.session_state.session_list = st.session_state.session_manager.list_sessions(agent)

    sessions = st.session_state.session_list

    if not sessions:
        st.caption("暂无历史会话")
    else:
        sm = st.session_state.session_manager
        for i, s in enumerate(sessions):
            ts = s.updated_at or s.created_at
            if ts:
                dt = datetime.fromtimestamp(ts, tz=UTC)
                time_str = dt.strftime("%m/%d %H:%M")
            else:
                time_str = "未知"

            marker = "🟢" if s.is_current else "  "
            sid_short = s.session_id[-12:]

            st.markdown(
                f'<div style="font-size:0.78rem;margin-bottom:2px;">'
                f"{marker} <strong>{s.session_name[:20]}</strong></div>"
                f'<div style="font-size:0.68rem;color:#94A3B8;margin-bottom:4px;">'
                f"🕒 {time_str} · {s.message_count} 条 · <code>{sid_short}</code></div>",
                unsafe_allow_html=True,
            )

            col_sw, col_rn, col_dl = st.columns([1, 1, 1])
            with col_sw:
                if not s.is_current:
                    if st.button(
                        "📂", key=f"sw_{i}", use_container_width=True, help="切换到此会话"
                    ):
                        sm.load_session(agent, s.session_id, st.session_state.chat_history)
                        st.session_state.session_list = sm.list_sessions(agent)
                        st.rerun()
                else:
                    st.button("✅", key=f"cur_{i}", disabled=True, use_container_width=True)

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
                if st.button("🗑️", key=f"dl_{i}", use_container_width=True, help="删除此会话"):
                    sm.delete_session(agent, s.session_id)
                    if s.is_current:
                        st.session_state.chat_history.clear()
                    st.session_state.session_list = sm.list_sessions(agent)
                    st.rerun()

            st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Main Area — ChatGPT 风格对话界面
# ============================================================
# ── 隐藏 Streamlit 默认 header/footer ──
st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ── Welcome Screen ──
if st.session_state.chat_history.empty:
    st.markdown(
        """
    <div class="dp-hero">
        <div class="dp-hero-logo">🔮</div>
        <h1 class="dp-hero-title">DeepQuery</h1>
        <p class="dp-hero-subtitle">本地智能，深度问答 · 100% 私有知识库</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Feature cards
    st.markdown(
        """
    <div class="dp-feature-grid">
        <div class="dp-feature-card">
            <div class="dp-feature-icon">🔒</div>
            <div class="dp-feature-title">完全私有</div>
            <div class="dp-feature-desc">数据永不出本机</div>
        </div>
        <div class="dp-feature-card">
            <div class="dp-feature-icon">⚡</div>
            <div class="dp-feature-title">本地极速</div>
            <div class="dp-feature-desc">零网络延迟</div>
        </div>
        <div class="dp-feature-card">
            <div class="dp-feature-icon">🎯</div>
            <div class="dp-feature-title">精准检索</div>
            <div class="dp-feature-desc">Reranker 精排</div>
        </div>
        <div class="dp-feature-card">
            <div class="dp-feature-icon">🧠</div>
            <div class="dp-feature-title">多轮对话</div>
            <div class="dp-feature-desc">上下文记忆</div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="text-align:center;color:var(--dp-text-muted);font-size:0.85rem;margin-top:8px;">
    在下方添加文档 URL 或上传文件，即可开始构建知识库并提问
    </div>
    """,
        unsafe_allow_html=True,
    )

# ── Render chat history ──
for msg in st.session_state.chat_history.messages:
    avatar = "👤" if msg.role == "user" else "🔮"
    with st.chat_message(msg.role, avatar=avatar):
        st.markdown(msg.content)

# ── Inline Knowledge Toolbar (URL input + file upload) ──
# 使用 HTML 容器包裹 Streamlit 组件
st.markdown(
    '<div class="dp-knowledge-bar"><span class="dp-kb-label">📁 添加知识</span>',
    unsafe_allow_html=True,
)

col_url, col_btn, col_div, col_file = st.columns([3, 0.8, 0.06, 1.2])

with col_url:
    new_url = st.text_input(
        "文档URL",
        placeholder="粘贴文档 URL…",
        label_visibility="collapsed",
        key="main_url_input",
    )

with col_btn:
    if st.button("🌐 添加", use_container_width=True, key="main_btn_add_url"):
        if new_url:
            if new_url not in st.session_state.urls:
                st.session_state.urls.append(new_url)
                with st.spinner("📥 正在处理文档……"):
                    try:
                        kb_manager.add_source(new_url)
                        st.success("✅ 已添加")
                    except Exception as e:
                        st.error(f"添加失败: {e}")
                st.rerun()
            else:
                st.warning("该 URL 已添加")
        else:
            st.error("请输入 URL")

# 空列充当竖线分隔符，用 CSS 渲染
st.markdown("</div>", unsafe_allow_html=True)

# 文件上传放在 toolbar 下方（因为 file_uploader 宽度较大）
col_f1, col_f2 = st.columns([4, 1])
with col_f1:
    uploaded_file = st.file_uploader(
        "上传本地文件",
        type=["pdf", "txt", "md", "markdown", "docx", "html", "htm"],
        label_visibility="collapsed",
        key="main_file_uploader",
        help="支持 PDF / Word / HTML / TXT / Markdown",
    )

# 处理文件上传
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
        st.info(f"📁 {uploaded_file.name} 已在知识库中")

# ── Chat input ──
query = st.chat_input(placeholder="输入你的问题，支持多轮追问……")

if query:
    # 添加用户消息
    st.session_state.chat_history.add_user_message(query)

    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    # 调用 Agent 并流式渲染
    with st.chat_message("assistant", avatar="🔮"):
        response = ""
        react_steps: list = []
        react_placeholder = st.empty()
        resp_container = st.empty()

        try:
            for chunk in agent.run(
                query,
                stream=True,
                session_id=get_session_id(),
            ):
                event = getattr(chunk, "event", "")

                # ── ToolCallStarted ──
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

                # ── ToolCallCompleted ──
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

                # ── ReasoningStep ──
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

                # ── ReasoningCompleted ──
                elif event == "ReasoningCompleted":
                    react_steps.append(
                        {
                            "phase": "think",
                            "icon": "✅",
                            "label": "推理完成",
                            "detail": "",
                        }
                    )

                # ── Update ReAct visualization ──
                if config.SHOW_REACT_PROCESS and react_steps:
                    react_placeholder.markdown(
                        _render_react_steps(react_steps), unsafe_allow_html=True
                    )

                # ── Accumulate response content ──
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

    if response:
        st.session_state.chat_history.add_assistant_message(response)


# ============================================================
# Footer — 系统原理说明
# ============================================================
with st.expander("📖 系统工作原理"):
    st.markdown(
        """
### 🏗️ 架构数据流

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

### 🧩 核心组件

| 组件 | 作用 |
|------|------|
| **EmbeddingGemma** | 将文本转换为 768 维向量表示 |
| **LanceDB** | 存储和检索文档向量（嵌入式、零运维） |
| **CrossEncoder Reranker** | 对候选文档块二次精排，提升检索精度 |
| **Llama 3.2** | 基于上下文生成自然语言回答 |
| **Agno Agent** | 智能编排检索与生成的全流程 |
| **🧠 对话记忆** | SQLite 持久化 session，支持多轮追问 |

### 💡 为什么选择 100% 本地部署？

- 🔒 **数据隐私**：文档永远不会离开你的机器
- 💰 **零成本**：无需支付任何 API 调用费用
- ⚡ **低延迟**：无网络往返，响应更快
- 🌐 **离线可用**：不依赖互联网连接
- 🏢 **企业就绪**：适合内网部署、合规要求严格的场景
        """
    )

# ── Brand Footer ──
st.markdown(
    """
<div style="text-align:center;padding:20px 0 8px 0;color:#94A3B8;font-size:0.72rem;">
    🔮 <strong>DeepQuery</strong> — Agentic RAG 知识库系统
    &nbsp;·&nbsp;
    100% 本地运行
    &nbsp;·&nbsp;
    EmbeddingGemma + LanceDB + Llama 3.2
</div>
""",
    unsafe_allow_html=True,
)
