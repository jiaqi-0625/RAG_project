"""
对话记忆模块 — 管理多轮对话的 session 和聊天历史。

设计思路:
- SessionManager 负责 session_id 的生成、切换、清理、历史查询
- ChatHistory 负责当前会话消息的存储和渲染
- 遵循项目现有的 ABC + 实现模式

面试时: "我设计了一个可插拔的对话记忆层，session 通过 SQLite 持久化，
支持多轮对话无缝衔接，并提供历史会话浏览和切换功能。"
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from agno.agent import Agent as AgnoAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """单条聊天消息"""

    role: str  # "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class SessionInfo:
    """历史会话摘要信息（用于列表展示）"""

    session_id: str
    session_name: str
    created_at: Optional[int] = None   # unix timestamp
    updated_at: Optional[int] = None   # unix timestamp
    message_count: int = 0
    is_current: bool = False


class ChatHistory:
    """
    聊天历史管理器。

    维护当前会话中的所有消息，提供增删查功能。
    消息以 dataclass 形式存储在内存中（重启后丢失），
    但 session 级别的历史由 Agno 的 SQLite DB 持久化。

    Usage:
        history = ChatHistory()
        history.add_user_message("什么是 RAG？")
        history.add_assistant_message("RAG 是检索增强生成……")
        for msg in history.messages:
            print(msg.role, msg.content)
    """

    def __init__(self, max_messages: int = 100):
        self._messages: List[ChatMessage] = []
        self._max_messages = max_messages

    @property
    def messages(self) -> List[ChatMessage]:
        """返回所有消息（只读副本）"""
        return list(self._messages)

    @property
    def empty(self) -> bool:
        """对话是否为空"""
        return len(self._messages) == 0

    def add_user_message(self, content: str) -> ChatMessage:
        """添加用户消息"""
        msg = ChatMessage(role="user", content=content)
        self._messages.append(msg)
        self._trim()
        return msg

    def add_assistant_message(self, content: str) -> ChatMessage:
        """添加助手消息"""
        msg = ChatMessage(role="assistant", content=content)
        self._messages.append(msg)
        self._trim()
        return msg

    def load_from_agno_messages(
        self, messages: List[Any], clear_first: bool = True
    ) -> None:
        """
        从 Agno 的 Message 对象列表加载消息到本地 ChatHistory。

        Args:
            messages: agno.models.message.Message 对象列表。
            clear_first: 是否先清空现有消息。
        """
        if clear_first:
            self._messages.clear()
        for msg in messages:
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")
            if content and role in ("user", "assistant"):
                self._messages.append(ChatMessage(role=role, content=str(content)))
        self._trim()

    def clear(self) -> None:
        """清空本地聊天历史"""
        self._messages.clear()

    def _trim(self) -> None:
        """保持消息数量在限制内（保留最新的）"""
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ChatHistory(messages={len(self._messages)})"


class SessionManager:
    """
    Session 管理器。

    负责：
    1. 生成/切换 session_id
    2. 将 session_id 透传给 Agent.run()
    3. 支持"新对话"按钮（生成新 session_id）
    4. 查询历史会话列表
    5. 加载历史会话（切换对话）
    6. 删除/重命名会话

    Usage:
        sm = SessionManager()
        sid = sm.current_session_id
        # 传给 agent.run(query, session_id=sid)
        sm.new_session()  # 开始新对话
        sessions = sm.list_sessions(agent)  # 查询历史
    """

    def __init__(self, session_id: Optional[str] = None):
        self._session_id = session_id or self._generate_session_id()

    # ============================================================
    # 当前 Session 管理
    # ============================================================

    @property
    def current_session_id(self) -> str:
        """当前 session_id"""
        return self._session_id

    def new_session(self) -> str:
        """创建新 session（开始全新对话），返回新 session_id"""
        self._session_id = self._generate_session_id()
        return self._session_id

    def switch_to(self, session_id: str) -> None:
        """切换到指定 session"""
        self._session_id = session_id

    # ============================================================
    # 历史会话查询（通过 Agno DB）
    # ============================================================

    def list_sessions(
        self,
        agent: "AgnoAgent",
        limit: int = 50,
    ) -> List[SessionInfo]:
        """
        从数据库获取历史会话列表。

        Args:
            agent: Agno Agent 实例（需要有 db 属性）。
            limit: 最大返回数量。

        Returns:
            SessionInfo 列表，按更新时间降序排列。
        """
        if agent.db is None:
            logger.warning("Agent 没有配置数据库，无法查询历史会话")
            return []

        try:
            from agno.db.base import SessionType
            sessions = agent.db.get_sessions(
                session_type=SessionType.AGENT,
                component_id=agent.id,
                limit=limit,
                sort_by="updated_at",
                sort_order="desc",
                deserialize=True,
            )
        except Exception as e:
            logger.error(f"查询历史会话失败: {e}")
            return []

        result = []
        for s in (sessions or []):
            # 提取会话名称
            session_data = s.session_data or {}
            session_name = session_data.get("session_name", "") or "未命名对话"

            # 统计消息数
            msg_count = 0
            if s.runs:
                for run in s.runs:
                    messages = getattr(run, "messages", None)
                    if messages:
                        # 只统计 user 和 assistant 消息
                        msg_count += sum(
                            1 for m in messages
                            if getattr(m, "role", "") in ("user", "assistant")
                        )

            result.append(SessionInfo(
                session_id=s.session_id,
                session_name=session_name,
                created_at=s.created_at,
                updated_at=s.updated_at,
                message_count=msg_count,
                is_current=(s.session_id == self._session_id),
            ))

        return result

    def get_session_messages(
        self,
        agent: "AgnoAgent",
        session_id: Optional[str] = None,
    ) -> List[Any]:
        """
        从数据库获取指定 session 的消息。

        Args:
            agent: Agno Agent 实例。
            session_id: 目标 session ID，默认使用当前 session。

        Returns:
            agno Message 对象列表。
        """
        sid = session_id or self._session_id
        if agent.db is None:
            return []

        try:
            from agno.agent._session import get_chat_history
            return get_chat_history(agent, session_id=sid)
        except Exception as e:
            logger.error(f"获取会话消息失败 (session={sid}): {e}")
            return []

    def load_session(
        self,
        agent: "AgnoAgent",
        session_id: str,
        chat_history: ChatHistory,
    ) -> bool:
        """
        加载历史会话：切换到指定 session 并恢复消息到 ChatHistory。

        Args:
            agent: Agno Agent 实例。
            session_id: 要加载的 session ID。
            chat_history: ChatHistory 实例，用于恢复消息。

        Returns:
            是否加载成功。
        """
        messages = self.get_session_messages(agent, session_id)
        if not messages:
            # session 存在但没有消息（新 session）
            self.switch_to(session_id)
            chat_history.clear()
            return True

        self.switch_to(session_id)
        chat_history.load_from_agno_messages(messages, clear_first=True)
        logger.info(f"已加载会话 {session_id}，共 {len(chat_history)} 条消息")
        return True

    def rename_session(
        self,
        agent: "AgnoAgent",
        session_id: str,
        new_name: str,
    ) -> bool:
        """重命名会话"""
        if agent.db is None:
            return False
        try:
            from agno.db.base import SessionType
            agent.db.rename_session(
                session_id=session_id,
                session_type=SessionType.AGENT,
                session_name=new_name,
            )
            return True
        except Exception as e:
            logger.error(f"重命名会话失败: {e}")
            return False

    def delete_session(
        self,
        agent: "AgnoAgent",
        session_id: str,
    ) -> bool:
        """
        删除指定会话。如果删除的是当前会话，自动创建新会话。

        Returns:
            是否删除成功。
        """
        if agent.db is None:
            return False
        try:
            agent.db.delete_session(session_id)
            if session_id == self._session_id:
                self.new_session()
            return True
        except Exception as e:
            logger.error(f"删除会话失败: {e}")
            return False

    # ============================================================
    # 内部工具
    # ============================================================

    @staticmethod
    def _generate_session_id() -> str:
        """生成唯一的 session ID"""
        return f"rag-session-{uuid4().hex[:12]}"

    def __repr__(self) -> str:
        return f"SessionManager(session_id={self._session_id})"
