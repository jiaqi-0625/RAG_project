"""
Agent 工具模块 — 为 Agent 提供可调用的外部工具（Function Calling）。

设计思路：
- 每个工具都是纯 Python 函数，用 Google 风格的 docstring 描述用途和参数
- Agno 框架的 Function.from_callable() 会自动从类型注解和 docstring 生成 JSON Schema
- 工具设计遵循"100% 本地 / 免费"原则，不依赖付费 API

支持的工具：
- search_web       → DuckDuckGo 网页搜索（免费，无需 API Key）
- calculate        → 安全数学表达式求值
- get_current_time → 获取当前日期时间

面试时说：
"我基于 Agno 的 Function Calling 机制为 Agent 添加了工具调用能力，
工具函数使用类型注解 + docstring 自动生成 JSON Schema，
Agent 可以自主决策何时调用哪个工具来增强回答质量。"
"""

import logging
from datetime import datetime
from typing import List, Optional

from agno.tools.function import Function

logger = logging.getLogger(__name__)

# ============================================================
# Tool Functions
# ============================================================


def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information using DuckDuckGo. Use this when you need
    up-to-date information beyond your training data, or when the knowledge base
    does not contain relevant information.

    Args:
        query: The search query string. Be specific and include relevant keywords.
        max_results: Maximum number of search results to return (default 5, max 10).

    Returns:
        A string containing the formatted search results with titles, snippets, and URLs.
    """
    try:
        from ddgs import DDGS

        max_results = min(max_results, 10)
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"### {r['title']}\n"
                    f"{r['body']}\n"
                    f"🔗 {r['href']}"
                )

        if not results:
            return (
                f'No results found for query: "{query}". '
                f"Try different keywords or check your network connection."
            )

        header = (
            f"🌐 Web search results for \"{query}\" "
            f"({len(results)} results):\n\n"
        )
        return header + "\n\n---\n\n".join(results)

    except ImportError:
        return (
            "⚠️ Web search is unavailable: ddgs package is not installed. "
            "Install it with: pip install ddgs"
        )
    except Exception as e:
        logger.error(f"Web search failed for '{query}': {e}")
        return f"⚠️ Search error: {e}. Please try again with different keywords."


def calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression. Use this for arithmetic,
    scientific calculations, or any quantitative reasoning the user requests.

    Supported operations:
    - Basic: +, -, *, /, // (floor division), % (modulo), ** (power)
    - Functions: sqrt, sin, cos, tan, log, log10, exp, abs, round, ceil, floor
    - Constants: pi, e

    Args:
        expression: A mathematical expression string to evaluate.
                    Examples: "2 + 3 * 4", "sqrt(16)", "sin(pi / 2)", "2**10".

    Returns:
        The calculated result with the expression echoed for verification.
    """
    import math

    # Safe namespace — only whitelisted math functions and constants
    safe_namespace = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": math.pow,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
        "degrees": math.degrees,
        "radians": math.radians,
    }

    # Strip whitespace and validate non-empty
    expression = expression.strip()
    if not expression:
        return "⚠️ Calculation error: empty expression."

    if len(expression) > 500:
        return "⚠️ Calculation error: expression too long (max 500 characters)."

    try:
        result = eval(expression, {"__builtins__": {}}, safe_namespace)
        # Format result nicely
        if isinstance(result, float):
            # Avoid floating-point noise for nice numbers
            if result == int(result):
                result_repr = f"{int(result)}"
            else:
                result_repr = f"{result:.10g}"
        else:
            result_repr = str(result)
        return f"📊 {expression} = {result_repr}"
    except SyntaxError as e:
        return f"⚠️ Syntax error in expression: {e}"
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        return f"⚠️ Math error: {e}"
    except Exception as e:
        return f"⚠️ Calculation error: {e}"


def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks about
    the current time, today's date, day of the week, or needs to calculate
    time differences relative to the present moment.

    Returns:
        Current date and time as a formatted string, including the day of the week.
    """
    now = datetime.now()
    # Use Chinese-friendly format that also works in English contexts
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = weekday_names[now.weekday()]
    return (
        f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({weekday})\n"
        f"   Unix timestamp: {int(now.timestamp())}"
    )


# ============================================================
# Tool Registry
# ============================================================

# 自动从函数生成 Agno Function 对象（含 JSON Schema）
_tool_functions = [search_web, calculate, get_current_time]

DEFAULT_TOOLS: List[Function] = [
    Function.from_callable(fn) for fn in _tool_functions
]

# 按名称索引，方便单独引用
TOOLS_BY_NAME: dict = {tool.name: tool for tool in DEFAULT_TOOLS}


def get_tools(
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[Function]:
    """
    按名称筛选工具列表。

    Args:
        include: 只返回这些名称的工具。为 None 则包含全部。
        exclude: 排除这些名称的工具。为 None 则不排除。

    Returns:
        筛选后的 Function 对象列表。

    Example:
        # 只要搜索和计算
        tools = get_tools(include=["search_web", "calculate"])

        # 排除搜索（不想让 Agent 联网）
        tools = get_tools(exclude=["search_web"])
    """
    tools = DEFAULT_TOOLS

    if include is not None:
        include_set = set(include)
        tools = [t for t in tools if t.name in include_set]

    if exclude is not None:
        exclude_set = set(exclude)
        tools = [t for t in tools if t.name not in exclude_set]

    return tools
