"""导出在线研究 Agent 可调用的工具。"""

from langchain_core.tools import BaseTool

from .local_knowledge import search_local_knowledge
from .source_reader import read_source
from .weather import get_current_weather
from .web_search import search_web

RESEARCH_TOOLS: list[BaseTool] = [
    search_local_knowledge,
    get_current_weather,
    search_web,
    read_source,
]

__all__ = [
    "RESEARCH_TOOLS",
    "get_current_weather",
    "read_source",
    "search_local_knowledge",
    "search_web",
]
