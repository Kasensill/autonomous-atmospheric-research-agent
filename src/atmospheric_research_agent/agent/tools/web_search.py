"""使用 Tavily 发现公开网页候选来源。"""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from langchain_core.tools import tool

from ...ingestion.build_index import PROJECT_ROOT
from ..settings import WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT_SECONDS


TAVILY_SEARCH_ENDPOINT = "https://api.tavily.com/search"

def _tavily_search(query: str, api_key: str) -> dict[str, Any]:
    """向 Tavily 发送最小搜索请求；不请求网页全文。"""

    payload = json.dumps(
        {
            "query": query,
            "search_depth": "basic",
            "max_results": WEB_SEARCH_MAX_RESULTS,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
    ).encode("utf-8")
    request = Request(
        TAVILY_SEARCH_ENDPOINT,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "AutonomousResearchAgent/0.1",
        },
    )
    with urlopen(request, timeout=WEB_SEARCH_TIMEOUT_SECONDS) as response:  # noqa: S310 - endpoint 为本模块常量
        return json.loads(response.read().decode("utf-8"))


def search_public_web(query: str) -> dict[str, Any]:
    """搜索候选网页；候选摘要不是正式研究证据。"""

    clean_query = query.strip()
    if not clean_query:
        return {
            "status": "error",
            "tool": "search_web",
            "message": "搜索问题不能为空。",
            "results": [],
        }

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "tool": "search_web",
            "query": clean_query,
            "message": "缺少 TAVILY_API_KEY，请在项目根目录 .env 中配置。",
            "results": [],
        }

    try:
        response = _tavily_search(clean_query, api_key)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
        return {
            "status": "error",
            "tool": "search_web",
            "query": clean_query,
            "message": f"网页搜索失败：{error}",
            "results": [],
        }

    results = [
        {
            "title": item.get("title", "未命名网页"),
            "url": item.get("url", ""),
            "summary": item.get("content", ""),
            "relevance_score": item.get("score"),
            "published_at": item.get("published_date"),
        }
        for item in response.get("results", [])
        if item.get("url")
    ]
    return {
        "status": "success",
        "tool": "search_web",
        "provider": "Tavily",
        "query": clean_query,
        "result_count": len(results),
        # 这些只是候选来源；应再用 read_source 读取正文后才可能成为 evidence。
        "results": results,
    }


@tool
def search_web(query: str) -> dict[str, Any]:
    """搜索公开网页，寻找可能的权威资料候选来源。

    该工具只用于发现候选 URL，不应把搜索摘要直接当作正式证据；需要时再调用
    ``read_source`` 读取来源正文与元数据。
    """

    return search_public_web(query)
