"""Qwen Text Rerank：只精排 RRF 候选，不承担大规模召回。"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from ...ingestion.build_index import PROJECT_ROOT
from ..settings import RERANK_FINAL_TOP_K, RERANK_MODEL


RERANK_INSTRUCTION = (
    "Rank the passages by how directly and accurately they support answering the "
    "atmospheric science query. Prefer explanatory evidence over a mere keyword match."
)


def _configure_dashscope() -> Any:
    """配置 DashScope；Rerank 专用 endpoint 可选，避免影响已有 Embedding。"""

    try:
        import dashscope
    except ImportError as error:  # pragma: no cover - 依赖安装问题
        raise RuntimeError("未安装 dashscope，请先安装项目依赖。") from error

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法调用 Qwen Rerank。")
    dashscope.api_key = api_key

    # qwen3.7-text-rerank 的新接口可要求业务空间专用地址。若用户已在 .env
    # 配置该地址则使用它；未配置时保留 SDK 默认地址，便于兼容已有账号设置。
    rerank_base_url = os.getenv("DASHSCOPE_RERANK_BASE_URL")
    if rerank_base_url:
        dashscope.base_http_api_url = rerank_base_url
    return dashscope


def _response_value(response: Any, key: str, default: Any = None) -> Any:
    """兼容 DashScope 响应的字典与属性访问形式。"""

    if isinstance(response, dict):
        return response.get(key, default)
    try:
        return response[key]
    except (KeyError, TypeError):
        return getattr(response, key, default)


def call_qwen_rerank(
    query: str,
    documents: list[str],
    *,
    top_k: int,
    model: str = RERANK_MODEL,
) -> list[dict[str, Any]]:
    """调用 Qwen Rerank，并规范化为 ``index + relevance_score`` 的结果。"""

    if not documents:
        return []
    if top_k < 1:
        raise ValueError("Rerank top_k 必须至少为 1。")

    dashscope = _configure_dashscope()
    response = dashscope.TextReRank.call(
        model=model,
        query=query,
        documents=documents,
        top_n=min(top_k, len(documents)),
        return_documents=False,
        instruct=RERANK_INSTRUCTION,
    )
    status_code = _response_value(response, "status_code")
    if status_code != 200:
        message = _response_value(response, "message", "未提供错误说明")
        raise RuntimeError(f"Qwen Rerank 请求失败（status={status_code}）：{message}")

    output = _response_value(response, "output", {})
    results = _response_value(output, "results", [])
    normalized_results: list[dict[str, Any]] = []
    for result in results:
        index = _response_value(result, "index")
        score = _response_value(result, "relevance_score")
        if not isinstance(index, int) or not isinstance(score, (float, int)):
            raise RuntimeError("Qwen Rerank 返回了缺少 index 或 relevance_score 的结果。")
        if index < 0 or index >= len(documents):
            raise RuntimeError("Qwen Rerank 返回了越界的候选索引。")
        normalized_results.append({"index": index, "relevance_score": float(score)})
    return normalized_results


def rerank_fused_candidates(
    query: str,
    fused_candidates: list[dict[str, Any]],
    *,
    top_k: int = RERANK_FINAL_TOP_K,
    model: str = RERANK_MODEL,
) -> list[dict[str, Any]]:
    """精排 RRF 候选，返回最终 Top-K 并保留融合阶段的全部诊断信息。"""

    documents: list[str] = []
    for candidate in fused_candidates:
        content = candidate.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Rerank 候选缺少可排序的正文 content。")
        documents.append(content)

    rerank_results = call_qwen_rerank(query, documents, top_k=top_k, model=model)
    reranked_candidates: list[dict[str, Any]] = []
    for rerank_rank, result in enumerate(rerank_results, start=1):
        candidate = dict(fused_candidates[result["index"]])
        candidate["retrieval"] = {
            **candidate.get("retrieval", {}),
            "rerank_rank": rerank_rank,
            # 仅表示本次 query 与候选的相对相关性，不表示来源可信度。
            "rerank_score": result["relevance_score"],
        }
        reranked_candidates.append(candidate)
    return reranked_candidates
