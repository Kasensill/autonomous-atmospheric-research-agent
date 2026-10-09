"""本地知识库检索评测的纯指标函数。

这些函数不调用 Embedding、Rerank 或 Chroma。它们只接收某次检索已经返回的
候选列表和人工标注的相关来源，因此可以通过普通单元测试验证计算本身。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceReference:
    """评测集里一个可检索知识片段的稳定人类标识。"""

    location: str
    section: str


@dataclass(frozen=True)
class RetrievalMetrics:
    """一条问题在指定 Top-K 下的检索质量结果。"""

    hit_at_k: bool
    reciprocal_rank: float
    coverage_at_k: float
    first_relevant_rank: int | None


def source_reference_from_candidate(candidate: dict[str, Any]) -> SourceReference | None:
    """从本地检索候选中读取与评测集一致的来源定位。"""

    source = candidate.get("source")
    if not isinstance(source, dict):
        return None
    location = source.get("location")
    section = source.get("section")
    if not isinstance(location, str) or not location:
        return None
    if not isinstance(section, str) or not section:
        return None
    return SourceReference(location=location, section=section)


def calculate_retrieval_metrics(
    candidates: Sequence[dict[str, Any]],
    expected_sources: Iterable[SourceReference],
    *,
    top_k: int,
) -> RetrievalMetrics:
    """计算 Hit@K、MRR@K 和 Coverage@K。

    - Hit@K：Top-K 中是否至少出现一个人工标注的相关片段。
    - MRR@K：第一条相关片段在 Top-K 内的倒数名次；未命中为 0。
    - Coverage@K：人工标注的所有相关片段中，Top-K 实际覆盖的比例。
    """

    if top_k < 1:
        raise ValueError("top_k 必须至少为 1。")

    expected = set(expected_sources)
    if not expected:
        raise ValueError("expected_sources 至少需要一个人工标注的相关来源。")

    retrieved = [
        reference
        for candidate in candidates[:top_k]
        if (reference := source_reference_from_candidate(candidate)) is not None
    ]
    relevant_ranks = [
        rank
        for rank, reference in enumerate(retrieved, start=1)
        if reference in expected
    ]
    first_relevant_rank = relevant_ranks[0] if relevant_ranks else None
    covered = set(retrieved) & expected

    return RetrievalMetrics(
        hit_at_k=first_relevant_rank is not None,
        reciprocal_rank=(1 / first_relevant_rank) if first_relevant_rank else 0.0,
        coverage_at_k=len(covered) / len(expected),
        first_relevant_rank=first_relevant_rank,
    )


def mean_metrics(results: Iterable[RetrievalMetrics]) -> dict[str, float]:
    """对多道题的指标取简单平均，供策略间横向对比。"""

    result_list = list(results)
    if not result_list:
        raise ValueError("至少需要一条评测结果。")
    count = len(result_list)
    return {
        "hit_rate": sum(item.hit_at_k for item in result_list) / count,
        "mrr": sum(item.reciprocal_rank for item in result_list) / count,
        "coverage": sum(item.coverage_at_k for item in result_list) / count,
    }
