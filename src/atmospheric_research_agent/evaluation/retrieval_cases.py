"""读取并校验 V1.6 本地检索评测题集。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .retrieval_metrics import SourceReference


@dataclass(frozen=True)
class RetrievalCase:
    """一条人工标注的问题及其应召回的现有知识片段。"""

    case_id: str
    question: str
    category: str
    expected_sources: tuple[SourceReference, ...]
    note: str
    known_data_gap: bool = False


def _require_non_empty_string(value: Any, field_name: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"评测题 {case_id} 的 {field_name} 必须是非空字符串。")
    return value.strip()


def load_retrieval_cases(path: Path) -> list[RetrievalCase]:
    """读取 JSON 题集，拒绝重复 ID、不完整来源和不可解释的空标注。"""

    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"未找到评测题集：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"评测题集不是合法 JSON：{path}") from error

    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("评测题集必须是至少包含一条题目的 JSON 数组。")

    cases: list[RetrievalCase] = []
    seen_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("评测题集中的每一条必须是 JSON 对象。")
        case_id = _require_non_empty_string(raw_case.get("id"), "id", "<unknown>")
        if case_id in seen_ids:
            raise ValueError(f"评测题 ID 重复：{case_id}")
        seen_ids.add(case_id)

        raw_sources = raw_case.get("expected_sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError(f"评测题 {case_id} 至少需要一条 expected_sources 标注。")
        sources: list[SourceReference] = []
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                raise ValueError(f"评测题 {case_id} 的来源标注必须是对象。")
            sources.append(
                SourceReference(
                    location=_require_non_empty_string(
                        raw_source.get("location"), "expected_sources.location", case_id
                    ),
                    section=_require_non_empty_string(
                        raw_source.get("section"), "expected_sources.section", case_id
                    ),
                )
            )
        if len(set(sources)) != len(sources):
            raise ValueError(f"评测题 {case_id} 存在重复的来源标注。")

        known_data_gap = raw_case.get("known_data_gap", False)
        if not isinstance(known_data_gap, bool):
            raise ValueError(f"评测题 {case_id} 的 known_data_gap 必须是布尔值。")

        cases.append(
            RetrievalCase(
                case_id=case_id,
                question=_require_non_empty_string(raw_case.get("question"), "question", case_id),
                category=_require_non_empty_string(raw_case.get("category"), "category", case_id),
                expected_sources=tuple(sources),
                note=_require_non_empty_string(raw_case.get("note"), "note", case_id),
                known_data_gap=known_data_gap,
            )
        )
    return cases
