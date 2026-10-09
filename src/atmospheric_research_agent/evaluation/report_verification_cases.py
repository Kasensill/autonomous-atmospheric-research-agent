"""读取 V2 Reflection / Verification 的人工标注评测案例。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class ReportVerificationCase:
    """一份草稿、其正式证据与人工标注的期望审查结论。"""

    case_id: str
    category: str
    question: str
    evidence: tuple[dict[str, str], ...]
    draft_report: str
    expected_verdict: Literal["pass", "revise"]
    note: str


def _required_text(value: Any, *, field_name: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"核验评测案例 {case_id} 的 {field_name} 必须是非空字符串。")
    return value.strip()


def load_report_verification_cases(path: Path) -> list[ReportVerificationCase]:
    """读取并校验案例，防止评测输入本身缺少证据或期望答案。"""

    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"未找到核验评测题集：{path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"核验评测题集不是合法 JSON：{path}") from error
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("核验评测题集必须是非空 JSON 数组。")

    cases: list[ReportVerificationCase] = []
    seen_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("核验评测题集中的每条案例必须是对象。")
        case_id = _required_text(raw_case.get("id"), field_name="id", case_id="<unknown>")
        if case_id in seen_ids:
            raise ValueError(f"核验评测案例 ID 重复：{case_id}")
        seen_ids.add(case_id)

        verdict = raw_case.get("expected_verdict")
        if verdict not in {"pass", "revise"}:
            raise ValueError(f"核验评测案例 {case_id} 的 expected_verdict 必须为 pass 或 revise。")
        raw_evidence = raw_case.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError(f"核验评测案例 {case_id} 至少需要一条 evidence。")
        evidence: list[dict[str, str]] = []
        for index, item in enumerate(raw_evidence, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"核验评测案例 {case_id} 的第 {index} 条 evidence 必须是对象。")
            evidence.append(
                {
                    "evidence_id": _required_text(
                        item.get("evidence_id"), field_name="evidence.evidence_id", case_id=case_id
                    ),
                    "content": _required_text(
                        item.get("content"), field_name="evidence.content", case_id=case_id
                    ),
                    "source_display_name": _required_text(
                        item.get("source_display_name"),
                        field_name="evidence.source_display_name",
                        case_id=case_id,
                    ),
                    "source_location": _required_text(
                        item.get("source_location"),
                        field_name="evidence.source_location",
                        case_id=case_id,
                    ),
                    "source_type": _required_text(
                        item.get("source_type"), field_name="evidence.source_type", case_id=case_id
                    ),
                    "tool_name": _required_text(
                        item.get("tool_name"), field_name="evidence.tool_name", case_id=case_id
                    ),
                }
            )
        cases.append(
            ReportVerificationCase(
                case_id=case_id,
                category=_required_text(raw_case.get("category"), field_name="category", case_id=case_id),
                question=_required_text(raw_case.get("question"), field_name="question", case_id=case_id),
                evidence=tuple(evidence),
                draft_report=_required_text(
                    raw_case.get("draft_report"), field_name="draft_report", case_id=case_id
                ),
                expected_verdict=verdict,
                note=_required_text(raw_case.get("note"), field_name="note", case_id=case_id),
            )
        )
    return cases
