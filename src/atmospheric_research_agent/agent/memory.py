"""V2.1 的最小项目记忆：本地持久化、可回忆、但不充当正式证据。"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .settings import MAX_MEMORY_EXCERPT_CHARACTERS, MEMORY_RECALL_LIMIT
from .state import EvidenceRecord, ProjectMemoryRecord


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MEMORY_PATH = PROJECT_ROOT / "data" / "runtime_memory" / "project_memory.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _memory_terms(text: str) -> set[str]:
    """用无模型、无网络的轻量词项重合，筛选与当前问题较相关的历史条目。"""

    normalized = text.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    bigrams = {chinese[index : index + 2] for index in range(max(0, len(chinese) - 1))}
    return words | bigrams


def load_project_memory(*, store_path: Path = DEFAULT_MEMORY_PATH) -> list[ProjectMemoryRecord]:
    """读取本地记忆；不存在时返回空列表，不把首次运行当作错误。"""

    if not store_path.exists():
        return []
    raw = json.loads(store_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("项目记忆文件应为 JSON 数组。")
    return [ProjectMemoryRecord(item) for item in raw if isinstance(item, dict)]


def recall_project_memory(
    question: str,
    *,
    limit: int = MEMORY_RECALL_LIMIT,
    store_path: Path = DEFAULT_MEMORY_PATH,
) -> list[ProjectMemoryRecord]:
    """按轻量词项重合召回相关记忆；没有重合时不硬塞无关历史。"""

    question_terms = _memory_terms(question)
    scored: list[tuple[int, str, ProjectMemoryRecord]] = []
    for record in load_project_memory(store_path=store_path):
        record_text = f"{record.get('question', '')}\n{record.get('summary', '')}"
        overlap = len(question_terms & _memory_terms(record_text))
        if overlap:
            scored.append((overlap, record.get("created_at", ""), record))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [record for _, _, record in scored[:limit]]


def _report_summary(report: str) -> str:
    """从最终报告中截取结论正文，排除程序附加的来源清单以控制记忆体积。"""

    body = report.split("## 资料来源", maxsplit=1)[0]
    compact = re.sub(r"\s+", " ", body).strip()
    return compact[:MAX_MEMORY_EXCERPT_CHARACTERS]


def make_memory_record(
    *,
    question: str,
    final_report: str,
    evidence: list[EvidenceRecord],
) -> ProjectMemoryRecord:
    """将已核验完成的研究压缩为一张项目资料卡。"""

    sources = list(
        dict.fromkeys(
            item.get("source_location") or item.get("source_display_name", "")
            for item in evidence
            if item.get("source_location") or item.get("source_display_name")
        )
    )
    return {
        "memory_id": str(uuid4()),
        "question": question,
        "summary": _report_summary(final_report),
        "source_locations": sources,
        "created_at": _utc_now(),
        "status": "verified_report",
    }


def save_project_memory(
    record: ProjectMemoryRecord,
    *,
    store_path: Path = DEFAULT_MEMORY_PATH,
) -> None:
    """原子写入记忆文件，避免中途写入留下损坏 JSON。"""

    records = load_project_memory(store_path=store_path)
    records.append(record)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = store_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(store_path)
