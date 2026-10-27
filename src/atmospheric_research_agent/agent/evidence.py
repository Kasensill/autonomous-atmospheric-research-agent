"""当前证据视图：保留原始记录，但不再使用过期的当前天气资料。"""

from datetime import UTC, datetime

from .settings import WEATHER_EVIDENCE_MAX_AGE_SECONDS
from .clock import reference_now


def is_stale_weather(record: dict, *, now: datetime | None = None) -> bool:
    if record.get("source_type") != "weather":
        return False
    timestamp = record.get("valid_at")
    if not timestamp:
        # 新增元数据前的天气证据无法可靠证明新鲜度，保守要求重新查询。
        return True
    try:
        observed = datetime.fromisoformat(timestamp)
        if observed.tzinfo is None:
            return True
    except (ValueError, TypeError):
        return True
    age = ((now or datetime.now(UTC)) - observed).total_seconds()
    return age > WEATHER_EVIDENCE_MAX_AGE_SECONDS or age < -15 * 60


def usable_evidence(state: dict, *, now: datetime | None = None) -> list[dict]:
    current = now if now is not None else reference_now(state)
    return [record for record in state.get("evidence", []) if not is_stale_weather(record, now=current)]


def research_evidence(state: dict) -> list[dict]:
    """用于研究守卫的证据：文本分析材料可用，线索材料不冒充独立证据。"""
    return [record for record in usable_evidence(state)
            if record.get("source_type") != "user_material" or record.get("role") == "analysis"]


def material_only_unverified(state: dict) -> bool:
    records = usable_evidence(state)
    return (any(record.get("source_type") == "user_material" and record.get("role") in {"clue", "both"}
                for record in records)
            and not any(record.get("source_type") != "user_material" for record in records))


def has_user_materials(state: dict) -> bool:
    return bool(state.get("materials")) or any(
        record.get("source_type") == "user_material" for record in state.get("evidence", [])
    )
