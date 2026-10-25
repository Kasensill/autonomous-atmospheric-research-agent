"""任务的显式回放时钟或调用时实际时钟；不从问题中的日期猜测当前时间。"""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

BEIJING_TIMEZONE = timezone(timedelta(hours=8))


def parse_reference_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("缺少时区")
    except (TypeError, ValueError) as error:
        raise ValueError("reference_time 必须是带时区的 ISO 时间；未指定时使用实际时钟。") from error
    return parsed.astimezone(UTC)


def reference_now(state: Mapping[str, Any]) -> datetime:
    value = state.get("reference_time")
    return parse_reference_time(value) if value is not None else datetime.now(UTC)


def time_context(state: Mapping[str, Any]) -> str:
    now = reference_now(state)
    origin = "调用者显式指定的回放时钟" if state.get("reference_time") is not None else "本次调用时的实际时钟"
    return (f"程序时间基准（{origin}）：UTC {now.isoformat()}；"
            f"北京时间 {now.astimezone(BEIJING_TIMEZONE).isoformat()}。"
            "判断过去／未来须以此为准；事件日期、资料发布日期不自动成为当前日期。")
