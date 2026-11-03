"""历史查询、候选资料与本轮证据的边界。"""
from collections.abc import Callable
from dataclasses import dataclass
import re
from typing import Any, Literal
from uuid import UUID

from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from .evidence import is_stale_weather
from .clock import reference_now
from .settings import HISTORY_MAX_LOOKUPS, HISTORY_MAX_CANDIDATES, HISTORY_MAX_EXCERPT


class HistoryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    purpose: Literal["answer_history", "support_research"]
    target: Literal["latest", "topic", "run_id"]
    value: str = Field(default="", max_length=200)
    include_evidence: bool = False
    list_matches: bool = False

@dataclass
class HistoryContext:
    lookup: Callable[[dict, dict], dict]


def matches_request(state: dict) -> bool:
    request = state.get("history_request")
    return bool(request and (state.get("history_result") or {}).get("request") == request)


def can_lookup(state: dict) -> bool:
    result = state.get("history_result") or {}
    return (bool(state.get("history_request"))
            and result.get("lookup_count", 0) < HISTORY_MAX_LOOKUPS
            and (not result or result.get("status") == "ambiguous"
                 and len(state.get("clarification_history", [])) > result["clarification_count"]))


def lookup_history_node(state: dict, runtime: Runtime[HistoryContext]) -> dict:
    request = state["history_request"]
    result = runtime.context.lookup(request, state) if runtime.context else {"status": "not_found"}
    # 摘要只兜底按主题查历史。精确任务和新研究依赖实际档案。
    if (result["status"] == "not_found" and request["purpose"] == "answer_history"
            and request["target"] == "topic" and request["value"]):
        topic = request["value"].casefold()
        summaries = [m for m in state.get("recalled_memories", [])
                     if m.get("status") == "verified_report" and m.get("summary")
                     and topic in (m.get("question", "") + " " + m.get("summary", "")).casefold()]
        if len(summaries) == 1:
            result = {"status": "matched", "memory": summaries[0]}
    result.update(request=request, lookup_count=(state.get("history_result") or {}).get("lookup_count", 0) + 1,
                  clarification_count=len(state.get("clarification_history", [])))
    return {"history_result": result, "trace": [{"node": "lookup_history", "event": "history_lookup",
             "detail": f"{request['purpose']} → {result['status']}"}]}


def select_evidence(state: dict, indices: list[int]) -> list[dict]:
    result = state.get("history_result") or {}
    if not matches_request(state) or result.get("status") != "matched":
        return []
    if result["request"]["purpose"] != "support_research":
        return []
    seen = {item.get("evidence_id") for item in state.get("evidence", [])}
    chosen = []
    for index in indices:
        entry = next((entry for entry in result.get("evidence", []) if entry["index"] == index), None)
        if not entry or not entry["eligible"]:
            continue
        evidence = entry["record"]
        if evidence["evidence_id"] in seen or is_stale_weather(evidence, now=reference_now(state)):
            continue
        seen.add(evidence["evidence_id"])
        chosen.append({**evidence, "historical_run_id": result["run_id"], "historical_evidence_index": index})
    return chosen


def answer_history_node(state: dict) -> dict:
    result = state.get("history_result") or {}
    if not matches_request(state) or result.get("status") != "matched":
        output = ("仍有多个匹配的历史任务，请提供任务编号。"
                  if result.get("status") == "ambiguous" else "没有找到能回答该问题的历史记录。")
    elif result.get("memory"):
        memory = result["memory"]
        output = f"[Memory] 历史研究摘要（{memory['memory_id']}）\n\n{memory['summary']}"
    elif result.get("listing"):
        output = "找到以下历史任务：\n\n" + "\n".join(
            f"- {row['question']}（{row['run_id']}）" for row in result["listing"])
    else:
        output = "历史报告 · 任务 {} · {}\n\n".format(result["run_id"], result["delivery_type"])
        output += result["report"]
        if result.get("report_truncated"):
            output += "\n\n（以上为历史报告节选，可在任务档案查看完整正文。）"
        if result["request"]["include_evidence"] and result.get("evidence"):
            output += "\n\n### 当时保存的证据片段\n"
            for entry in result["evidence"]:
                record = entry["record"]
                label = record.get("source_display_name") or record.get("source_location")
                output += f"\n- 档案条目 #{entry['index']} · {label}：{record['content'][:HISTORY_MAX_EXCERPT]}\n"
    return {"final_report": output, "trace": [{"node": "answer_history", "event": "history_answered",
             "detail": f"历史查询交付：{result.get('status', 'not_found')}"}]}


def archived_history(request: dict, state: dict, rows: list[dict], *, parent_run_id: str | None) -> dict:
    """只在程序选定的旧档案集合中定位，拒绝凭模型生成的 UUID 越出请求。"""
    purpose, target, value = (request[key] for key in ("purpose", "target", "value"))
    user_text = "\n".join([state["question"], *(entry["answer"] for entry in state.get("clarification_history", []))])
    confirmed_id = None
    if target == "run_id":
        try:
            confirmed_id = str(UUID(value))
        except ValueError:
            return {"status": "not_found"}
        mentioned = re.findall(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", user_text)
        if confirmed_id not in {parent_run_id, *(item.lower() for item in mentioned)}:
            return {"status": "not_found"}
        rows = [row for row in rows if row["run_id"] == confirmed_id]
    reports = [row for row in rows if row["status"] == "completed"
               and row["delivery_type"] in {"full_report", "limited_report"}
               and row["state"].get("final_report")]
    candidates = reports if purpose == "answer_history" or target == "latest" else rows
    if target == "latest":
        candidates = candidates[:1]
    elif target == "topic":
        if not value:
            return {"status": "not_found"}
        candidates = [row for row in candidates if value.casefold() in row["question"].casefold()
                      or value.casefold() in (row["state"].get("final_report") or "").casefold()]
    if not candidates:
        return {"status": "not_found"}
    if purpose == "answer_history" and target == "topic" and request["list_matches"]:
        return {"status": "matched", "listing": [
            {"run_id": row["run_id"], "question": row["question"][:200]}
            for row in candidates[:HISTORY_MAX_CANDIDATES]]}
    if len(candidates) > 1:
        return {"status": "ambiguous", "candidates": [
            {"run_id": row["run_id"], "question": row["question"][:200]}
            for row in candidates[:HISTORY_MAX_CANDIDATES]]}
    row = candidates[0]
    report = row["state"].get("final_report") or ""
    result: dict[str, Any] = {"status": "matched", "run_id": row["run_id"],
                              "delivery_type": row.get("delivery_type"), "task_status": row["status"]}
    if purpose == "answer_history":
        result.update(report=report[:HISTORY_MAX_EXCERPT], report_truncated=len(report) > HISTORY_MAX_EXCERPT)
    if purpose == "support_research" or request["include_evidence"]:
        result["evidence"] = []
        completed = row["status"] == "completed" and row["delivery_type"] in {"full_report", "limited_report"}
        for index, evidence in enumerate(row["state"].get("evidence", []), 1):
            if not evidence.get("content"):
                continue
            eligible = bool(completed and evidence.get("evidence_id") and evidence.get("source_location")
                            and evidence.get("source_type") in {"web", "local_knowledge", "weather"}
                            and not is_stale_weather(evidence, now=reference_now(state)))
            result["evidence"].append({"index": index, "eligible": eligible, "record": evidence})
            if len(result["evidence"]) >= HISTORY_MAX_CANDIDATES:
                break
    return result
