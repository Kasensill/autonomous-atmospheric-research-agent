"""研究工具调用、协议响应及结果转证据节点。"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from .state import EvidenceRecord, ResearchState, TraceEntry
from .tools import RESEARCH_TOOLS
from .tools.clarification import ASK_USER_NAME


def _control_response(call: dict[str, Any], message: str, *, error: bool = False) -> ToolMessage:
    """关闭控制/取消请求的协议，不作为研究结果进入计数或证据。"""
    return ToolMessage(name=call["name"], tool_call_id=call["id"],
                       content=json.dumps({"status": "error" if error else "waiting_for_user", "message": message}, ensure_ascii=False),
                       additional_kwargs={"protocol_only": True},
                       status="error" if error else "success")

def _latest_tool_call_message(messages: list[BaseMessage]) -> AIMessage:
    """找到最近一条带工具调用请求的模型消息。"""

    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.tool_calls:
            return message
    raise ValueError("tools 节点没有找到待执行的 tool_calls。")

def _tool_registry(tools: list[BaseTool]) -> dict[str, BaseTool]:
    """用工具名称建立查表，避免让模型名称直接决定可执行代码。"""

    return {tool.name: tool for tool in tools}

def execute_tool_calls(
    tool_calls: list[dict[str, Any]], tools: list[BaseTool]
) -> list[ToolMessage]:
    """执行一批模型请求的工具，并把结果转为 LangChain 协议消息。"""

    registry = _tool_registry(tools)
    tool_messages: list[ToolMessage] = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_call_id = tool_call["id"]
        tool = registry.get(tool_name)

        if tool_name == ASK_USER_NAME:
            result = {"status": "error", "tool": tool_name,
                      "message": "ask_user 是控制声明，禁止在研究工具执行层调用。"}
            status: Literal["success", "error"] = "error"
        elif tool is None:
            result: dict[str, Any] = {
                "status": "error",
                "tool": tool_name,
                "message": "请求的工具不在本应用允许的工具列表中。",
            }
            status = "error"
        else:
            try:
                result = tool.invoke(tool_call["args"])
                status = "error" if result.get("status") == "error" else "success"
            except Exception as error:
                # 工具失败也作为协议消息回传模型；后续 Agent 才能决定重试、
                # 换工具，或在预算耗尽时交给受限结论报告。
                result = {
                    "status": "error",
                    "tool": tool_name,
                    "message": f"工具执行失败：{error}",
                }
                status = "error"

        tool_messages.append(
            ToolMessage(
                content=json.dumps(result, ensure_ascii=False, default=str),
                name=tool_name,
                tool_call_id=tool_call_id,
                status=status,
            )
        )

    return tool_messages

def tools_node(state: ResearchState) -> dict[str, Any]:
    """执行最近一轮 Tool Calling，并只追加相应的 ToolMessage。"""

    request_message = _latest_tool_call_message(state.get("messages", []))
    messages = state.get("messages", [])
    request_index = next(i for i in range(len(messages) - 1, -1, -1) if messages[i] is request_message)
    answered = {m.tool_call_id for m in messages[request_index + 1:] if isinstance(m, ToolMessage)}
    pending = [call for call in request_message.tool_calls if call["id"] not in answered]
    control_calls = [call for call in pending if call["name"] == ASK_USER_NAME]
    return {"messages": execute_tool_calls(pending, RESEARCH_TOOLS),
            "trace": [{"node": "tools", "event": "control_tool_not_executable",
                       "detail": "ask_user 误入研究执行层，已安全拒绝，未执行控制工具。"}] if control_calls else []}

def _recent_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    """取出本轮 tools 节点刚追加的连续 ToolMessage。"""

    recent_messages: list[ToolMessage] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        if message.name != ASK_USER_NAME and not message.additional_kwargs.get("protocol_only"):
            recent_messages.append(message)
    return list(reversed(recent_messages))

def _evidence_from_local_knowledge(message: ToolMessage, result: dict[str, Any]) -> list[EvidenceRecord]:
    """将本地检索工具的成功结果转换为统一的 EvidenceRecord。"""

    if result.get("status") != "success":
        return []

    evidence_records: list[EvidenceRecord] = []
    for index, item in enumerate(result.get("results", []), start=1):
        source = item.get("source", {})
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        source_location = source.get("location", "")
        # 同一知识库段落可能被模型以相近 query 重复检索。证据 ID 以内容和来源
        # 稳定生成，而不是使用一次性的 tool_call_id，才能跨轮识别它。
        stable_id_input = f"local_knowledge::{source_location}::{content}"
        evidence_id = sha256(stable_id_input.encode("utf-8")).hexdigest()[:24]
        evidence_records.append(
            {
                "evidence_id": evidence_id,
                **({"chunk_id": item["chunk_id"]} if item.get("chunk_id") else {}),
                "content": content,
                "source_display_name": source.get("display_name", "知识库：未标注来源"),
                "source_location": source_location,
                "source_type": "local_knowledge",
                "tool_name": message.name or "search_local_knowledge",
                **{key: source[key] for key in ("source_url", "source_page", "doi", "license",
                                               "content_kind", "source_scope", "extraction_note") if source.get(key)},
                **({"author_or_organization": source["source_author"]} if source.get("source_author") else {}),
                **({"published_or_observed_at": source["published_at"]} if source.get("published_at") else {}),
            }
        )
    return evidence_records

def _evidence_from_tool_message(
    message: ToolMessage,
) -> tuple[list[EvidenceRecord], str, str]:
    """按工具协议解析一个结果；未知或失败工具暂不伪造证据。"""

    try:
        result = json.loads(message.content)
    except (TypeError, json.JSONDecodeError):
        return [], "工具结果不是可解析的 JSON，未写入证据。", "tool_result_invalid"

    if not isinstance(result, dict):
        return [], "工具结果不是对象结构，未写入证据。", "tool_result_invalid"
    if result.get("status") != "success":
        return [], f"工具返回失败状态：{result.get('message', '未提供原因')}", "tool_failed"
    if message.name == "search_local_knowledge":
        records = _evidence_from_local_knowledge(message, result)
        return records, f"从本地知识库整理了 {len(records)} 条证据。", "evidence_extracted"
    if message.name == "get_current_weather":
        source = result.get("source", {})
        evidence_text = result.get("evidence_text")
        if not isinstance(evidence_text, str) or not evidence_text.strip():
            return [], "天气工具未返回可引用的规范化当前条件。", "tool_result_invalid"
        stable_id_input = f"weather::{source.get('location', '')}::{result.get('observed_at', '')}"
        evidence_id = sha256(stable_id_input.encode("utf-8")).hexdigest()[:24]
        record: EvidenceRecord = {
            "evidence_id": evidence_id,
            "content": evidence_text,
            "source_display_name": source.get("display_name", "Open-Meteo 当前条件"),
            "source_location": source.get("location", ""),
            "source_type": "weather",
            "tool_name": message.name,
            "author_or_organization": source.get("organization", "Open-Meteo"),
            "published_or_observed_at": result.get("observed_at", ""),
            "valid_at": result.get("valid_at"),
            "fetched_at": result.get("fetched_at"),
        }
        return [record], "从当前天气结果整理了 1 条证据。", "evidence_extracted"
    if message.name == "search_web":
        return [], (
            f"发现 {result.get('result_count', 0)} 个网页候选来源；"
            "候选摘要尚未作为正式证据。"
        ), "search_candidates_available"
    if message.name == "read_source":
        source = result.get("source", {})
        content = result.get("content")
        if not isinstance(content, str) or not content.strip():
            return [], "网页读取结果没有可引用正文。", "tool_result_invalid"
        stable_id_input = f"web::{source.get('location', '')}::{content}"
        evidence_id = sha256(stable_id_input.encode("utf-8")).hexdigest()[:24]
        record = {
            "evidence_id": evidence_id,
            "content": content,
            "source_display_name": source.get("display_name", "未标注网页来源"),
            "source_location": source.get("location", ""),
            "source_type": "web",
            "tool_name": message.name,
        }
        if source.get("author"):
            record["author_or_organization"] = source["author"]
        elif source.get("organization"):
            record["author_or_organization"] = source["organization"]
        if source.get("published_at"):
            record["published_or_observed_at"] = source["published_at"]
        if result.get("original_pdf_url"):
            record["original_pdf_url"] = result["original_pdf_url"]
        return [record], "从网页正文整理了 1 条证据。", "evidence_extracted"
    return [], f"工具 {message.name} 的证据整理规则尚未实现。", "tool_result_unhandled"

def record_results_node(state: ResearchState) -> dict[str, Any]:
    """把本轮工具结果整理为证据与轨迹，并更新程序维护的计数。"""

    tool_messages = _recent_tool_messages(state.get("messages", []))
    if not tool_messages:
        if any(isinstance(m, ToolMessage) and (m.name == ASK_USER_NAME or m.additional_kwargs.get("protocol_only"))
               for m in state.get("messages", [])[-1:]):
            return {"trace": [{"node": "record_results", "event": "control_result_ignored",
                               "detail": "控制工具回应不计研究轮次、工具次数，也不产生证据。"}]}
        raise ValueError("record_results 节点没有找到本轮 ToolMessage。")

    evidence_records: list[EvidenceRecord] = []
    known_evidence_ids = {
        record.get("evidence_id") for record in state.get("evidence", []) if record.get("evidence_id")
    }
    trace_entries: list[TraceEntry] = []
    for message in tool_messages:
        records, detail, outcome = _evidence_from_tool_message(message)
        try:
            result_diagnostics = json.loads(message.content)
            if not isinstance(result_diagnostics, dict):
                result_diagnostics = {}
        except (ValueError, TypeError):
            result_diagnostics = {}
        new_records = []
        for record in records:
            if record["evidence_id"] not in known_evidence_ids:
                new_records.append(record)
                known_evidence_ids.add(record["evidence_id"])
        duplicate_count = len(records) - len(new_records)
        if duplicate_count:
            detail = f"{detail} 跳过 {duplicate_count} 条重复证据。"
        evidence_records.extend(new_records)
        if new_records:
            event = "evidence_recorded"
        elif records:
            event = "evidence_duplicates_skipped"
        else:
            event = outcome
        trace_entries.append(
            {
                "node": "record_results",
                "event": event,
                "status": "error" if event in {"tool_failed", "tool_result_invalid"} else "success",
                "detail": detail,
                **({"source_diagnostics": result_diagnostics["diagnostics"]}
                   if message.name == "read_source" and isinstance(result_diagnostics.get("diagnostics"), dict) else {}),
                "tool_name": message.name,
                "returned_count": len(records),
                "accepted_count": len(new_records),
                "duplicate_count": duplicate_count,
                "relevance_gate": result_diagnostics.get("relevance_gate", {}),
                "top_scores": [item.get("retrieval", {}).get("rerank_score") for item in result_diagnostics.get("results", [])[:4]],
            }
        )

    # 这里返回的是更新后的总值，不是“本轮加几次”。
    return {
        "evidence": evidence_records,
        "tool_call_count": state.get("tool_call_count", 0) + len(tool_messages),
        "research_round": state.get("research_round", 0) + 1,
        "trace": trace_entries,
    }
