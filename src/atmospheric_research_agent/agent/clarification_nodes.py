"""人工澄清与恢复节点。"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from .state import ResearchState
from .evidence import is_stale_weather
from .clock import reference_now


def request_clarification_node(state: ResearchState) -> dict[str, Any]:
    """暂停同一任务；恢复时节点重入，因此 interrupt 前不调用模型、不更新计数。"""

    clarification_question = state.get("clarification_question")
    if not clarification_question or not clarification_question.strip():
        raise ValueError("request_clarification 需要非空 clarification_question。")

    answer = interrupt({"question": clarification_question, "run_id": state.get("run_id")})
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("澄清回答必须是非空文本。")
    answer = answer.strip()
    stale_count = sum(is_stale_weather(item, now=reference_now(state)) for item in state.get("evidence", []))
    return {
        "clarification_question": None,
        "clarification_history": [{"question": clarification_question, "answer": answer}],
        "messages": [HumanMessage(content=f"针对追问：{clarification_question}\n我的补充：{answer}")],
        "assessment": {},
        "trace": [
            {
                "node": "request_clarification",
                "event": "clarification_received",
                "detail": f"已接收用户补充，保留原任务证据与预算；当前有 {stale_count} 条过期/时间未知天气证据不再参与报告，按需重新查询。",
            }
        ]
    }
