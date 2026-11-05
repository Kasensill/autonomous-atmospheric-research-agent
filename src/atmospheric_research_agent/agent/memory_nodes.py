"""长期记忆召回与写入节点。"""
from __future__ import annotations

from typing import Any


from .memory import make_memory_record, recall_project_memory, save_project_memory, is_time_sensitive_question
from .state import ResearchState, TraceEntry
from .evidence import has_user_materials


from .node_shared import _task_question
def recall_memory_node(state: ResearchState) -> dict[str, Any]:
    """START 后读取相关项目记忆；首次或无关任务可自然返回空列表。"""

    memories = recall_project_memory(state["question"])
    return {
        "recalled_memories": memories,
        "trace": [
            {
                "node": "recall_memory",
                "event": "memory_recalled",
                "detail": f"召回 {len(memories)} 条相关项目记忆；仅作为背景，不计入正式证据。",
            }
        ],
    }


def store_memory_node(state: ResearchState) -> dict[str, list[TraceEntry]]:
    """仅持久化明确为 full、通过核验且非时效性的最终报告。"""

    final_report = state.get("final_report")
    review = state.get("report_review")
    if not isinstance(final_report, str) or not final_report.strip():
        raise ValueError("store_memory 需要已生成的 final_report。")
    if not review or review.get("verdict") != "pass":
        raise ValueError("store_memory 只能保存已通过 Verification 的报告。")

    if has_user_materials(state):
        return {"trace": [{"node": "store_memory", "event": "memory_skipped",
                           "detail": "含用户材料，本次运行不写入长期记忆。"}]}
    mode = state.get("report_mode")
    if mode != "full":
        reason = ("受限报告证据不全，不写入长期记忆" if mode == "limited"
                  else "报告模式未明确为 full，不写入长期记忆")
        return {"trace": [{"node": "store_memory", "event": "memory_skipped", "detail": reason}]}

    if (is_time_sensitive_question(_task_question(state))
            or any(item.get("source_type") == "weather" for item in state.get("evidence", []))):
        return {"trace": [{"node": "store_memory", "event": "memory_skipped",
                           "detail": "时效性天气任务不写入长期项目记忆；任务结果仍正常交付。"}]}

    record = make_memory_record(
        question=_task_question(state),
        final_report=final_report,
        evidence=state.get("evidence", []),
        report_mode=mode,
    )
    record["run_id"] = state["run_id"]
    save_project_memory(record)
    return {
        "trace": [
            {
                "node": "store_memory",
                "event": "memory_stored",
                "detail": (
                    f"已保存 1 条通过核验的项目记忆（{record['memory_id']}），"
                    "供后续相关任务作为背景召回。"
                ),
            }
        ]
    }
