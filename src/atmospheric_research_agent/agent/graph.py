"""声明并编译大气科学研究 Agent 的 LangGraph。"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import wraps
from time import perf_counter
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from .nodes import (
    agent_node,
    assess_evidence_node,
    finalize_limited_report_node,
    finalize_report_node,
    generate_draft_node,
    record_results_node,
    recall_memory_node,
    request_clarification_node,
    revise_report_node,
    research_guard_node,
    route_after_agent,
    route_after_evidence_assessment,
    route_after_research_guard,
    route_after_report_verification,
    tools_node,
    store_memory_node,
    verify_report_node,
)
from .state import ResearchState


def _utc_now() -> str:
    """返回统一的 UTC 时间，便于未来跨机器、跨服务关联运行记录。"""

    return datetime.now(UTC).isoformat()


def _instrument_node(node_name: str, node: Callable[[ResearchState], dict[str, Any]]) -> Callable:
    """为任意业务节点补齐统一的开始、完成和耗时 Trace。

    业务节点仍只关心自己的领域事件，例如 ``evidence_recorded``；这个包装层
    负责通用运行观测，因此新增节点时不会漏掉基础计时与生命周期记录。
    """

    @wraps(node)
    def instrumented_node(state: ResearchState) -> dict[str, Any]:
        started_at = _utc_now()
        started = perf_counter()
        update = node(state)
        if not isinstance(update, dict):
            raise TypeError(f"节点 {node_name} 必须返回 dict 类型的 State 增量。")

        completed_at = _utc_now()
        duration_ms = round((perf_counter() - started) * 1000, 2)

        # 旧业务节点返回的 trace 统一补齐完成时间和成功状态；它们仍保留各自的
        # event/detail，不会被自动生命周期事件取代。
        business_trace = [
            {
                **entry,
                "timestamp": entry.get("timestamp", completed_at),
                "status": entry.get("status", "success"),
            }
            for entry in update.get("trace", [])
        ]

        return {
            **update,
            "trace": [
                {
                    "node": node_name,
                    "event": "node_started",
                    "detail": "节点开始执行。",
                    "timestamp": started_at,
                    "status": "started",
                },
                *business_trace,
                {
                    "node": node_name,
                    "event": "node_completed",
                    "detail": "节点执行完成。",
                    "timestamp": completed_at,
                    "status": "success",
                    "duration_ms": duration_ms,
                },
            ],
        }

    return instrumented_node


def build_research_graph() -> Any:
    """构建一张可执行的图；模型和工具仅在 invoke 时才会真正调用。"""

    # 当前 V2 的完整研究流程图：
    #
    # START
    #   ↓
    # recall_memory ── 读取相关历史摘要；它只提供背景，绝不成为本次 evidence
    #   ↓
    # agent  ── 模型根据 question、messages、evidence 做本轮决策
    #   │
    #   ├── 有 tool_calls ─────────────→ tools
    #   │                                  ↓
    #   │                            record_results
    #   │                                  ↓
    #   │                               agent     （工具结果进入 messages，继续循环）
    #   │
    #   ├── 缺少用户才能提供的信息 ────→ request_clarification
    #   │                                  ↓
    #   │                                END：等待用户补充信息
    #   │
    #   └── 暂不调用工具 / 新工具调用超预算 → research_guard
    #                                      │
    #                                      ├── evidence 为空、预算尚余 → agent
    #                                      ├── evidence 为空、预算耗尽 → finalize_limited_report → END
    #                                      └── evidence 非空 → assess_evidence
    #                                                               │
    #                                                               ├── continue → agent
    #                                                               ├── clarify → request_clarification → END：等待用户
    #                                                               └── full_report / limited_report
    #                                                                            → generate_draft
    #                                                                                 ↓
    #                                                                            verify_report
    #                                                                             ├── pass → finalize_report → store_memory → END
    #                                                                             └── revise → revise_report → verify_report
    #                                                                                              │
    #                                                                                  超过修订预算仍 revise
    #                                                                                              ↓
    #                                                                              finalize_limited_report → END
    #
    # 关键数据流：
    # messages 只保存模型与工具的协议消息；record_results 把合格工具结果整理为
    # evidence；assess_evidence 只评估 evidence；草稿、审查与修订都只读取
    # evidence；finalize_report 由程序统一附加来源。

    builder = StateGraph(ResearchState)

    # 任务记忆：固定在入口读取，不交给模型临时决定；只作背景，不作本轮证据。
    builder.add_node("recall_memory", _instrument_node("recall_memory", recall_memory_node))

    # 决策中心：让模型决定本轮是否调用工具、需要澄清，或交给证据守卫处理。
    builder.add_node("agent", _instrument_node("agent", agent_node))

    # 执行层：逐个执行 Agent 请求的工具，并把结果包装为 ToolMessage。
    builder.add_node("tools", _instrument_node("tools", tools_node))

    # 证据层：把工具结果整理、去重为可引用 evidence，并更新运行计数与轨迹。
    builder.add_node("record_results", _instrument_node("record_results", record_results_node))

    # 程序守卫：禁止零证据直接结束；决定继续研究、评估证据或受限结论。
    builder.add_node("research_guard", _instrument_node("research_guard", research_guard_node))

    # 质量判断：模型以结构化格式判断证据覆盖、缺口、冲突与建议下一步。
    builder.add_node("assess_evidence", _instrument_node("assess_evidence", assess_evidence_node))

    # 输出层第一步：先生成草稿，不直接交付。
    builder.add_node("generate_draft", _instrument_node("generate_draft", generate_draft_node))

    # Reflection / Verification：核验主张、引用和限定是否忠于正式证据。
    builder.add_node("verify_report", _instrument_node("verify_report", verify_report_node))

    # 受控修订，以及两种最终交付节点。
    builder.add_node("revise_report", _instrument_node("revise_report", revise_report_node))
    builder.add_node("finalize_report", _instrument_node("finalize_report", finalize_report_node))
    builder.add_node(
        "finalize_limited_report",
        _instrument_node("finalize_limited_report", finalize_limited_report_node),
    )

    # 人在回路：保留具体追问，以“等待用户补充信息”结束本轮任务。
    builder.add_node(
        "request_clarification",
        _instrument_node("request_clarification", request_clarification_node),
    )

    # 只保存已经通过 Reflection / Verification 的最终报告。
    builder.add_node("store_memory", _instrument_node("store_memory", store_memory_node))

    builder.add_edge(START, "recall_memory")
    builder.add_edge("recall_memory", "agent")

    # Agent 的模型结果决定第一层分流。
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "request_clarification": "request_clarification",
            "research_guard": "research_guard",
        },
    )

    # 工具执行后必须先整理证据，再回到 Agent 决定下一轮动作。
    builder.add_edge("tools", "record_results")
    builder.add_edge("record_results", "agent")

    # Guard 只处理零证据终止问题；零证据不能进入草稿生成。
    builder.add_conditional_edges(
        "research_guard",
        route_after_research_guard,
        {
            "assess_evidence": "assess_evidence",
            "agent": "agent",
            "finalize_limited_report": "finalize_limited_report",
        },
    )

    # 证据评估后再决定继续、澄清或写报告草稿。
    builder.add_conditional_edges(
        "assess_evidence",
        route_after_evidence_assessment,
        {
            "agent": "agent",
            "request_clarification": "request_clarification",
            "generate_draft": "generate_draft",
        },
    )

    # 草稿必须先审查；修订后回到审查，最多一次；仍未通过则安全降级。
    builder.add_edge("generate_draft", "verify_report")
    builder.add_conditional_edges(
        "verify_report",
        route_after_report_verification,
        {
            "finalize_report": "finalize_report",
            "revise_report": "revise_report",
            "finalize_limited_report": "finalize_limited_report",
        },
    )
    builder.add_edge("revise_report", "verify_report")

    # 两种最终交付路径与澄清路径均结束本轮。
    builder.add_edge("finalize_report", "store_memory")
    builder.add_edge("store_memory", END)
    builder.add_edge("finalize_limited_report", END)
    builder.add_edge("request_clarification", END)

    return builder.compile()
