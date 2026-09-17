"""声明并编译大气科学研究 Agent 的 LangGraph。"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes import (
    agent_node,
    assess_evidence_node,
    generate_report_node,
    record_results_node,
    request_clarification_node,
    research_guard_node,
    route_after_agent,
    route_after_evidence_assessment,
    route_after_research_guard,
    tools_node,
)
from .state import ResearchState


def build_research_graph() -> Any:
    """构建一张可执行的图；模型和工具仅在 invoke 时才会真正调用。"""

    # 当前 V1 的完整研究流程图：
    #
    # START
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
    #   ├── 新工具调用会超出预算 ──────→ generate_report
    #   │                                  ↓
    #   │                                END：受限结论报告
    #   │
    #   └── 暂不调用工具 ──────────────→ research_guard
    #                                      │
    #                                      ├── evidence 为空、预算尚余 → agent
    #                                      ├── evidence 为空、预算耗尽 → generate_report → END：受限结论
    #                                      └── evidence 非空 → assess_evidence
    #                                                               │
    #                                                               ├── continue → agent
    #                                                               ├── clarify → request_clarification → END：等待用户
    #                                                               └── full_report / limited_report
    #                                                                            → generate_report → END：研究报告
    #
    # 关键数据流：
    # messages 只保存模型与工具的协议消息；record_results 把合格工具结果整理为
    # evidence；assess_evidence 只评估 evidence；generate_report 只引用 evidence。

    builder = StateGraph(ResearchState)

    # 决策中心：让模型决定本轮是否调用工具、需要澄清，或交给证据守卫处理。
    builder.add_node("agent", agent_node)

    # 执行层：逐个执行 Agent 请求的工具，并把结果包装为 ToolMessage。
    builder.add_node("tools", tools_node)

    # 证据层：把工具结果整理、去重为可引用 evidence，并更新运行计数与轨迹。
    builder.add_node("record_results", record_results_node)

    # 程序守卫：禁止零证据直接结束；决定继续研究、评估证据或受限结论。
    builder.add_node("research_guard", research_guard_node)

    # 质量判断：模型以结构化格式判断证据覆盖、缺口、冲突与建议下一步。
    builder.add_node("assess_evidence", assess_evidence_node)

    # 输出层：基于正式证据生成完整或受限报告，并由程序追加来源清单。
    builder.add_node("generate_report", generate_report_node)

    # 人在回路：保留具体追问，以“等待用户补充信息”结束本轮任务。
    builder.add_node("request_clarification", request_clarification_node)

    builder.add_edge(START, "agent")

    # Agent 的模型结果决定第一层分流。
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "request_clarification": "request_clarification",
            "research_guard": "research_guard",
            "generate_report": "generate_report",
        },
    )

    # 工具执行后必须先整理证据，再回到 Agent 决定下一轮动作。
    builder.add_edge("tools", "record_results")
    builder.add_edge("record_results", "agent")

    # Guard 只处理零证据终止问题。
    builder.add_conditional_edges(
        "research_guard",
        route_after_research_guard,
        {
            "assess_evidence": "assess_evidence",
            "agent": "agent",
            "generate_report": "generate_report",
        },
    )

    # 证据评估后再决定继续、澄清或写报告。
    builder.add_conditional_edges(
        "assess_evidence",
        route_after_evidence_assessment,
        {
            "agent": "agent",
            "request_clarification": "request_clarification",
            "generate_report": "generate_report",
        },
    )

    # 两种报告模式都结束；澄清路径则以“等待用户补充”结束本轮。
    builder.add_edge("generate_report", END)
    builder.add_edge("request_clarification", END)

    return builder.compile()
