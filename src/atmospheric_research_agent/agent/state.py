"""研究 Agent 在整张 LangGraph 中共享的运行状态。

这里定义的是一次研究任务的“公共数据表”：节点从 State 读取自己需要的
信息，并只返回自己产生或更新的字段。它不包含模型、API 客户端或配置；
那些是应用的静态依赖，不属于一次运行会变化的数据。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage


class ResearchPlan(TypedDict, total=False):
    """模型为当前问题形成的研究计划，而不是实际工具调用的记录。"""

    goal: str
    focus_points: list[str]
    known_conditions: list[str]
    missing_conditions: list[str]
    current_direction: str


class EvidenceRecord(TypedDict, total=False):
    """已被整理为可用于报告的证据条目。"""

    evidence_id: str
    content: str
    source_display_name: str
    source_location: str
    source_type: Literal["local_knowledge", "weather", "web"]
    tool_name: str
    author_or_organization: str
    published_or_observed_at: str


class EvidenceAssessment(TypedDict, total=False):
    """对当前证据是否足以回答问题的结构化判断。"""

    sufficient: bool
    covered_points: list[str]
    missing_points: list[str]
    potential_conflicts: list[str]
    next_step: Literal["continue", "clarify", "full_report", "limited_report"]
    reason: str


class TraceEntry(TypedDict, total=False):
    """面向调试与复盘的简短运行记录。"""

    node: str
    event: str
    detail: str


class ResearchState(TypedDict, total=False):
    """一次研究任务在节点之间流动的全部动态数据。

    ``messages`` 保存模型消息与 ToolMessage，供 Agent Tool Calling 循环继续使用。
    ``evidence`` 和 ``trace`` 使用 ``Annotated[..., add]``：节点返回一个列表时，
    LangGraph 会把它追加到旧列表，而不是整段覆盖。
    其余字段采用默认“新值覆盖旧值”的规则。
    """

    # 用户输入：在开始一次新研究任务时必须提供。
    question: str

    # Tool Calling 协议消息。每个 Agent 或工具节点只追加本轮新增消息。
    messages: Annotated[list[BaseMessage], add]

    # Agent 逐步更新的研究意图与当前方向。
    research_plan: ResearchPlan

    # record_results 节点从工具结果中整理出的正式证据。
    evidence: Annotated[list[EvidenceRecord], add]

    # 程序负责计数；节点返回更新后的总值，而不是“加多少”。
    research_round: int
    tool_call_count: int

    # assess_evidence 的最近一次判断。
    assessment: EvidenceAssessment

    # 无法继续时向用户提出的问题；用户补充后应开始新的运行。
    clarification_question: str | None

    # generate_report 节点产出的完整或受限结论报告。
    final_report: str | None

    # 保留简短轨迹，供调试、学习和后续可观测性扩展使用。
    trace: Annotated[list[TraceEntry], add]


def create_initial_state(question: str) -> ResearchState:
    """为一次新研究任务创建明确的初始 State。

    LangGraph 不会自动替未传入的字段补默认值。因此入口程序应调用这个函数，
    或等价地传入同样的初始字段，再执行图。
    """

    clean_question = question.strip()
    if not clean_question:
        raise ValueError("研究问题不能为空。")

    return {
        "question": clean_question,
        "messages": [],
        "evidence": [],
        "research_round": 0,
        "tool_call_count": 0,
        "clarification_question": None,
        "final_report": None,
        "trace": [],
    }
