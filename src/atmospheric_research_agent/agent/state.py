"""研究 Agent 在整张 LangGraph 中共享的运行状态。

这里定义的是一次研究任务的“公共数据表”：节点从 State 读取自己需要的
信息，并只返回自己产生或更新的字段。它不包含模型、API 客户端或配置；
那些是应用的静态依赖，不属于一次运行会变化的数据。
"""

from __future__ import annotations

from datetime import UTC, datetime
from operator import add
from typing import Annotated, Literal
from uuid import uuid4

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


class ReportReview(TypedDict, total=False):
    """对报告草稿的结构化 Reflection / Verification 结果。

    Reflection 负责发现报告可能写得过头、遗漏限定或偏离问题的地方；
    Verification 则要求这些判断必须逐项对照已正式记录的 evidence。它不保存
    模型的隐藏推理，只保存后续修订节点能执行、用户也能审查的结论。
    """

    verdict: Literal["pass", "revise"]
    supported_points: list[str]
    unsupported_or_overstated_claims: list[str]
    citation_issues: list[str]
    required_changes: list[str]
    reason: str


class ProjectMemoryRecord(TypedDict, total=False):
    """跨运行保存的一张研究资料卡，只作背景，不作本轮 evidence。"""

    memory_id: str
    question: str
    summary: str
    source_locations: list[str]
    created_at: str
    status: Literal["verified_report"]


class TraceEntry(TypedDict, total=False):
    """一次运行中的结构化事件。

    ``event`` 说明发生了什么；``node`` 标明发生在哪个图节点；``timestamp``
    让事件可以按时间复盘。自动生成的节点生命周期事件还会携带
    ``duration_ms``。所有事件通过同一个 ``run_id`` 归属于一次用户请求。
    """

    node: str
    event: str
    detail: str
    timestamp: str
    status: Literal["started", "success"]
    duration_ms: float


class ResearchState(TypedDict, total=False):
    """一次研究任务在节点之间流动的全部动态数据。

    ``messages`` 保存模型消息与 ToolMessage，供 Agent Tool Calling 循环继续使用。
    ``evidence`` 和 ``trace`` 使用 ``Annotated[..., add]``：节点返回一个列表时，
    LangGraph 会把它追加到旧列表，而不是整段覆盖。
    其余字段采用默认“新值覆盖旧值”的规则。
    """

    # 用户输入：在开始一次新研究任务时必须提供。
    question: str

    # 一次 invoke 的关联标识。未来写入日志、数据库或监控平台时，可用它把同一
    # 用户请求的所有事件、错误与指标重新串起来。
    run_id: str
    started_at: str
    finished_at: str | None
    run_duration_ms: float | None

    # Tool Calling 协议消息。每个 Agent 或工具节点只追加本轮新增消息。
    messages: Annotated[list[BaseMessage], add]

    # Agent 逐步更新的研究意图与当前方向。
    research_plan: ResearchPlan

    # START 后固定读取的历史研究摘要。它不使用 evidence reducer，避免被误认为
    # 本次可引用的正式资料。
    recalled_memories: list[ProjectMemoryRecord]

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

    # V2：报告先作为草稿接受 Reflection / Verification；只有通过检查或完成
    # 受控修订后，finalize_report 才把它写入 final_report。
    draft_report: str | None
    report_review: ReportReview
    report_revision_count: int

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

    started_at = datetime.now(UTC).isoformat()
    run_id = str(uuid4())
    return {
        "question": clean_question,
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": None,
        "run_duration_ms": None,
        "messages": [],
        "recalled_memories": [],
        "evidence": [],
        "research_round": 0,
        "tool_call_count": 0,
        "clarification_question": None,
        "final_report": None,
        "draft_report": None,
        "report_revision_count": 0,
        "trace": [
            {
                "node": "run",
                "event": "run_started",
                "detail": "已创建本次研究任务，等待进入 LangGraph。",
                "timestamp": started_at,
                "status": "started",
            }
        ],
    }
