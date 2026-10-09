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

    # 本轮研究最终想回答什么；通常是对用户问题的更明确表述。
    goal: str
    # 研究时应优先覆盖的几个机制、条件或子问题。
    focus_points: list[str]
    # 用户已经明确给出、因此不需要重新猜测的地点、时间或其他前提。
    known_conditions: list[str]
    # 继续研究或可靠回答前仍缺少的关键条件。
    missing_conditions: list[str]
    # Agent 当前正在沿着哪条研究方向检索，便于下一轮延续上下文。
    current_direction: str


class EvidenceRecord(TypedDict, total=False):
    """已被整理为可用于报告的证据条目。"""

    # 程序根据来源与正文生成的稳定 ID；报告中的 [E1] 等编号由它排序得来。
    evidence_id: str
    # 本次确实可支持报告主张的正文片段，而不是网页搜索摘要。
    content: str
    # 给人阅读的来源名称，例如“知识库：大气动力学 > 急流 > 形成机制”。
    source_display_name: str
    # 可追溯的原始位置：本地 Markdown 相对路径、URL 或天气地点标识。
    source_location: str
    # 来源属于内部知识库、当前天气结果还是公开网页；影响后续解释与引用边界。
    source_type: Literal["local_knowledge", "weather", "web"]
    # 哪个 Tool 产出了这条证据，便于 Trace、调试和来源审计。
    tool_name: str
    # 网页作者、机构，或天气数据提供机构；工具能获得时才填写。
    author_or_organization: str
    # 网页发表时间或天气观测时间；没有可靠值时不填写。
    published_or_observed_at: str


class EvidenceAssessment(TypedDict, total=False):
    """对当前证据是否足以回答问题的结构化判断。"""

    # 当前证据是否已经足以支持可靠回答；不是“模型是否想结束”的随意意见。
    sufficient: bool
    # 已有证据覆盖的关键要点，帮助后续报告只写已被支持的内容。
    covered_points: list[str]
    # 仍缺什么证据、机制、地点或时间信息；未来可变成定向补查任务。
    missing_points: list[str]
    # 看似矛盾的证据或结论，以及需要进一步区分的尺度、条件或来源差异。
    potential_conflicts: list[str]
    # 条件路由的建议：继续研究、询问用户、完整报告或受限报告。
    next_step: Literal["continue", "clarify", "full_report", "limited_report"]
    # 以上判断的简短可审查理由，显示给 Trace 和后续节点。
    reason: str


class ReportReview(TypedDict, total=False):
    """对报告草稿的结构化 Reflection / Verification 结果。

    Reflection 负责发现报告可能写得过头、遗漏限定或偏离问题的地方；
    Verification 则要求这些判断必须逐项对照已正式记录的 evidence。它不保存
    模型的隐藏推理，只保存后续修订节点能执行、用户也能审查的结论。
    """

    # 核验是否允许交付草稿：pass 可定稿；revise 必须先修订或安全降级。
    verdict: Literal["pass", "revise"]
    # 审查器确认已被本轮正式 evidence 支持的关键结论。
    supported_points: list[str]
    # 没有证据、过度推断或超出适用范围的主张。
    unsupported_or_overstated_claims: list[str]
    # 不存在的 [E#]、缺失引用或引用与文本不匹配等问题。
    citation_issues: list[str]
    # 交给 revise_report 的可执行修改要求，例如删除、弱化或补充限定。
    required_changes: list[str]
    # 对本次 pass / revise 的总体简短说明。
    reason: str


class ProjectMemoryRecord(TypedDict, total=False):
    """跨运行保存的一张研究资料卡，只作背景，不作本轮 evidence。"""

    # 这张资料卡自己的唯一 ID；不同于一次运行的 run_id。
    memory_id: str
    # 当时用户提出的原始研究问题，用于后续相关性召回。
    question: str
    # 已核验最终报告的压缩正文摘要；只作背景，不能当作本轮正式证据。
    summary: str
    # 这份历史研究使用过的来源位置，帮助追溯但不会自动写进本轮引用。
    source_locations: list[str]
    # 资料卡写入本地持久化文件的 UTC 时间。
    created_at: str
    # 当前只允许保存已通过 Verification 的报告，避免错误草稿污染长期记忆。
    status: Literal["verified_report"]


class TraceEntry(TypedDict, total=False):
    """一次运行中的结构化事件。

    ``event`` 说明发生了什么；``node`` 标明发生在哪个图节点；``timestamp``
    让事件可以按时间复盘。自动生成的节点生命周期事件还会携带
    ``duration_ms``。所有事件通过同一个 ``run_id`` 归属于一次用户请求。
    """

    # 产生事件的图节点，例如 agent、tools、verify_report。
    node: str
    # 机器可筛选的事件类型，例如 node_started、tool_failed、report_finalized。
    event: str
    # 给人阅读的具体说明；不存模型隐藏思维或密钥。
    detail: str
    # 事件发生的 UTC 时间，便于按一次运行的顺序复盘。
    timestamp: str
    # 节点生命周期状态；当前自动记录 started 与 success。
    status: Literal["started", "success"]
    # 节点或整次运行的耗时，单位毫秒；只有完成事件才有。
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
    # 本次图运行创建时的 UTC 时间。
    started_at: str
    # 图结束时的 UTC 时间；刚创建 State 时为 None。
    finished_at: str | None
    # 从开始到 END 的总耗时，单位毫秒；刚创建 State 时为 None。
    run_duration_ms: float | None

    # Tool Calling 协议消息。每个 Agent 或工具节点只追加本轮新增消息。
    # 它包含 HumanMessage、AIMessage、ToolMessage，是模型下一轮真正看到的对话上下文。
    messages: Annotated[list[BaseMessage], add]

    # Agent 逐步更新的研究意图与当前方向；它是计划，不是已经获得的事实。
    research_plan: ResearchPlan

    # START 后固定读取的历史研究摘要。它不使用 evidence reducer，避免被误认为
    # 本次可引用的正式资料。
    recalled_memories: list[ProjectMemoryRecord]

    # record_results 节点从工具结果中整理出的正式证据；只有这里的内容才能支持报告引用。
    evidence: Annotated[list[EvidenceRecord], add]

    # 已执行过几批工具调用；用于限制 Agent Loop，避免无限研究。
    research_round: int
    # 已实际执行的工具调用总数；不同于 research_round，一批可含多个 Tool。
    tool_call_count: int

    # Evidence Review 节点的最近一次判断；决定继续、澄清还是进入报告阶段。
    assessment: EvidenceAssessment

    # 只有用户才能提供关键条件时的追问；不为空时本轮在 request_clarification 结束。
    clarification_question: str | None

    # 最终交付给用户的报告；只有 finalize 节点可写入，草稿不能直接写这里。
    final_report: str | None

    # Report Writer 生成、但尚未交付的 Markdown 草稿。
    draft_report: str | None
    # Report Verification 对当前 draft_report 的结构化审查结果。
    report_review: ReportReview
    # 当前草稿已经受控修订过几次；用于限制“修订 → 再核验”循环。
    report_revision_count: int

    # 运行期间追加的结构化轨迹；用于调试、学习和后续可观测性扩展。
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
