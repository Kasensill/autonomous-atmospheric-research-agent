"""研究续行空转与恢复后步数的离线回归；不调用外部模型或工具。"""

import json
from itertools import product
from math import ceil

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from atmospheric_research_agent.agent import nodes
import atmospheric_research_agent.agent.graph as graph_module
from atmospheric_research_agent.agent.settings import (
    GRAPH_RECURSION_LIMIT, MAX_RESEARCH_ROUNDS, MAX_REPORT_REVISION_ROUNDS, MAX_TOOL_CALLS,
)
from atmospheric_research_agent.agent.state import create_initial_state


def evidence():
    return {"evidence_id": "one", "source_type": "local_knowledge", "source_location": "test.md",
            "source_display_name": "测试资料", "content": "有一般机制证据，尚缺本地实况。"}


class ContinueModel:
    def __init__(self):
        self.inputs = []

    def invoke(self, messages):
        self.inputs.append(messages)
        return nodes.EvidenceAssessmentOutput(
            sufficient=False, covered_points=["机制"], missing_points=["本地实况"],
            potential_conflicts=[], next_step="continue", reason="缺少本地实况和边界层证据。",
        )


def configure_graph(monkeypatch, *, tool_rounds, response="valid", assess_each_batch=False, revisions=0,
                    batch_size=1, clarify=True):
    """mock 只替换模型/工具 I/O；守卫、路由、计数及规范化均使用真实实现。"""
    monkeypatch.setattr(graph_module, "recall_memory_node", lambda _: {})
    agent_calls = []

    class Agent:
        def invoke(self, messages):
            agent_calls.append(messages)
            if clarify and not any("我的补充：" in str(m.content) for m in messages):
                return AIMessage(content="", tool_calls=[{
                    "id": "clarify", "name": "ask_user", "args": {"question": "地点日期？"},
                }])
            completed_batches = sum(isinstance(m, ToolMessage) and m.name == "search_local_knowledge" for m in messages) // batch_size
            just_got_tools = isinstance(messages[-1], ToolMessage) and messages[-1].name == "search_local_knowledge"
            if completed_batches < tool_rounds and not (assess_each_batch and just_got_tools):
                return AIMessage(content="", tool_calls=[{
                    "id": f"research-{completed_batches}-{index}", "name": "search_local_knowledge", "args": {"query": "实况"},
                } for index in range(batch_size)])
            content = ('{"decision":"assess","plan_update":{}}' if response == "valid"
                       else "没有工具，暂时无法取得更多资料。")
            return AIMessage(content=content)

    def tools(state):
        calls = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage)).tool_calls
        return {"messages": [ToolMessage(name=call["name"], tool_call_id=call["id"], content=json.dumps({
            "status": "success", "results": [{"content": "有一般机制证据，尚缺本地实况。",
            "source": {"display_name": "测试资料", "location": "test.md"}}],
        }, ensure_ascii=False)) for call in calls]}

    class Writer:
        def invoke(self, messages):
            return AIMessage(content="仅能解释机制，缺少事件实况。[E1]")

    class Reviewer:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            revise = self.calls <= revisions
            return nodes.ReportReviewOutput(
                verdict="revise" if revise else "pass", supported_points=[],
                unsupported_or_overstated_claims=[], citation_issues=[],
                required_changes=["收紧表述"] if revise else [], reason="离线审查。",
            )

    assessment = ContinueModel()
    reviewer = Reviewer()
    monkeypatch.setattr(nodes, "create_research_model", Agent)
    monkeypatch.setattr(nodes, "create_assessment_model", lambda: assessment)
    monkeypatch.setattr(graph_module, "tools_node", tools)
    monkeypatch.setattr(graph_module, "generate_draft_node", lambda s: nodes.generate_draft_with_model(s, Writer()))
    monkeypatch.setattr(graph_module, "verify_report_node", lambda s: nodes.verify_report_with_model(s, reviewer))
    monkeypatch.setattr(graph_module, "revise_report_node", lambda s: nodes.revise_report_with_model(s, Writer()))
    monkeypatch.setattr(graph_module, "store_memory_node", lambda _: {})
    return assessment


@pytest.mark.parametrize("response", ["valid", "invalid"])
@pytest.mark.parametrize("assess_each_batch", [False, True])
def test_three_tool_batches_then_no_tool_continue_terminates_limited(monkeypatch, response, assess_each_batch):
    assessment = configure_graph(
        monkeypatch, tool_rounds=MAX_RESEARCH_ROUNDS - 1, response=response, batch_size=2,
        assess_each_batch=assess_each_batch, revisions=MAX_REPORT_REVISION_ROUNDS,
    )
    graph = graph_module.build_research_graph(checkpointer=InMemorySaver())
    state = create_initial_state("分析强降水")
    config = {"recursion_limit": GRAPH_RECURSION_LIMIT, "configurable": {"thread_id": state["run_id"]}}
    paused = graph.invoke(state, config)
    assert paused["__interrupt__"]
    assert paused["research_round"] == paused["tool_call_count"] == 0
    result = graph.invoke(Command(resume="济宁，指定日期"), config)
    assert result["research_round"] == MAX_RESEARCH_ROUNDS
    assert result["tool_call_count"] == 2 * (MAX_RESEARCH_ROUNDS - 1)
    assert len(result["clarification_history"]) == 1
    assert result["report_mode"] == "limited"
    assert "受限报告：当前证据不全" in result["final_report"]
    idle, = [e for e in result["trace"] if e["event"] == "idle_round_counted"]
    assert idle["research_round_before"] == MAX_RESEARCH_ROUNDS - 1
    assert idle["research_round_after"] == MAX_RESEARCH_ROUNDS
    assert idle["remaining_research_rounds"] == 0
    assert idle["remaining_tool_calls"] == MAX_TOOL_CALLS - 2 * (MAX_RESEARCH_ROUNDS - 1)
    assert result["assessment"]["next_step"] == "limited_report"
    assert result["report_revision_count"] == MAX_REPORT_REVISION_ROUNDS
    assert len(assessment.inputs) == (MAX_RESEARCH_ROUNDS if assess_each_batch else 2)
    assert "研究预算：已耗尽，不得选择 continue。" in assessment.inputs[-1][1].content
    if response == "invalid":
        assert any(e["event"] == "invalid_no_tool_response" for e in result["trace"])


@pytest.mark.parametrize("rounds,calls", [(MAX_RESEARCH_ROUNDS, 0), (0, MAX_TOOL_CALLS)])
def test_continue_is_normalized_when_either_budget_is_exhausted(rounds, calls):
    state = create_initial_state("研究")
    state.update(evidence=[evidence()], research_round=rounds, tool_call_count=calls)
    update = nodes.assess_evidence_with_model(state, ContinueModel())
    assert update["assessment"]["next_step"] == "limited_report"
    assert update["assessment"]["sufficient"] is False
    assert any(e["event"] == "assessment_next_step_normalized" for e in update["trace"])


@pytest.mark.parametrize("prior_assessment,latest,expected", [
    ({}, AIMessage(content="{}"), False),
    ({"next_step": "continue"}, AIMessage(content="{}"), True),
    ({"next_step": "continue"}, AIMessage(content="非法 JSON"), True),
    ({"next_step": "continue"}, AIMessage(content=""), True),
    ({"next_step": "full_report"}, AIMessage(content="{}"), False),
    ({"next_step": "continue"}, AIMessage(content="", tool_calls=[{
        "id": "tool", "name": "search_local_knowledge", "args": {"query": "检索"},
    }]), False),
    ({"next_step": "continue"}, AIMessage(content="", tool_calls=[{
        "id": "ask", "name": "ask_user", "args": {"question": "地点？"},
    }]), False),
])
def test_guard_counts_only_no_tool_continuation(prior_assessment, latest, expected):
    state = create_initial_state("研究")
    state.update(evidence=[evidence()], assessment=prior_assessment, messages=[latest], research_round=1)
    update = nodes.research_guard_node(state)
    assert update.get("research_round", state["research_round"]) == 1 + expected
    assert any(e["event"] == "idle_round_counted" for e in update["trace"]) == expected


def test_stale_continue_after_successful_tool_batch_does_not_double_count():
    state = create_initial_state("研究")
    state.update(evidence=[evidence()], assessment={"next_step": "continue"}, research_round=2,
                 messages=[ToolMessage(content="资料", name="search_local_knowledge", tool_call_id="one"),
                           AIMessage(content="{}")])
    update = nodes.research_guard_node(state)
    assert "research_round" not in update
    assert all(e["event"] != "idle_round_counted" for e in update["trace"])


def test_empty_evidence_guard_still_counts_exactly_one_attempt():
    state = create_initial_state("研究")
    state.update(assessment={"next_step": "continue"}, research_round=1, messages=[AIMessage(content="{}")])
    update = nodes.research_guard_node(state)
    assert update["research_round"] == 2
    assert [e["event"] for e in update["trace"]] == ["empty_evidence_blocked"]


def test_normal_research_path_does_not_add_idle_round(monkeypatch):
    assessment = configure_graph(monkeypatch, tool_rounds=1)

    def full_report(messages):
        return nodes.EvidenceAssessmentOutput(
            sufficient=True, covered_points=["机制"], missing_points=[], potential_conflicts=[],
            next_step="full_report", reason="证据足够。",
        )

    monkeypatch.setattr(assessment, "invoke", full_report)
    graph = graph_module.build_research_graph(checkpointer=InMemorySaver())
    initial = create_initial_state("研究")
    config = {"recursion_limit": GRAPH_RECURSION_LIMIT, "configurable": {"thread_id": initial["run_id"]}}
    graph.invoke(initial, config)
    result = graph.invoke(Command(resume="地点时间"), config)
    assert result["report_mode"] == "full"
    assert result["research_round"] == result["tool_call_count"] == 1
    assert not any(e["event"] == "idle_round_counted" for e in result["trace"])


@pytest.mark.parametrize("clarify", [False, True], ids=["without-clarification", "after-resume"])
def test_longest_admitted_progress_path_fits_production_limit(monkeypatch, clarify):
    expected_maximum = 1 + 6 * MAX_RESEARCH_ROUNDS + 4 + 2 * MAX_REPORT_REVISION_ROUNDS
    required_limit = 40 if ceil(expected_maximum * 1.2) <= 40 else ceil(expected_maximum / 5) * 5 + 5
    assert GRAPH_RECURSION_LIMIT >= required_limit, (
        f"最长路径 {expected_maximum} 步，当前上限 {GRAPH_RECURSION_LIMIT}，"
        f"按余量规则至少需要 {required_limit}；请重新核算 recursion_limit"
    )
    configure_graph(monkeypatch, tool_rounds=MAX_RESEARCH_ROUNDS, assess_each_batch=True,
                    revisions=MAX_REPORT_REVISION_ROUNDS, batch_size=2, clarify=clarify)
    graph = graph_module.build_research_graph(checkpointer=InMemorySaver())
    initial = create_initial_state("研究")
    config = {"recursion_limit": GRAPH_RECURSION_LIMIT, "configurable": {"thread_id": initial["run_id"]}}
    result = graph.invoke(initial, config)
    before = 0
    if clarify:
        assert result["__interrupt__"]
        before = len([e for e in result["trace"] if e["event"] == "node_completed"])
        result = graph.invoke(Command(resume="地点时间"), config)
    sequence = [e["node"] for e in result["trace"] if e["event"] == "node_completed"][before:]
    expected = ["request_clarification" if clarify else "recall_memory"]
    for _ in range(MAX_RESEARCH_ROUNDS):
        expected += ["agent", "tools", "record_results", "agent", "research_guard", "assess_evidence"]
    expected += ["generate_draft", "verify_report"]
    expected += [node for _ in range(MAX_REPORT_REVISION_ROUNDS) for node in ("revise_report", "verify_report")]
    expected += ["finalize_report", "store_memory"]
    assert sequence == expected, "图节点执行序列发生变化，请重新核算 recursion_limit"
    assert result["research_round"] == MAX_RESEARCH_ROUNDS
    assert result["tool_call_count"] == MAX_TOOL_CALLS
    assert result["report_revision_count"] == MAX_REPORT_REVISION_ROUNDS
    assert not any(e["event"] == "idle_round_counted" for e in result["trace"])
    # 每次空转消费一轮，不能把 4 个工具轮次和额外空转轮次同时叠加。
    combinations = [(tools, idle, 1 + 6 * tools + 3 * idle + 4 + 2 * MAX_REPORT_REVISION_ROUNDS)
                    for tools, idle in product(range(MAX_RESEARCH_ROUNDS + 1), repeat=2)
                    if tools + idle == MAX_RESEARCH_ROUNDS]
    maximum = max(total for _, _, total in combinations)
    assert maximum == len(sequence) == expected_maximum, "步数公式与实际图不一致，请重新核算 recursion_limit"
    label = "resume" if clarify else "without clarification"
    print(f"{label} step combinations (tools, idle, steps): {combinations}")
    # 有取证后才追问：旧证据可能允许恢复后的首次免费评估，但也已经消耗过轮次。
    prior_evidence_cases = [
        (spent, tools, idle, 1 + 3 + 6 * tools + 3 * idle + 4 + 2 * MAX_REPORT_REVISION_ROUNDS)
        for spent in range(1, MAX_RESEARCH_ROUNDS + 1)
        for tools, idle in product(range(MAX_RESEARCH_ROUNDS + 1), repeat=2)
        if spent + tools + idle == MAX_RESEARCH_ROUNDS
    ]
    prior_maximum = max(total for _, _, _, total in prior_evidence_cases)
    assert prior_maximum <= maximum, "已有证据恢复路径更长，请重新核算 recursion_limit"
    print(f"resume with existing evidence: at least one round already spent; max={prior_maximum}")
    print(f"{label}: actual={len(sequence)}, margin_target={required_limit}, production_limit={GRAPH_RECURSION_LIMIT}")
