"""先固定现有路由和观测包装器的边界行为，再抽取共用预算函数。"""

from itertools import product

import pytest
from langchain_core.messages import AIMessage

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.graph import _instrument_node
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.settings import (
    MAX_RESEARCH_ROUNDS, MAX_TOOL_CALLS, research_rounds_exhausted,
)


@pytest.mark.parametrize("rounds,count,batch", list(product(
    (MAX_RESEARCH_ROUNDS - 1, MAX_RESEARCH_ROUNDS), (MAX_TOOL_CALLS - 1, MAX_TOOL_CALLS), (1, 2),
)))
def test_agent_budget_and_wrapper_preserve_batch_admission(rounds, count, batch):
    state = create_initial_state("研究")
    message = AIMessage(content="", tool_calls=[{
        "id": str(i), "name": "get_current_weather", "args": {"location": "北京"},
    } for i in range(batch)])
    state.update(research_round=rounds, tool_call_count=count, messages=[message])
    blocked = rounds >= MAX_RESEARCH_ROUNDS or count + batch > MAX_TOOL_CALLS
    destination = "research_guard" if blocked else "tools"
    assert nodes.route_after_agent(state) == destination
    wrapped = _instrument_node("agent", lambda _: {})
    update = wrapped(state)
    event = next(e for e in update["trace"] if e["event"] == "route_decision")
    assert event["destination"] == destination
    expected = (["research_rounds_exhausted"] if rounds >= MAX_RESEARCH_ROUNDS
                else ["tool_batch_exceeds_budget"]) if blocked else []
    assert update.get("program_limitations", []) == expected
    if blocked:
        assert ("研究轮次耗尽" if rounds >= MAX_RESEARCH_ROUNDS else "超过工具调用上限") in event["detail"]


@pytest.mark.parametrize("rounds,count", list(product(
    (MAX_RESEARCH_ROUNDS - 1, MAX_RESEARCH_ROUNDS), (MAX_TOOL_CALLS - 1, MAX_TOOL_CALLS),
)))
@pytest.mark.parametrize("node_name", ("research_guard", "assess_evidence"))
def test_guard_and_assessment_budget_boundary_and_wrapper(rounds, count, node_name):
    state = create_initial_state("研究")
    state.update(research_round=rounds, tool_call_count=count, assessment={"next_step": "continue"})
    exhausted = rounds >= MAX_RESEARCH_ROUNDS or count >= MAX_TOOL_CALLS
    route = (nodes.route_after_research_guard if node_name == "research_guard"
             else nodes.route_after_evidence_assessment)
    expected = ("finalize_limited_report" if node_name == "research_guard"
                else "generate_draft") if exhausted else "agent"
    assert route(state) == expected
    update = _instrument_node(node_name, lambda _: {})(state)
    assert update.get("program_limitations", []) == (["budget_exhausted"] if exhausted else [])


@pytest.mark.parametrize("next_step,expected", [
    ("clarify", "request_clarification"), ("full_report", "generate_draft"),
    ("limited_report", "generate_draft"),
])
def test_assessment_decision_precedes_budget(next_step, expected):
    state = create_initial_state("研究")
    state.update(research_round=MAX_RESEARCH_ROUNDS, tool_call_count=MAX_TOOL_CALLS, assessment={"next_step": next_step})
    assert nodes.route_after_evidence_assessment(state) == expected


def test_guard_usable_evidence_precedes_exhausted_budget():
    state = create_initial_state("研究")
    state.update(research_round=MAX_RESEARCH_ROUNDS, tool_call_count=MAX_TOOL_CALLS, evidence=[{"source_type": "local_knowledge"}])
    assert nodes.route_after_research_guard(state) == "assess_evidence"


def test_research_budget_defaults_are_four_rounds_and_eight_tools():
    # 固定本次配置要求；其余边界用例均从常量计算，不再绑定旧上限。
    assert MAX_RESEARCH_ROUNDS == 4
    assert MAX_TOOL_CALLS == 8


@pytest.mark.parametrize("rounds,exhausted,expected", [
    (MAX_RESEARCH_ROUNDS - 1, False, "tools"),
    (MAX_RESEARCH_ROUNDS, True, "research_guard"),
])
def test_last_available_round_admits_tools_but_limit_blocks_them(rounds, exhausted, expected):
    state = create_initial_state("研究预算边界")
    state.update(research_round=rounds, tool_call_count=0, messages=[AIMessage(
        content="", tool_calls=[{"id": "boundary", "name": "search_local_knowledge", "args": {"query": "测试"}}],
    )])
    assert research_rounds_exhausted(state) is exhausted
    assert nodes.route_after_agent(state) == expected
