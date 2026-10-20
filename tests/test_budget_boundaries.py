"""先固定现有路由和观测包装器的边界行为，再抽取共用预算函数。"""

from itertools import product

import pytest
from langchain_core.messages import AIMessage

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.graph import _instrument_node
from atmospheric_research_agent.agent.state import create_initial_state


@pytest.mark.parametrize("rounds,count,batch", list(product((2, 3), (7, 8), (1, 2))))
def test_agent_budget_and_wrapper_preserve_batch_admission(rounds, count, batch):
    state = create_initial_state("研究")
    message = AIMessage(content="", tool_calls=[{
        "id": str(i), "name": "get_current_weather", "args": {"location": "北京"},
    } for i in range(batch)])
    state.update(research_round=rounds, tool_call_count=count, messages=[message])
    blocked = rounds >= 3 or count + batch > 8
    destination = "research_guard" if blocked else "tools"
    assert nodes.route_after_agent(state) == destination
    wrapped = _instrument_node("agent", lambda _: {})
    update = wrapped(state)
    event = next(e for e in update["trace"] if e["event"] == "route_decision")
    assert event["destination"] == destination
    expected = (["research_rounds_exhausted"] if rounds >= 3
                else ["tool_batch_exceeds_budget"]) if blocked else []
    assert update.get("program_limitations", []) == expected
    if blocked:
        assert ("研究轮次耗尽" if rounds >= 3 else "超过工具调用上限") in event["detail"]


@pytest.mark.parametrize("rounds,count", list(product((2, 3), (7, 8))))
@pytest.mark.parametrize("node_name", ("research_guard", "assess_evidence"))
def test_guard_and_assessment_budget_boundary_and_wrapper(rounds, count, node_name):
    state = create_initial_state("研究")
    state.update(research_round=rounds, tool_call_count=count, assessment={"next_step": "continue"})
    exhausted = rounds >= 3 or count >= 8
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
    state.update(research_round=3, tool_call_count=8, assessment={"next_step": next_step})
    assert nodes.route_after_evidence_assessment(state) == expected


def test_guard_usable_evidence_precedes_exhausted_budget():
    state = create_initial_state("研究")
    state.update(research_round=3, tool_call_count=8, evidence=[{"source_type": "local_knowledge"}])
    assert nodes.route_after_research_guard(state) == "assess_evidence"
