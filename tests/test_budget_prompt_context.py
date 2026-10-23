from copy import deepcopy

import pytest

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.settings import MAX_RESEARCH_ROUNDS, MAX_TOOL_CALLS
from atmospheric_research_agent.agent.state import create_initial_state


@pytest.mark.parametrize("rounds,calls,exhausted", [
    (0, 0, False),
    (MAX_RESEARCH_ROUNDS - 1, MAX_TOOL_CALLS - 1, False),
    (MAX_RESEARCH_ROUNDS, 0, True),
    (0, MAX_TOOL_CALLS, True),
    (MAX_RESEARCH_ROUNDS + 1, MAX_TOOL_CALLS + 1, True),
])
def test_assessment_receives_program_budget_status_not_raw_counts(rounds, calls, exhausted):
    state = create_initial_state("研究")
    state.update(research_round=rounds, tool_call_count=calls)
    before = deepcopy(state)
    messages = nodes._assessment_messages(state)
    content = "\n".join(message.content for message in messages)
    sentence = ("研究预算：已耗尽，不得选择 continue。" if exhausted
                else "研究预算：可以继续检索。")
    assert messages[1].content.endswith(sentence)
    assert "当前研究轮次" not in content
    assert "当前工具调用数" not in content
    assert "reason 里不得描述预算余量或剩余额度" in content
    assert state == before


@pytest.mark.parametrize("exhausted", [False, True])
def test_assessment_reuses_budget_helper(monkeypatch, exhausted):
    state = create_initial_state("研究")
    seen = []

    def budget_check(received):
        seen.append(received)
        return exhausted

    monkeypatch.setattr(nodes, "research_budget_exhausted", budget_check)
    content = nodes._assessment_messages(state)[1].content
    assert seen == [state]
    assert ("研究预算：已耗尽，不得选择 continue。" if exhausted
            else "研究预算：可以继续检索。") in content


@pytest.mark.parametrize("rounds,calls", [
    (0, 0),
    (MAX_RESEARCH_ROUNDS - 1, MAX_TOOL_CALLS - 2),
    (MAX_RESEARCH_ROUNDS, 0),
    (0, MAX_TOOL_CALLS),
    (MAX_RESEARCH_ROUNDS + 1, MAX_TOOL_CALLS + 1),
])
def test_agent_receives_remaining_budget_and_batch_rule(rounds, calls):
    state = create_initial_state("研究")
    state.update(research_round=rounds, tool_call_count=calls)
    before = deepcopy(state)
    content = nodes._messages_for_model(state)[1].content
    remaining_rounds = max(0, MAX_RESEARCH_ROUNDS - rounds)
    remaining_tools = max(0, MAX_TOOL_CALLS - calls)
    assert f"剩余研究轮次 {remaining_rounds}，剩余工具调用 {remaining_tools}。" in content
    assert f"一批请求的工具调用总数不得超过 {remaining_tools}，超过时整批不会执行" in content
    assert "剩余研究轮次或剩余工具调用为 0 时不要再发起研究工具调用。" in content
    assert '"research_round"' not in content
    assert '"tool_call_count"' not in content
    assert state == before


def test_agent_remaining_budget_uses_constants(monkeypatch):
    monkeypatch.setattr(nodes, "MAX_RESEARCH_ROUNDS", 11)
    monkeypatch.setattr(nodes, "MAX_TOOL_CALLS", 19)
    state = create_initial_state("研究")
    state.update(research_round=2, tool_call_count=3)
    assert "剩余研究轮次 9，剩余工具调用 16。" in nodes._messages_for_model(state)[1].content


def test_brevity_rule_applies_only_to_limited_report():
    state = create_initial_state("研究")
    rule = (
        "受限报告要简洁：每个证据缺口只写一次，不要在结论、解释、局限三处重复同一条'无法判断'；"
        "证据已有的内容写清楚，没有的内容集中放在局限性一节。"
    )
    assert rule in nodes._report_messages(state, "limited")[1].content
    assert rule not in nodes._report_messages(state, "full")[1].content
