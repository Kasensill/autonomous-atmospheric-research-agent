import json

from langchain_core.messages import AIMessage

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.state import create_initial_state


def test_plan_round_is_recorded_by_program_without_changing_model_protocol(monkeypatch):
    class Model:
        def invoke(self, messages):
            return AIMessage(content=json.dumps({
                "decision": "assess", "clarification_question": None,
                "plan_update": {"goal": "研究下沉运动"},
            }, ensure_ascii=False))

    monkeypatch.setattr(nodes, "create_research_model", Model)
    state = create_initial_state("研究")
    state["research_round"] = 1
    state.update(nodes.agent_node(state))
    assert state["research_plan_round"] == 1
    state["research_round"] = 3
    content = nodes._assessment_messages(state)[1].content
    assert "当前研究计划产生于第 1 轮，之后的进展以证据和消息为准" in content
    assert "研究下沉运动" in content


def test_legacy_or_missing_plan_does_not_invent_origin_round():
    state = create_initial_state("研究")
    content = nodes._assessment_messages(state)[1].content
    assert "当前研究计划产生轮次未记录" in content
