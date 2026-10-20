import json

import pytest
from langchain_core.messages import AIMessage

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.graph import build_research_graph
from atmospheric_research_agent.agent.memory import save_project_memory, recall_project_memory
from atmospheric_research_agent.agent.state import create_initial_state


class DraftModel:
    def invoke(self, messages):
        return AIMessage(content="下沉运动抑制云雨。[E1]")


def state_with_evidence():
    state = create_initial_state("副热带高压的下沉运动")
    state["evidence"] = [{"evidence_id": "1", "content": "下沉运动抑制云雨。",
                          "source_type": "local_knowledge", "source_location": "theory.md"}]
    return state


@pytest.mark.parametrize("sufficient,next_step,mode", [
    (True, "full_report", "full"), (False, "full_report", "limited"),
    (True, "limited_report", "limited"), (False, "continue", "limited"),
])
def test_draft_sets_mode_without_program_notice(sufficient, next_step, mode):
    state = state_with_evidence()
    state["assessment"] = {"sufficient": sufficient, "next_step": next_step}
    update = nodes.generate_draft_with_model(state, DraftModel())
    assert update["report_mode"] == mode
    assert update["draft_report"] == "下沉运动抑制云雨。[E1]"


@pytest.mark.parametrize("evidence", [
    [],
    [{"source_type": "weather", "valid_at": "2000-01-01T00:00:00+00:00",
      "content": "过期天气快照"}],
], ids=["empty-evidence", "all-expired-weather"])
def test_draft_rejects_unusable_evidence_without_calling_model(evidence):
    class UnexpectedModel:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise AssertionError("没有可用证据时不能调用生成模型")

    state = create_initial_state("没有可用证据的问题")
    state["evidence"] = evidence
    model = UnexpectedModel()
    with pytest.raises(ValueError, match="generate_draft 需要正式证据"):
        nodes.generate_draft_with_model(state, model)
    assert model.calls == 0


@pytest.mark.parametrize("mode", ("full", "limited"))
def test_verified_report_saves_mode_and_limited_notice_only_at_finalize(monkeypatch, tmp_path, mode):
    state = state_with_evidence()
    state.update(assessment={"sufficient": mode == "full", "next_step": f"{mode}_report"})
    state.update(nodes.generate_draft_with_model(state, DraftModel()))
    state["report_review"] = {"verdict": "pass"}
    original = state["draft_report"]
    state.update(nodes.finalize_report_node(state))
    assert ("受限报告：当前证据不全" in state["final_report"]) == (mode == "limited")
    assert state["draft_report"] == original
    if mode == "full":
        assert state["final_report"] == original + "\n\n" + nodes._reference_section(nodes._numbered_evidence(state), report=original)
    path = tmp_path / "memory.json"
    monkeypatch.setattr(nodes, "save_project_memory", lambda record: save_project_memory(record, store_path=path))
    assert nodes.store_memory_node(state)["trace"][0]["event"] == "memory_stored"
    record = json.loads(path.read_text(encoding="utf-8"))[0]
    assert record["status"] == "verified_report"
    assert record["report_mode"] == mode
    state["recalled_memories"] = recall_project_memory(state["question"], store_path=path)
    assert state["recalled_memories"]
    before = json.dumps(state["recalled_memories"], ensure_ascii=False)
    context = "\n".join(message.content for message in nodes._messages_for_model(state))
    assert ("证据不全，仅作线索，不可当作结论" in context) == (mode == "limited")
    assert json.dumps(state["recalled_memories"], ensure_ascii=False) == before


def test_failure_explanation_never_reaches_memory(monkeypatch):
    import atmospheric_research_agent.agent.graph as graph_module

    def unexpected(_):
        pytest.fail("失败说明不能进入记忆存储节点")

    monkeypatch.setattr(graph_module, "recall_memory_node", lambda _: {})
    monkeypatch.setattr(graph_module, "agent_node", lambda _: {"messages": [AIMessage(content="")]})
    monkeypatch.setattr(graph_module, "store_memory_node", unexpected)
    result = build_research_graph().invoke(create_initial_state("研究"))
    assert "未取得可引用的正式证据" in result["final_report"]
    assert not any(e["node"] == "store_memory" for e in result["trace"])
