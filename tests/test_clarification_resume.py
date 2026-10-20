"""通过真实 LangGraph/checkpointer 验证暂停恢复，不请求外部服务。"""

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import atmospheric_research_agent.agent.graph as graph_module
from atmospheric_research_agent.agent.nodes import (
    EvidenceAssessmentOutput, _task_question, assess_evidence_with_model,
)
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.run import run_research


@pytest.mark.parametrize("late", [False, True])
def test_pause_resume_preserves_task_and_counters(monkeypatch, late):
    calls = {"recall": 0, "draft": 0}

    def recall(state):
        calls["recall"] += 1
        return {"recalled_memories": [{"memory_id": "old"}]}

    def agent(state):
        update = {"messages": [AIMessage(content="{}") ]}
        if not late and not state["clarification_history"]:
            update["clarification_question"] = "哪一年？"
        return update

    class AssessmentModel:
        def invoke(self, messages):
            return EvidenceAssessmentOutput(
                sufficient=False, covered_points=[], missing_points=["年份"],
                potential_conflicts=[], next_step="clarify", reason="需确定事件",
                clarification_question="哪一年？",
            )

    def assess(state):
        if late and not state["clarification_history"]:
            return assess_evidence_with_model(state, AssessmentModel())
        assert "2021" in _task_question(state)
        return {"assessment": {"sufficient": True, "next_step": "full_report"}}

    def draft(state):
        calls["draft"] += 1
        return {"draft_report": "结论 [E1]"}

    monkeypatch.setattr(graph_module, "recall_memory_node", recall)
    monkeypatch.setattr(graph_module, "agent_node", agent)
    monkeypatch.setattr(graph_module, "assess_evidence_node", assess)
    monkeypatch.setattr(graph_module, "generate_draft_node", draft)
    monkeypatch.setattr(graph_module, "verify_report_node", lambda state: {"report_review": {"verdict": "pass"}})
    monkeypatch.setattr(graph_module, "store_memory_node", lambda state: {})
    graph = graph_module.build_research_graph(checkpointer=InMemorySaver())
    state = create_initial_state("分析暴雨")
    state.update(evidence=[{"evidence_id": "one", "content": "资料", "source_location": "test.md"}],
                 research_round=2, tool_call_count=3)
    config = {"configurable": {"thread_id": state["run_id"]}}
    paused = graph.invoke(state, config)
    assert paused["__interrupt__"][0].value["question"] == "哪一年？"
    assert graph.get_state(config).next == ("request_clarification",)
    assert calls == {"recall": 1, "draft": 0}
    assert paused["finished_at"] is None
    resumed = graph.invoke(Command(resume="2021 年"), config)
    assert resumed["run_id"] == state["run_id"]
    assert resumed["evidence"] == state["evidence"]
    assert resumed["research_round"] == 2
    assert resumed["tool_call_count"] == 3
    assert resumed["recalled_memories"] == [{"memory_id": "old"}]
    assert resumed["clarification_question"] is None
    assert len(resumed["clarification_history"]) == 1
    assert calls == {"recall": 1, "draft": 1}
    assert "2021" in resumed["messages"][-2].content
    assert resumed["final_report"]


def test_runner_reads_answer_and_does_not_restart(monkeypatch):
    calls = []
    def agent(state):
        calls.append(state["run_id"])
        if not state["clarification_history"]:
            return {"messages": [AIMessage(content="{}")], "clarification_question": "地点？"}
        return {"messages": [AIMessage(content="{}")]}
    monkeypatch.setattr(graph_module, "agent_node", agent)
    monkeypatch.setattr(graph_module, "recall_memory_node", lambda state: {})
    prompts = []
    def answer(prompt):
        prompts.append(prompt)
        return "北京"
    result = run_research("天气", answer_provider=answer)
    assert prompts == ["地点？"]
    assert len(set(calls)) == 1
    assert len(result["clarification_history"]) == 1
    assert result["trace"][-1]["event"] == "run_completed"


def test_assessment_cannot_request_clarification_without_question():
    with pytest.raises(ValueError, match="clarification_question"):
        EvidenceAssessmentOutput(sufficient=False, covered_points=[], missing_points=[],
                                 potential_conflicts=[], next_step="clarify", reason="缺地点")
