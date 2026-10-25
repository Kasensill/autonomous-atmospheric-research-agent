"""图级检查点隔离：模型固定，保留真实 interrupt/resume 与状态归并。"""

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from atmospheric_research_agent.agent import graph as graph_module
from atmospheric_research_agent.agent.state import create_initial_state


@pytest.fixture
def isolated_graph(monkeypatch):
    def agent(state):
        update = {"messages": [AIMessage(content='{"decision":"assess"}')]}
        if not state["clarification_history"]:
            update["clarification_question"] = f"{state['question']}：地点？"
        return update

    monkeypatch.setattr(graph_module, "agent_node", agent)
    monkeypatch.setattr(graph_module, "recall_memory_node", lambda state: {})
    monkeypatch.setattr(graph_module, "assess_evidence_node", lambda state: {"assessment": {"next_step": "full_report"}})
    monkeypatch.setattr(graph_module, "generate_draft_node", lambda state: {"draft_report": "本任务的独立证据 [E1]。", "report_mode": "full"})
    monkeypatch.setattr(graph_module, "verify_report_node", lambda state: {"report_review": {"verdict": "pass"}})
    monkeypatch.setattr(graph_module, "store_memory_node", lambda state: {})
    return graph_module.build_research_graph(checkpointer=InMemorySaver())


def test_interleaved_threads_preserve_answers_evidence_and_budgets(isolated_graph):
    states = [create_initial_state(name) for name in ("任务A", "任务B")]
    configs = [{"configurable": {"thread_id": state["run_id"]}} for state in states]
    for i, (state, config) in enumerate(zip(states, configs)):
        state.update(research_round=i, tool_call_count=i + 1, evidence=[{
            "evidence_id": f"evidence-{i}", "source_type": "local_knowledge",
            "content": f"仅属于任务{i}的资料", "source_location": f"thread-{i}.md",
        }])
        result = isolated_graph.invoke(state, config)
        assert result["__interrupt__"]
    result_b = isolated_graph.invoke(Command(resume="上海"), configs[1])
    still_a = isolated_graph.get_state(configs[0]).values
    assert still_a["clarification_history"] == []
    assert still_a["question"] == "任务A"
    assert still_a["tool_call_count"] == 1
    result_a = isolated_graph.invoke(Command(resume="北京"), configs[0])
    for result, answer, original in ((result_a, "北京", states[0]), (result_b, "上海", states[1])):
        assert result["clarification_history"] == [{"question": f"{original['question']}：地点？", "answer": answer}]
        assert result["run_id"] == original["run_id"]
        assert result["tool_call_count"] == original["tool_call_count"]
        assert result["evidence"] == original["evidence"]
        assert result["final_report"]


@pytest.mark.parametrize("answer", ["", "   ", None, 42])
def test_invalid_resume_does_not_spend_clarification(isolated_graph, answer):
    state = create_initial_state("任务A")
    config = {"configurable": {"thread_id": state["run_id"]}}
    isolated_graph.invoke(state, config)
    with pytest.raises(ValueError):
        isolated_graph.invoke(Command(resume=answer), config)
    saved = isolated_graph.get_state(config).values
    assert saved["clarification_history"] == []
    assert saved["tool_call_count"] == 0
    result = isolated_graph.invoke(Command(resume="北京"), config)
    assert len(result["clarification_history"]) == 1


def test_new_saver_cannot_restore_paused_task(isolated_graph):
    state = create_initial_state("任务A")
    config = {"configurable": {"thread_id": state["run_id"]}}
    isolated_graph.invoke(state, config)
    other = graph_module.build_research_graph(checkpointer=InMemorySaver())
    assert other.get_state(config).values == {}
    assert isolated_graph.get_state(config).next == ("request_clarification",)


def test_unknown_thread_does_not_expose_paused_state(isolated_graph):
    state = create_initial_state("任务A")
    config = {"configurable": {"thread_id": state["run_id"]}}
    isolated_graph.invoke(state, config)
    unknown = {"configurable": {"thread_id": "not-the-original-thread"}}
    assert isolated_graph.get_state(unknown).values == {}
    assert isolated_graph.get_state(config).values["question"] == "任务A"
