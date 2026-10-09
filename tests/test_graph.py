import json

from langchain_core.messages import AIMessage, ToolMessage

import atmospheric_research_agent.agent.graph as graph_module
from atmospheric_research_agent.agent.graph import build_research_graph
from atmospheric_research_agent.agent.state import create_initial_state


def test_research_graph_compiles_with_all_expected_nodes() -> None:
    graph = build_research_graph()
    node_names = set(graph.get_graph().nodes)

    assert {
        "__start__",
        "__end__",
        "recall_memory",
        "agent",
        "tools",
        "record_results",
        "research_guard",
        "assess_evidence",
        "generate_draft",
        "verify_report",
        "revise_report",
        "finalize_report",
        "finalize_limited_report",
        "request_clarification",
        "store_memory",
    }.issubset(node_names)


def test_research_graph_runs_tool_evidence_assessment_and_report_path(monkeypatch) -> None:
    """不请求外部模型，验证实际图能沿主路径走到报告 END。"""

    def fake_agent(state):
        if state.get("evidence"):
            return {"messages": [AIMessage(content="{")]}
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_local_knowledge",
                            "args": {"query": "测试"},
                            "id": "call_test_1",
                        }
                    ],
                )
            ]
        }

    def fake_tools(state):
        return {
            "messages": [
                ToolMessage(
                    name="search_local_knowledge",
                    tool_call_id="call_test_1",
                    content=json.dumps(
                        {
                            "status": "success",
                            "results": [
                                {
                                    "content": "测试证据。",
                                    "source": {
                                        "display_name": "知识库：测试来源",
                                        "location": "test.md",
                                    },
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                )
            ]
        }

    def fake_assessment(state):
        return {
            "assessment": {
                "sufficient": True,
                "covered_points": ["测试要点"],
                "missing_points": [],
                "potential_conflicts": [],
                "next_step": "full_report",
                "reason": "测试证据充分。",
            }
        }

    def fake_draft(state):
        return {"draft_report": "# 测试报告草稿\n\n引用 [E1]。"}

    def fake_verify(state):
        return {"report_review": {"verdict": "pass", "reason": "测试通过。"}}

    def fake_finalize(state):
        return {"final_report": "# 测试报告\n\n引用 [E1]。"}

    def fake_recall_memory(state):
        return {"recalled_memories": []}

    def fake_store_memory(state):
        return {"trace": [{"node": "store_memory", "event": "memory_stored", "detail": "测试保存。"}]}

    monkeypatch.setattr(graph_module, "recall_memory_node", fake_recall_memory)
    monkeypatch.setattr(graph_module, "agent_node", fake_agent)
    monkeypatch.setattr(graph_module, "tools_node", fake_tools)
    monkeypatch.setattr(graph_module, "assess_evidence_node", fake_assessment)
    monkeypatch.setattr(graph_module, "generate_draft_node", fake_draft)
    monkeypatch.setattr(graph_module, "verify_report_node", fake_verify)
    monkeypatch.setattr(graph_module, "finalize_report_node", fake_finalize)
    monkeypatch.setattr(graph_module, "store_memory_node", fake_store_memory)

    result = graph_module.build_research_graph().invoke(create_initial_state("测试问题"))

    assert result["final_report"] == "# 测试报告\n\n引用 [E1]。"
    assert result["tool_call_count"] == 1
    assert result["research_round"] == 1
    assert len(result["evidence"]) == 1
    assert result["assessment"]["sufficient"] is True
    node_events = [(entry["node"], entry["event"]) for entry in result["trace"]]
    assert ("agent", "node_started") in node_events
    assert ("agent", "node_completed") in node_events
    assert ("verify_report", "node_completed") in node_events
    assert ("store_memory", "node_completed") in node_events


def test_research_graph_ends_safely_when_no_evidence_is_found(monkeypatch) -> None:
    """零证据达到预算后应交付说明原因的受限结论，而不是在最终节点抛错。"""

    def fake_agent(state):
        return {"messages": [AIMessage(content="{}")]} 

    def fake_recall_memory(state):
        return {"recalled_memories": []}

    monkeypatch.setattr(graph_module, "recall_memory_node", fake_recall_memory)
    monkeypatch.setattr(graph_module, "agent_node", fake_agent)

    result = graph_module.build_research_graph().invoke(create_initial_state("没有证据的测试"))

    assert "研究受限结论" in result["final_report"]
    assert "未取得可引用的正式证据" in result["final_report"]
    assert any(
        entry["event"] == "limited_report_without_evidence" for entry in result["trace"]
    )
