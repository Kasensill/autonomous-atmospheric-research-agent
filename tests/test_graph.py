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
        "agent",
        "tools",
        "record_results",
        "research_guard",
        "assess_evidence",
        "generate_report",
        "request_clarification",
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

    def fake_report(state):
        return {"final_report": "# 测试报告\n\n引用 [E1]。"}

    monkeypatch.setattr(graph_module, "agent_node", fake_agent)
    monkeypatch.setattr(graph_module, "tools_node", fake_tools)
    monkeypatch.setattr(graph_module, "assess_evidence_node", fake_assessment)
    monkeypatch.setattr(graph_module, "generate_report_node", fake_report)

    result = graph_module.build_research_graph().invoke(create_initial_state("测试问题"))

    assert result["final_report"] == "# 测试报告\n\n引用 [E1]。"
    assert result["tool_call_count"] == 1
    assert result["research_round"] == 1
    assert len(result["evidence"]) == 1
    assert result["assessment"]["sufficient"] is True
