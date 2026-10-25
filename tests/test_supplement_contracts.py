"""原始要求的缺口补测；保留真实工具执行、证据转换和LangGraph控制。"""

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from atmospheric_research_agent.agent import graph as graphs, nodes
from atmospheric_research_agent.agent.settings import GRAPH_RECURSION_LIMIT, MAX_CLARIFICATIONS
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.tools import RESEARCH_TOOLS
from atmospheric_research_agent.agent.tools import local_knowledge, source_reader, weather, web_search


def raise_timeout(*args, **kwargs):
    raise TimeoutError("补测超时")


def local_response(results):
    return {"results": results, "retrieval_mode": "hybrid_rrf_rerank", "warnings": []}


@pytest.mark.parametrize("tool_name", ["search_local_knowledge", "search_web", "read_source", "get_current_weather"])
@pytest.mark.parametrize("failure", ["timeout", "empty", "malformed"])
def test_four_tools_faults_do_not_create_formal_evidence(monkeypatch, tool_name, failure):
    """12组合：损坏正文/核心数值不能因JSON可解析就变成事实。"""
    if tool_name == "search_local_knowledge":
        value = local_response([]) if failure == "empty" else "损坏的召回结构"
        monkeypatch.setattr(local_knowledge, "_hybrid_local_retrieval", raise_timeout if failure == "timeout" else lambda *a, **k: value)
        args = {"query": "降水"}
    elif tool_name == "search_web":
        monkeypatch.setenv("TAVILY_API_KEY", "offline-test-placeholder")
        value = {"results": []} if failure == "empty" else {"results": "损坏的候选列表"}
        monkeypatch.setattr(web_search, "_tavily_search", raise_timeout if failure == "timeout" else lambda *a: value)
        args = {"query": "降水"}
    elif tool_name == "read_source":
        fetch = raise_timeout if failure == "timeout" else lambda *a, **k: SimpleNamespace(data=b"<html>test</html>", status=200, headers={"content-type": "text/html"})
        fake = SimpleNamespace(fetch_response=fetch,
                               extract=lambda *a, **k: "" if failure == "empty" else "\ufffd" * 100,
                               extract_metadata=lambda *a, **k: SimpleNamespace(title="损坏正文测试", author=None, sitename="example.org", hostname="example.org", date=None))
        monkeypatch.setitem(sys.modules, "trafilatura", fake)
        args = {"url": "https://example.org/fault"}
    else:
        now = datetime.now(UTC).isoformat()
        current = {"time": now, "temperature_2m": "乱码不是气温", "relative_humidity_2m": -99,
                   "wind_speed_10m": "乱码不是风速", "precipitation": None}
        responses = [({"results": [{"name": "北京", "latitude": 39.9, "longitude": 116.4, "timezone": "Asia/Shanghai"}]}, "https://example.org/geo"),
                     ({"current": None if failure == "empty" else current, "timezone": "Asia/Shanghai"}, "https://example.org/weather")]
        monkeypatch.setattr(weather, "_get_json", raise_timeout if failure == "timeout" else lambda *a: responses.pop(0))
        args = {"location": "北京"}
    messages = nodes.execute_tool_calls([{"id": "fault", "name": tool_name, "args": args}], RESEARCH_TOOLS)
    assert len(messages) == 1 and messages[0].tool_call_id == "fault"
    state = create_initial_state("故障资料不能当证据")
    state["messages"] = messages
    update = nodes.record_results_node(state)
    assert update["tool_call_count"] == 1
    assert update["evidence"] == [], f"{tool_name}/{failure}把故障结果记成正式证据"


def test_mixed_research_batch_preserves_success_and_records_failure(monkeypatch):
    item = {"content": "下沉常抑制云雨。", "source": {"location": "test.md", "display_name": "固定资料"}, "retrieval": {"rerank_score": 0.9}}
    monkeypatch.setattr(local_knowledge, "_hybrid_local_retrieval", lambda *a, **k: local_response([item]))
    monkeypatch.setattr(weather, "_get_json", raise_timeout)
    calls = [{"id": "local-ok", "name": "search_local_knowledge", "args": {"query": "高压"}},
             {"id": "weather-fail", "name": "get_current_weather", "args": {"location": "北京"}}]
    messages = nodes.execute_tool_calls(calls, RESEARCH_TOOLS)
    state = create_initial_state("多工具一成一败")
    state["messages"] = messages
    update = nodes.record_results_node(state)
    assert [m.tool_call_id for m in messages] == ["local-ok", "weather-fail"]
    assert update["tool_call_count"] == 2 and update["research_round"] == 1
    assert len(update["evidence"]) == 1
    assert update["evidence"][0]["tool_name"] == "search_local_knowledge"
    assert any(e["event"] == "tool_failed" for e in update["trace"])


@pytest.mark.parametrize("question", ["", " ", "\n\t"])
def test_empty_research_question_is_rejected_before_any_model_call(question):
    with pytest.raises(ValueError, match="不能为空"):
        create_initial_state(question)


def install_fixed_report(monkeypatch):
    monkeypatch.setattr(graphs, "recall_memory_node", lambda state: {})
    monkeypatch.setattr(graphs, "generate_draft_node", lambda state: {"draft_report": "已取得的机制 [E1]。", "report_mode": "limited"})
    monkeypatch.setattr(graphs, "verify_report_node", lambda state: {"report_review": {"verdict": "pass"}})
    monkeypatch.setattr(graphs, "store_memory_node", lambda state: {})


@pytest.mark.parametrize("order", ["assessment-only", "assessment-then-agent"])
def test_shared_clarification_quota_with_actual_two_resumes(monkeypatch, order):
    install_fixed_report(monkeypatch)

    def agent(state):
        if order == "assessment-then-agent" and len(state["clarification_history"]) == 1:
            return {"messages": [AIMessage(content="", tool_calls=[{"id": "agent-ask", "name": "ask_user", "args": {"question": "第二个用户条件？"}}])],
                    "clarification_question": "第二个用户条件？"}
        return {"messages": [AIMessage(content='{"decision":"assess"}')]}

    class Assess:
        def invoke(self, messages):
            return nodes.EvidenceAssessmentOutput(sufficient=False, covered_points=["机制"], missing_points=["条件"],
                                                  potential_conflicts=[], next_step="clarify", reason="需要用户条件", clarification_question="用户条件？")

    monkeypatch.setattr(graphs, "agent_node", agent)
    monkeypatch.setattr(graphs, "assess_evidence_node", lambda state: nodes.assess_evidence_with_model(state, Assess()))
    state = create_initial_state("共享澄清两入口")
    state["evidence"] = [{"evidence_id": "one", "source_type": "local_knowledge", "content": "机制", "source_location": "fixed.md"}]
    graph = graphs.build_research_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": state["run_id"]}}
    first = graph.invoke(state, config)
    assert first["__interrupt__"]
    second = graph.invoke(Command(resume="第一条件"), config)
    assert second["__interrupt__"]
    result = graph.invoke(Command(resume="第二条件"), config)
    assert len(result["clarification_history"]) == MAX_CLARIFICATIONS
    assert not result.get("__interrupt__") and result["final_report"]
    assert result["research_round"] == result["tool_call_count"] == 0


def test_duplicate_resume_after_completion_does_not_repeat_work(monkeypatch):
    install_fixed_report(monkeypatch)
    calls = []

    def agent(state):
        calls.append(state["run_id"])
        if not state["clarification_history"]:
            return {"messages": [AIMessage(content='{"decision":"assess"}')], "clarification_question": "地点？"}
        return {"messages": [AIMessage(content='{"decision":"assess"}')]}

    monkeypatch.setattr(graphs, "agent_node", agent)
    graph = graphs.build_research_graph(checkpointer=InMemorySaver())
    state = create_initial_state("重复恢复")
    config = {"configurable": {"thread_id": state["run_id"]}}
    graph.invoke(state, config)
    result = graph.invoke(Command(resume="北京"), config)
    before = graph.get_state(config).values
    before_calls = len(calls)
    try:
        graph.invoke(Command(resume="北京"), config)
    except (ValueError, RuntimeError):
        pass  # 明确拒绝或幂等返回均可，不能再次研究或写记忆。
    after = graph.get_state(config).values
    assert len(calls) == before_calls
    assert before == after
    assert result["final_report"]


@pytest.mark.parametrize("steps", [GRAPH_RECURSION_LIMIT - 1, GRAPH_RECURSION_LIMIT, GRAPH_RECURSION_LIMIT + 1])
def test_production_recursion_limit_exact_framework_boundary(steps):
    """单独验证生产40值的框架语义；不冒充正常业务图会运行40步。"""
    from typing_extensions import TypedDict

    class Counter(TypedDict):
        count: int

    graph = StateGraph(Counter)
    graph.add_node("step", lambda state: {"count": state["count"] + 1})
    graph.add_edge(START, "step")
    graph.add_conditional_edges("step", lambda state: END if state["count"] >= steps else "step")
    compiled = graph.compile(checkpointer=InMemorySaver())
    config = {"recursion_limit": GRAPH_RECURSION_LIMIT, "configurable": {"thread_id": f"boundary-{steps}"}}
    if steps < GRAPH_RECURSION_LIMIT:
        assert compiled.invoke({"count": 0}, config)["count"] == steps
    else:
        with pytest.raises(GraphRecursionError):
            compiled.invoke({"count": 0}, config)
        # 当前LangGraph在40轮处进入out_of_steps；终止检查同样需要框架推进。
        # 这里固定实际语义，初次错误地假设40节点正常返回的记录另行保留。
        assert compiled.get_state(config).values["count"] == GRAPH_RECURSION_LIMIT
