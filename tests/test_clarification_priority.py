"""专用澄清控制声明、消息配对、共享上限与真实图暂停恢复的离线回归。"""

import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from atmospheric_research_agent.agent import nodes
import atmospheric_research_agent.agent.graph as graph_module
from atmospheric_research_agent.agent.settings import MAX_CLARIFICATIONS, MAX_RESEARCH_ROUNDS
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.tools.clarification import ASK_USER_TOOL


def ask(identifier="ask", question="哪个城市？"):
    return {"id": identifier, "name": "ask_user", "args": {"question": question}}


def search(identifier="search"):
    return {"id": identifier, "name": "search_local_knowledge", "args": {"query": "下沉运动"}}


def history(count=MAX_CLARIFICATIONS):
    return [{"question": f"问题{i}", "answer": f"回答{i}"} for i in range(count)]


def evidence():
    return [{"evidence_id": "old", "content": "下沉运动抑制云雨。",
             "source_type": "local_knowledge", "source_location": "old.md"}]


def assert_protocol_closed(messages):
    pending = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, "工具响应缺少请求或重复响应"
            pending.remove(message.tool_call_id)
        else:
            assert not pending, "工具请求未全部回应就出现了非工具消息"
            if isinstance(message, AIMessage):
                pending.update(call["id"] for call in message.tool_calls)
    assert not pending, "存在悬空工具请求"


class SequenceModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def invoke(self, messages):
        assert_protocol_closed(messages)
        self.calls += 1
        return next(self.responses)


def apply_update(state, update):
    for key, value in update.items():
        if key in {"messages", "trace", "clarification_history", "evidence"}:
            state[key] = [*state.get(key, []), *value]
        else:
            state[key] = value


def test_ask_user_is_bound_but_not_registered_for_execution(monkeypatch):
    class BindingModel:
        def bind_tools(self, tools):
            assert ASK_USER_TOOL in tools
            assert len(tools) == 5
            return self

    model = BindingModel()
    monkeypatch.setattr(nodes, "_create_base_model", lambda: model)
    assert nodes.create_research_model() is model
    assert "ask_user" not in {tool.name for tool in nodes.RESEARCH_TOOLS}


def test_multiple_asks_cancel_research_and_have_closed_protocol_before_pause(monkeypatch):
    state = create_initial_state("天气")
    state.update(research_round=MAX_RESEARCH_ROUNDS, tool_call_count=8)
    model = SequenceModel([AIMessage(content="", tool_calls=[ask(), search(), ask("ask2", "哪一年？")])])
    monkeypatch.setattr(nodes, "create_research_model", lambda: model)
    update = nodes.agent_node(state)
    assert "research_round" not in update and "tool_call_count" not in update and "evidence" not in update
    apply_update(state, update)
    assert state["clarification_question"] == "哪个城市？\n哪一年？"
    assert nodes.route_after_agent(state) == "request_clarification"  # 优先于已耗尽的预算
    assert_protocol_closed(state["messages"])
    responses = [m for m in update["messages"] if isinstance(m, ToolMessage)]
    assert len(responses) == 3
    assert "等待用户，非回答" in responses[0].content
    assert "本轮因澄清优先取消，未执行" in responses[1].content
    # 即便未来误把这些控制回应交给整理层，仍不允许计数或形成证据。
    ignored = nodes.record_results_node(state)
    assert "tool_call_count" not in ignored and "research_round" not in ignored and "evidence" not in ignored


@pytest.mark.parametrize("question", ["", "   ", None, 123])
@pytest.mark.parametrize("with_research", [False, True])
@pytest.mark.parametrize("budget_available", [False, True])
def test_invalid_ask_preserves_research_and_never_pauses(monkeypatch, question, with_research, budget_available):
    state = create_initial_state("研究")
    state["research_round"] = 0 if budget_available else MAX_RESEARCH_ROUNDS
    calls = [ask(question=question), *([search()] if with_research else [])]
    model = SequenceModel([AIMessage(content="", tool_calls=calls)])
    monkeypatch.setattr(nodes, "create_research_model", lambda: model)
    update = nodes.agent_node(state)
    apply_update(state, update)
    assert state["clarification_question"] is None
    assert nodes.route_after_agent(state) == ("tools" if with_research and budget_available else "research_guard")
    if with_research and budget_available:
        assert len([m for m in update["messages"] if isinstance(m, ToolMessage)]) == 1
        # 使用空注册表使普通工具得到安全错误回应，不调用真实服务。
        monkeypatch.setattr(nodes, "RESEARCH_TOOLS", [])
        apply_update(state, nodes.tools_node(state))
    assert_protocol_closed(state["messages"])
    assert any(e["event"] == "invalid_clarification_request" for e in update["trace"])


@pytest.mark.parametrize("situation", ["valid", "exhausted", "invalid-empty", "invalid-type"])
def test_final_graph_messages_have_exactly_one_response_per_call_id(monkeypatch, situation):
    question = "哪个城市？" if situation in {"valid", "exhausted"} else "" if situation == "invalid-empty" else 123
    response = AIMessage(content="", tool_calls=[ask("ask-exact", question), search("research-1"), search("research-2")])
    model = SequenceModel([response, AIMessage(content='{"decision":"assess","plan_update":{}}')])
    executed = []

    @tool
    def search_local_knowledge(query: str) -> dict:
        """只记录执行，不联网。"""
        executed.append(query)
        return {"status": "success", "results": []}

    monkeypatch.setattr(nodes, "RESEARCH_TOOLS", [search_local_knowledge])
    graph = setup_real_graph(monkeypatch, model)
    state = create_initial_state("研究")
    state["evidence"] = evidence()
    if situation == "exhausted":
        state["clarification_history"] = history()
    config = {"configurable": {"thread_id": state["run_id"]}}
    result = graph.invoke(state, config)
    if situation == "valid":
        assert result["__interrupt__"]
        result = graph.invoke(Command(resume="北京"), config)
    else:
        assert not result.get("__interrupt__")
    assert result["final_report"]
    assert_protocol_closed(result["messages"])
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                matches = [m for m in result["messages"] if isinstance(m, ToolMessage)
                           and m.tool_call_id == call["id"]]
                assert len(matches) == 1, f"{call['id']} 必须恰好得到一条回应"
    assert len(executed) == (0 if situation == "valid" else 2)
    assert result["tool_call_count"] == len(executed)


def test_tools_safely_refuses_control_call_even_if_registry_contains_it(monkeypatch):
    @tool
    def ask_user(question: str) -> dict:
        """不应被执行的陷阱工具。"""
        pytest.fail("控制声明不能实际执行")

    message = nodes.execute_tool_calls([ask()], [ask_user])[0]
    assert message.status == "error"
    state = create_initial_state("研究")
    state["messages"] = [AIMessage(content="", tool_calls=[ask()])]
    update = nodes.tools_node(state)
    assert update["trace"][0]["event"] == "control_tool_not_executable"
    apply_update(state, update)
    apply_update(state, nodes.record_results_node(state))
    assert state["tool_call_count"] == 0 and state["research_round"] == 0
    assert state["evidence"] == []
    assert_protocol_closed(state["messages"])


@pytest.mark.parametrize("batch,expected", [(1, "tools"), (2, "research_guard")])
def test_limit_rejects_only_ask_and_uses_normal_batch_budget(monkeypatch, batch, expected):
    state = create_initial_state("研究")
    state.update(clarification_history=history(), research_round=MAX_RESEARCH_ROUNDS - 1, tool_call_count=7)
    response = AIMessage(content="", tool_calls=[ask(), *[search(str(i)) for i in range(batch)]])
    monkeypatch.setattr(nodes, "create_research_model", lambda: SequenceModel([response]))
    update = graph_module._instrument_node("agent", nodes.agent_node)(state)
    apply_update(state, update)
    assert nodes.route_after_agent(state) == expected
    assert state["clarification_question"] is None
    assert "clarification_limit_reached" in state["program_limitations"]
    rejected = [m for m in update["messages"] if isinstance(m, ToolMessage) and m.name == "ask_user"]
    assert len(rejected) == 1 and "已达上限，不可再追问" in rejected[0].content
    if batch == 1:
        assert len([m for m in update["messages"] if isinstance(m, ToolMessage)]) == 1
    else:
        assert "tool_batch_exceeds_budget" in state["program_limitations"]
        assert_protocol_closed(state["messages"])
    assert state["tool_call_count"] == 7


@pytest.mark.parametrize("rounds,count", [(0, 0), (MAX_RESEARCH_ROUNDS, 8)])
def test_limit_without_evidence_fails_without_spending_rounds(monkeypatch, rounds, count):
    state = create_initial_state("研究")
    state.update(clarification_history=history(), research_round=rounds, tool_call_count=count)
    model = SequenceModel([AIMessage(content="", tool_calls=[ask()])])
    monkeypatch.setattr(nodes, "create_research_model", lambda: model)
    apply_update(state, nodes.agent_node(state))
    apply_update(state, graph_module._instrument_node("research_guard", nodes.research_guard_node)(state))
    assert nodes.route_after_research_guard(state) == "finalize_limited_report"
    result = nodes.finalize_limited_report_node(state)
    assert "澄清次数达到上限" in result["final_report"]
    assert "预算耗尽" not in result["final_report"]
    assert "budget_exhausted" not in state["program_limitations"]
    assert state["research_round"] == rounds and state["tool_call_count"] == count
    assert_protocol_closed(state["messages"])


class ClarifyingAssessment:
    def invoke(self, messages):
        assert "当前不可再追问用户。" in messages[0].content
        return nodes.EvidenceAssessmentOutput(
            sufficient=False, covered_points=[], missing_points=["年份"], potential_conflicts=[],
            next_step="clarify", reason="缺年份", clarification_question="哪一年？",
        )


def test_assessment_shares_limit_and_converts_clarify_to_limited_report():
    state = create_initial_state("研究")
    state.update(clarification_history=history(), evidence=evidence())
    assert "不可再追问" in nodes._messages_for_model(state)[1].content
    update = nodes.assess_evidence_with_model(state, ClarifyingAssessment())
    assert update["clarification_question"] is None
    assert update["assessment"]["next_step"] == "limited_report"
    assert not update["assessment"]["sufficient"]
    assert any(e["event"] == "clarification_limit_reached" for e in update["trace"])
    apply_update(state, update)
    assert nodes.route_after_evidence_assessment(state) == "generate_draft"
    assert nodes._report_mode(state) == "limited"
    # 路由本身也防御直接构造/旧检查点中的 clarify。
    state["assessment"] = {"next_step": "clarify"}
    assert nodes.route_after_evidence_assessment(state) == "generate_draft"


def test_model_obeying_no_more_questions_still_exits_without_empty_round(monkeypatch):
    state = create_initial_state("研究")
    state["clarification_history"] = history()
    model = SequenceModel([AIMessage(content='{"decision":"assess","plan_update":{}}')])
    monkeypatch.setattr(nodes, "create_research_model", lambda: model)
    apply_update(state, nodes.agent_node(state))
    apply_update(state, nodes.research_guard_node(state))
    assert nodes.route_after_research_guard(state) == "finalize_limited_report"
    assert state["research_round"] == state["tool_call_count"] == 0
    assert "clarification_limit_reached" in state["program_limitations"]
    assert "澄清次数达到上限" in nodes.finalize_limited_report_node(state)["final_report"]


def setup_real_graph(monkeypatch, model):
    monkeypatch.setattr(nodes, "create_research_model", lambda: model)
    monkeypatch.setattr(graph_module, "recall_memory_node", lambda _: {})
    monkeypatch.setattr(graph_module, "assess_evidence_node", lambda _: {
        "assessment": {"sufficient": True, "next_step": "full_report"}})
    monkeypatch.setattr(graph_module, "generate_draft_node", lambda state: {
        "draft_report": "下沉运动抑制云雨。[E1]", "report_mode": nodes._report_mode(state)})
    monkeypatch.setattr(graph_module, "verify_report_node", lambda _: {"report_review": {"verdict": "pass"}})
    monkeypatch.setattr(graph_module, "store_memory_node", lambda _: {})
    return graph_module.build_research_graph(checkpointer=InMemorySaver())


def test_real_pause_resume_cancels_mixed_tools_and_next_model_input_is_valid(monkeypatch):
    model = SequenceModel([AIMessage(content="", tool_calls=[ask(), search()]),
                           AIMessage(content='{"decision":"assess","plan_update":{}}')])
    graph = setup_real_graph(monkeypatch, model)
    monkeypatch.setattr(graph_module, "tools_node", lambda _: pytest.fail("被取消的工具不应执行"))
    # 重新构图以绑定 tools 陷阱。
    graph = graph_module.build_research_graph(checkpointer=InMemorySaver())
    state = create_initial_state("研究")
    state.update(evidence=evidence(), research_round=MAX_RESEARCH_ROUNDS, tool_call_count=8)
    config = {"configurable": {"thread_id": state["run_id"]}}
    paused = graph.invoke(state, config)
    assert paused["__interrupt__"]
    assert_protocol_closed(paused["messages"])
    assert paused["tool_call_count"] == 8 and paused["research_round"] == MAX_RESEARCH_ROUNDS
    resumed = graph.invoke(Command(resume="北京"), config)
    assert_protocol_closed(resumed["messages"])
    assert len(resumed["clarification_history"]) == 1
    assert resumed["tool_call_count"] == 8 and resumed["research_round"] == MAX_RESEARCH_ROUNDS
    assert model.calls == 2 and resumed["final_report"]


def test_two_real_pauses_then_limit_keeps_normal_research_call(monkeypatch):
    model = SequenceModel([
        AIMessage(content="", tool_calls=[ask("a1")]),
        AIMessage(content="", tool_calls=[ask("a2", "哪一年？")]),
        AIMessage(content="", tool_calls=[ask("a3"), search()]),
        AIMessage(content='{"decision":"assess","plan_update":{}}'),
    ])
    calls = []

    @tool
    def search_local_knowledge(query: str) -> dict:
        """离线研究工具。"""
        calls.append(query)
        return {"status": "success", "results": [{"content": "下沉运动抑制云雨。",
                "source": {"location": "new.md", "display_name": "内部资料"}}]}

    monkeypatch.setattr(nodes, "RESEARCH_TOOLS", [search_local_knowledge])
    graph = setup_real_graph(monkeypatch, model)
    state = create_initial_state("研究")
    config = {"configurable": {"thread_id": state["run_id"]}}
    first = graph.invoke(state, config)
    assert first["__interrupt__"] and first["research_round"] == first["tool_call_count"] == 0
    second = graph.invoke(Command(resume="北京"), config)
    assert second["__interrupt__"] and len(second["clarification_history"]) == 1
    assert second["research_round"] == second["tool_call_count"] == 0
    result = graph.invoke(Command(resume="2021 年"), config)
    assert "__interrupt__" not in result or not result["__interrupt__"]
    assert len(result["clarification_history"]) == 2
    assert calls == ["下沉运动"]
    assert result["tool_call_count"] == result["research_round"] == 1
    assert len(result["evidence"]) == 1
    assert_protocol_closed(result["messages"])
    assert model.calls == 4


def test_agent_and_assessment_share_two_pauses_in_same_real_graph(monkeypatch):
    assess_response = AIMessage(content='{"decision":"assess","plan_update":{}}')
    model = SequenceModel([AIMessage(content="", tool_calls=[ask()]), assess_response, assess_response])
    setup_real_graph(monkeypatch, model)

    class Reviewer:
        def invoke(self, messages):
            return nodes.EvidenceAssessmentOutput(
                sufficient=False, covered_points=[], missing_points=["年份"], potential_conflicts=[],
                next_step="clarify", reason="年份未确认", clarification_question="哪一年？",
            )

    monkeypatch.setattr(graph_module, "assess_evidence_node",
                        lambda state: nodes.assess_evidence_with_model(state, Reviewer()))
    graph = graph_module.build_research_graph(checkpointer=InMemorySaver())
    state = create_initial_state("研究")
    state["evidence"] = evidence()
    config = {"configurable": {"thread_id": state["run_id"]}}
    assert graph.invoke(state, config)["__interrupt__"]
    second = graph.invoke(Command(resume="北京"), config)
    assert second["__interrupt__"][0].value["question"] == "哪一年？"
    result = graph.invoke(Command(resume="不清楚"), config)
    assert not result.get("__interrupt__")
    assert len(result["clarification_history"]) == 2
    assert result["assessment"]["next_step"] == "limited_report"
    assert result["report_mode"] == "limited"
    assert result["research_round"] == result["tool_call_count"] == 0
    assert_protocol_closed(result["messages"])
    assert any(e["event"] == "clarification_limit_reached" for e in result["trace"])


def test_control_responses_do_not_consume_an_extra_tool_slot_after_limit(monkeypatch):
    state = create_initial_state("研究")
    state.update(clarification_history=history(), research_round=MAX_RESEARCH_ROUNDS - 1, tool_call_count=7)
    model = SequenceModel([AIMessage(content="", tool_calls=[ask(), search()])])
    @tool
    def search_local_knowledge(query: str) -> dict:
        """不联网的空研究结果。"""
        return {"status": "success", "results": []}

    monkeypatch.setattr(nodes, "RESEARCH_TOOLS", [search_local_knowledge])
    monkeypatch.setattr(nodes, "create_research_model", lambda: model)
    apply_update(state, nodes.agent_node(state))
    apply_update(state, nodes.tools_node(state))
    apply_update(state, nodes.record_results_node(state))
    assert state["tool_call_count"] == 8 and state["research_round"] == MAX_RESEARCH_ROUNDS
    assert_protocol_closed(state["messages"])
