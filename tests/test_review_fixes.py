"""复盘缺陷回归：早停协议、澄清边界与证据支持范围，不调用外部 API。"""

import asyncio
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from atmospheric_research_agent.agent import graph as graphs, nodes
from atmospheric_research_agent.agent.settings import GRAPH_RECURSION_LIMIT
from atmospheric_research_agent.agent.state import create_initial_state


def test_stop_decision_requires_reason():
    parsed = nodes.parse_no_tool_decision(
        '{"decision":"stop","stop_reason":"用户明确说明对象为虚构，无法查询真实事件"}'
    )
    assert parsed.decision == "stop"
    assert parsed.stop_reason
    for reason in (None, "", "   "):
        with pytest.raises(ValueError):
            nodes.parse_no_tool_decision(json.dumps({"decision": "stop", "stop_reason": reason}))


def test_explicit_stop_ends_without_empty_research_rounds(monkeypatch):
    calls = []

    class Model:
        def invoke(self, messages):
            calls.append(messages)
            return AIMessage(content=json.dumps({
                "decision": "stop", "stop_reason": "用户明确说明地点为虚构，无法核实事件。",
            }, ensure_ascii=False))

    monkeypatch.setattr(nodes, "create_research_model", lambda: Model())
    monkeypatch.setattr(graphs, "recall_memory_node", lambda state: {})
    state = create_initial_state("分析虚构甲市的真实强降水；此地点明确是虚构的。")
    graph = graphs.build_research_graph(checkpointer=InMemorySaver())
    result = graph.invoke(state, {"recursion_limit": GRAPH_RECURSION_LIMIT,
                                  "configurable": {"thread_id": state["run_id"]}})
    assert len(calls) == 1
    assert result["research_round"] == result["tool_call_count"] == 0
    assert "未取得可引用的正式证据" in result["final_report"]
    assert "未执行研究工具" in result["final_report"]
    assert result["research_stop_reason"] == "用户明确说明地点为虚构，无法核实事件。"
    assert result["research_stop_reason"] not in result["final_report"]
    assert "预算耗尽" not in result["final_report"]
    assert any(e["event"] == "research_stopped_without_evidence" for e in result["trace"])
    assert not any(e["node"] == "store_memory" for e in result["trace"])


def test_stop_with_evidence_still_requires_assessment():
    state = evidence_state()
    state["research_stop_reason"] = "无法继续取得比较基准。"
    update = nodes.research_guard_node(state)
    state.update(update)
    assert nodes.route_after_research_guard(state) == "assess_evidence"

    class Model:
        def invoke(self, messages):
            return nodes.EvidenceAssessmentOutput(
                sufficient=False, covered_points=["已有天气"], missing_points=["气候基准"],
                potential_conflicts=[], next_step="continue", reason="缺少气候基准。",
            )

    result = nodes.assess_evidence_with_model(state, Model())
    assert result["assessment"]["next_step"] == "limited_report"
    assert nodes.route_after_evidence_assessment({**state, **result}) == "generate_draft"


def evidence_state():
    state = create_initial_state("简要说明北京当前天气；能否据此判断比常年偏冷？")
    state["evidence"] = [{"evidence_id": "test", "content": "北京气温12°C，不含气候基准或防护信息。",
                          "source_type": "web", "source_location": "https://example.org/weather"}]
    return state


def test_stop_with_evidence_cannot_bypass_report_verification(monkeypatch):
    class Agent:
        def invoke(self, messages):
            return AIMessage(content='{"decision":"stop","stop_reason":"无法继续取得比较基准。"}')

    class Assessment:
        def invoke(self, messages):
            return nodes.EvidenceAssessmentOutput(
                sufficient=False, covered_points=["温度"], missing_points=["基准"],
                potential_conflicts=[], next_step="continue", reason="缺少比较资料。",
            )

    class Writer:
        def invoke(self, messages):
            return AIMessage(content="仅有气温12°C，不能判断是否偏冷 [E1]。")

    reviews = []

    class Reviewer:
        def invoke(self, messages):
            reviews.append(messages)
            return nodes.ReportReviewOutput(verdict="pass", supported_points=["温度"],
                unsupported_or_overstated_claims=[], citation_issues=[], required_changes=[], reason="有限结论有引用。")

    monkeypatch.setattr(graphs, "recall_memory_node", lambda state: {})
    monkeypatch.setattr(nodes, "create_research_model", lambda: Agent())
    monkeypatch.setattr(nodes, "create_assessment_model", lambda: Assessment())
    monkeypatch.setattr(nodes, "_create_base_model", lambda: Writer())
    monkeypatch.setattr(nodes, "_verification_thinking", lambda: "disabled")
    monkeypatch.setattr(nodes, "create_verification_model", lambda **kwargs: Reviewer())
    result = graphs.build_research_graph().invoke(evidence_state())
    assert len(reviews) == 1
    assert result["report_mode"] == "limited"
    assert result["report_review"]["verdict"] == "pass"
    assert "受限报告：当前证据不全" in result["final_report"]
    assert any(t["event"] == "memory_skipped" for t in result["trace"])


def test_assessment_distinguishes_answer_limits_from_missing_user_conditions():
    prompt = nodes._assessment_messages(evidence_state())[0].content
    assert "能否据此判断" in prompt
    assert "不能判断" in prompt
    assert "不得为补齐资料而追问" in prompt
    assert "不得要求用户确认是否接受合成资料" in prompt


def test_writing_revision_and_review_preserve_causality_and_advice_scope():
    state = evidence_state()
    state.update(draft_report="一般无需特别防护 [E1]。", report_review={
        "verdict": "revise", "required_changes": ["删除无依据的保障结论"],
    })
    prompts = [nodes._report_messages(state, "limited")[0].content,
               nodes._revision_messages(state)[0].content,
               nodes._verification_messages(state)[0].content]
    for prompt in prompts:
        assert "有利条件不等于普遍必要条件" in prompt
        assert "无需特别防护" in prompt
    assert "属于无证据事实主张" in prompts[2]


def test_knowledge_does_not_make_low_level_jet_a_universal_requirement():
    from atmospheric_research_agent.ingestion.build_index import split_markdown_file

    root = Path(__file__).resolve().parents[1] / "data/knowledge_base"
    chunks = split_markdown_file(root / "03_大气动力学与环流/23_低空急流与强降水.md", root)
    assert chunks
    for chunk in chunks:
        # 现在是北美研究原文：安全边界由来源范围传给模型，不能要求原作者写旧中文摘要。
        scope = chunk.metadata["source_scope"]
        assert chunk.metadata["content_kind"] == "general_mechanism"
        assert "不是东亚暴雨的直接观测证明" in scope
        assert "区域阈值不能普遍化" in scope
        assert "不能推出低空急流是所有暴雨必要或充分条件" in scope
        assert scope in chunk.text
        assert "为暴雨提供必要环境条件之一" not in chunk.text

    from langchain_core.messages import ToolMessage
    from atmospheric_research_agent.agent.tools.local_knowledge import _source_from_metadata

    chunk = chunks[0]
    source = _source_from_metadata(chunk.metadata)
    evidence = nodes._evidence_from_local_knowledge(
        ToolMessage(content="", name="search_local_knowledge", tool_call_id="offline"),
        {"status": "success", "results": [{"content": chunk.text, "source": source}]},
    )
    assert evidence[0]["source_scope"] == chunk.metadata["source_scope"]
    state = create_initial_state("850 hPa 低空急流与暴雨有什么关系？")
    state["evidence"] = evidence
    assert scope in nodes._assessment_messages(state)[1].content


@pytest.mark.parametrize("method", ["invoke", "stream", "ainvoke", "astream"])
def test_public_graph_entry_rejects_bad_resume_before_checkpoint_write(method, monkeypatch):
    monkeypatch.setattr(graphs, "recall_memory_node", lambda state: {})
    monkeypatch.setattr(graphs, "agent_node", lambda state: {
        "clarification_question": "地点？", "messages": [AIMessage(content="")],
    })
    graph = graphs.build_research_graph(checkpointer=InMemorySaver())
    state = create_initial_state("当前天气")
    config = {"configurable": {"thread_id": state["run_id"]}}
    graph.invoke(state, config)
    before = graph.get_state(config)
    for answer in (None, "", " ", 42, {"interrupt-id": ""}):
        with pytest.raises(ValueError, match="非空文本"):
            if method == "invoke":
                graph.invoke(Command(resume=answer), config)
            elif method == "stream":
                list(graph.stream(Command(resume=answer), config))
            else:
                async def execute():
                    if method == "ainvoke":
                        await graph.ainvoke(Command(resume=answer), config)
                    else:
                        async for _ in graph.astream(Command(resume=answer), config):
                            pass
                asyncio.run(execute())
        after = graph.get_state(config)
        assert after.config == before.config
        assert after.values == before.values
