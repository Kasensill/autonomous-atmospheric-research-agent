"""P20/P22/超长输入的回归：时钟、失效原因和显式单句格式。"""

import pytest
from langchain_core.messages import AIMessage

from atmospheric_research_agent.agent import graph as graphs, nodes
from atmospheric_research_agent.agent.evidence import usable_evidence
from atmospheric_research_agent.agent.state import create_initial_state


REFERENCE = "2026-10-25T04:00:00+00:00"


def state_with_clock(question="核实北京2026年7月1日试验资料。"):
    state = create_initial_state(question)
    state["reference_time"] = REFERENCE
    return state


def weather_record(valid_at):
    return {"evidence_id": "weather", "source_type": "weather", "valid_at": valid_at,
            "source_location": "https://example.org/weather", "source_display_name": "固定模型天气快照",
            "content": "旧快照气温12°C，不代表现在。"}


def test_agent_receives_explicit_utc_and_beijing_reference_time():
    text = "\n".join(str(m.content) for m in nodes._messages_for_model(state_with_clock()))
    assert REFERENCE in text
    assert "2026-10-25T12:00:00+08:00" in text
    assert "程序时间基准" in text
    assert "未检索不等于来源不存在" in text


@pytest.mark.parametrize("value", ["2026-10-25", "invalid", "2026-10-25T12:00:00", 42])
def test_replay_reference_time_must_be_aware_iso_timestamp(value):
    with pytest.raises(ValueError, match="reference_time"):
        create_initial_state("研究", reference_time=value)


def test_initial_state_accepts_explicit_replay_clock_without_changing_started_at():
    state = create_initial_state("研究", reference_time="2000-01-01T00:00:00+00:00")
    assert state["reference_time"].startswith("2000-")
    assert not state["started_at"].startswith("2000-")


def test_replay_clock_controls_both_usable_and_numbered_evidence():
    state = create_initial_state("天气")
    state["reference_time"] = "2000-01-01T00:30:00+00:00"
    fresh = weather_record("2000-01-01T00:00:00+00:00")
    state["evidence"] = [fresh]
    assert usable_evidence(state) == [fresh]
    assert nodes._numbered_evidence(state)[0]["citation_id"] == "E1"


def test_default_clock_is_read_on_each_call_not_frozen_at_task_start(monkeypatch):
    from datetime import UTC, datetime
    from atmospheric_research_agent.agent import clock

    times = iter([datetime(2026, 10, 25, tzinfo=UTC), datetime(2026, 10, 26, tzinfo=UTC)])

    class Clock:
        @staticmethod
        def now(tz):
            return next(times)

    monkeypatch.setattr(clock, "datetime", Clock)
    state = {"started_at": "2000-01-01T00:00:00+00:00", "reference_time": None}
    assert clock.reference_now(state).day == 25
    assert clock.reference_now(state).day == 26


def test_all_model_stages_receive_the_same_explicit_clock():
    state = state_with_clock("请解释副高。")
    state.update(draft_report="下沉抑云 [E1]。", report_review={"verdict": "revise", "required_changes": []})
    for messages in [nodes._assessment_messages(state), nodes._report_messages(state, "full"),
                     nodes._verification_messages(state), nodes._revision_messages(state)]:
        assert REFERENCE in "\n".join(str(m.content) for m in messages)


def test_run_research_accepts_explicit_clock_at_python_entry(monkeypatch):
    from atmospheric_research_agent.agent import run

    class Graph:
        def invoke(self, state, config):
            return {**state, "final_report": "固定结果"}

    monkeypatch.setattr(run, "build_research_graph", lambda **kw: Graph())
    result = run.run_research("固定回放", reference_time=REFERENCE)
    assert result["reference_time"] == REFERENCE
    assert result["final_report"] == "固定结果"


@pytest.mark.parametrize("tool_count", [0, 2])
def test_unchecked_stop_reason_never_becomes_user_facing_fact(tool_count):
    state = state_with_clock()
    state.update(research_stop_reason="该日期在未来，所有来源都不存在。", tool_call_count=tool_count)
    result = nodes.finalize_limited_report_node(state)
    assert "日期在未来" not in result["final_report"]
    assert "所有来源都不存在" not in result["final_report"]
    assert "未取得可引用的正式证据" in result["final_report"]
    assert ("未执行研究工具" if tool_count == 0 else "2 次研究工具调用") in result["final_report"]
    assert state["research_stop_reason"] == "该日期在未来，所有来源都不存在。"


def test_stale_weather_failure_explains_the_time_mismatch_without_reusing_value():
    state = state_with_clock("以2026年10月25日12时北京时间为基准查询北京气温。")
    old = weather_record("2026-10-24T22:00:00+00:00")
    state.update(evidence=[old], research_round=4)
    output = nodes.finalize_limited_report_node(state)["final_report"]
    assert "已过期" in output
    assert "2026-10-25T06:00:00+08:00" in output
    assert "2026-10-25T12:00:00+08:00" in output
    assert "12°C" not in output
    assert "[E1]" not in output
    assert state["evidence"] == [old]


@pytest.mark.parametrize("valid_at,expected", [(None, "有效时间无法确认"),
    ("2026-10-25T05:00:00+00:00", "晚于时间基准")])
def test_unusable_weather_reason_is_not_always_expired(valid_at, expected):
    state = state_with_clock()
    state["evidence"] = [weather_record(valid_at)]
    output = nodes.finalize_limited_report_node(state)["final_report"]
    assert expected in output


@pytest.mark.parametrize("draft", ["## 结论\n下沉抑云 [E1]。", "下沉抑云 [E1]。\n\n具体因条件而异。",
    "下沉抑云 [E1]。这有利于晴热。", "- 下沉抑云 [E1]。"])
def test_single_sentence_program_check_rejects_sections_lists_and_multiple_sentences(draft):
    state = state_with_clock("天气背景。" * 4000 + "请用一句话解释副高。")
    state["draft_report"] = draft
    assert nodes._single_sentence_issue(state)


@pytest.mark.parametrize("draft", ["850 hPa并不是唯一诊断高度 [E1]。", "阈值0.07只是相关性门槛 [E1]。",
    "有利条件不是充分条件。[E1]", "下沉常抑制云雨 [E1]"])
def test_single_sentence_check_preserves_numbers_and_citations(draft):
    state = state_with_clock("请用一句话解释。")
    state["draft_report"] = draft
    assert nodes._single_sentence_issue(state) is None


def test_explicit_negation_keeps_default_report_structure():
    state = state_with_clock("不要用一句话，请详细解释。")
    assert not nodes._single_sentence_requested(state)
    assert "结论或受限结论" in nodes._report_messages(state, "full")[0].content


def test_single_sentence_draft_and_revision_prompts_do_not_require_sections():
    state = state_with_clock("请用一句话解释。")
    state.update(draft_report="## 结论\n下沉抑云 [E1]。", report_review={"verdict": "revise"})
    prompt = nodes._report_messages(state, "limited")[0].content
    assert "不分章节" in prompt
    assert "1. 结论或受限结论" not in prompt
    for messages in [nodes._report_messages(state, "limited"), nodes._revision_messages(state), nodes._verification_messages(state)]:
        assert "单句正文" in "\n".join(str(m.content) for m in messages)


def test_program_forces_single_sentence_revision_even_when_model_passes(monkeypatch):
    state = state_with_clock("请用一句话解释副高。")
    state["evidence"] = [{"evidence_id": "local", "source_type": "local_knowledge",
                          "content": "下沉常抑制云雨。", "source_location": "test.md"}]
    drafts = iter(["## 结论\n下沉抑云 [E1]。\n\n## 解释\n有利于晴热。", "下沉常抑制云雨，有利于晴热 [E1]。"])

    class Writer:
        def invoke(self, messages):
            return AIMessage(content=next(drafts))

    class Reviewer:
        def invoke(self, messages):
            return nodes.ReportReviewOutput(verdict="pass", supported_points=[],
                unsupported_or_overstated_claims=[], citation_issues=[], required_changes=[], reason="模拟模型漏判格式")

    monkeypatch.setattr(graphs, "recall_memory_node", lambda s: {})
    monkeypatch.setattr(graphs, "agent_node", lambda s: {"messages": [AIMessage(content='{"decision":"assess"}')]})
    monkeypatch.setattr(graphs, "assess_evidence_node", lambda s: {"assessment": {"sufficient": True, "next_step": "full_report"}})
    monkeypatch.setattr(nodes, "_create_base_model", lambda: Writer())
    monkeypatch.setattr(nodes, "create_verification_model", lambda **kw: Reviewer())
    monkeypatch.setattr(nodes, "_verification_thinking", lambda: "disabled")
    monkeypatch.setattr(graphs, "store_memory_node", lambda s: {})
    result = graphs.build_research_graph().invoke(state)
    assert result["report_revision_count"] == 1
    assert result["draft_report"] == "下沉常抑制云雨，有利于晴热 [E1]。"
    assert "资料来源" in result["final_report"]
    assert result["report_review"]["verdict"] == "pass"
