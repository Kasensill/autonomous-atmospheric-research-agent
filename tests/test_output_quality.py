"""外部审查发现的数据丢失、来源与记忆策略的行为回归。"""

from langchain_core.messages import AIMessage, ToolMessage
import json

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.memory import save_project_memory, recall_project_memory
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.tools.weather import _weather_evidence_text
from atmospheric_research_agent.agent.graph import _instrument_node
from atmospheric_research_agent.agent.settings import MAX_RESEARCH_ROUNDS


def test_weather_fields_survive_tool_to_formal_evidence():
    text = _weather_evidence_text({"name": "北京"}, {"cloud_cover": 23, "wind_direction_10m": 90}, "Asia/Shanghai")
    message = ToolMessage(content=json.dumps({"status": "success", "evidence_text": text}),
                          name="get_current_weather", tool_call_id="w")
    records, _, _ = nodes._evidence_from_tool_message(message)
    assert "云量 23%" in records[0]["content"]
    assert "风向 90°" in records[0]["content"]
    assert "模型当前条件" in records[0]["content"]


def test_final_sources_only_include_cited_ids_without_renumbering():
    state = create_initial_state("研究")
    state.update(draft_report="结论 [E2]。", report_review={"verdict": "pass"}, evidence=[
        {"evidence_id": "1", "source_location": "unused.md"},
        {"evidence_id": "2", "source_location": "used.md"},
    ])
    result = nodes.finalize_report_node(state)["final_report"]
    assert "unused.md" not in result
    assert "- [E2]" in result
    assert "used.md" in result


def test_two_cited_fragments_at_same_source_keep_both_citation_ids():
    records = [{"citation_id": key, "source_display_name": "同页", "source_location": "same.md"} for key in ("E1", "E2")]
    result = nodes._reference_section(records, report="第一点 [E1]，第二点 [E2]")
    assert "- [E1]" in result and "- [E2]" in result


def test_same_batch_duplicates_are_not_formalized_twice():
    state = create_initial_state("研究")
    item = {"content": "同一正文", "source": {"location": "same.md"}}
    state["messages"] = [ToolMessage(name="search_local_knowledge", tool_call_id="a",
        content=json.dumps({"status": "success", "results": [item, item]}))]
    update = nodes.record_results_node(state)
    assert len(update["evidence"]) == 1
    assert update["trace"][0]["duplicate_count"] == 1


def test_budget_trace_explains_blocked_tool_request():
    state = create_initial_state("研究")
    state["research_round"] = MAX_RESEARCH_ROUNDS
    wrapped = _instrument_node("agent", lambda state: {"messages": [AIMessage(content="", tool_calls=[
        {"id": "a", "name": "get_current_weather", "args": {"location": "北京"}}])]})
    event = next(item for item in wrapped(state)["trace"] if item["event"] == "route_decision")
    assert "研究轮次耗尽" in event["detail"]
    assert event["destination"] == "research_guard"


def test_brief_report_cannot_pass_length_rule_even_if_model_passes():
    class Model:
        def invoke(self, messages):
            return nodes.ReportReviewOutput(verdict="pass", supported_points=[], unsupported_or_overstated_claims=[],
                citation_issues=[], required_changes=[], reason="假模型通过")
    state = create_initial_state("请简要说明")
    state.update(evidence=[{"evidence_id": "1"}], draft_report="长" * 950 + "[E1]")
    review = nodes.verify_report_with_model(state, Model())["report_review"]
    assert review["verdict"] == "revise"
    assert any("900" in change for change in review["required_changes"])


def test_weather_reports_skip_memory(monkeypatch):
    def unexpected(record):
        raise AssertionError("不应写入")
    monkeypatch.setattr(nodes, "save_project_memory", unexpected)
    for question, evidence, assessment in [
        ("当前天气", [], {"sufficient": True, "next_step": "full_report"}),
        ("机制", [{"source_type": "weather"}], {"sufficient": True, "next_step": "full_report"}),
    ]:
        state = create_initial_state(question)
        state.update(final_report="正文", report_review={"verdict": "pass"}, evidence=evidence, assessment=assessment)
        assert nodes.store_memory_node(state)["trace"][0]["event"] == "memory_skipped"


def test_legacy_weather_memory_not_recalled_and_file_is_preserved(tmp_path):
    path = tmp_path / "memory.json"
    save_project_memory({"question": "北京天气分析", "summary": "北京气温", "source_locations": ["https://api.open-meteo.com/v1/forecast"]}, store_path=path)
    assert recall_project_memory("北京天气分析", store_path=path) == []
    assert "open-meteo" in path.read_text(encoding="utf-8")


def test_verified_full_non_weather_report_still_writes_memory(monkeypatch):
    saved = []
    monkeypatch.setattr(nodes, "save_project_memory", saved.append)
    state = create_initial_state("副高机制")
    state.update(final_report="下沉抑制云雨", report_review={"verdict": "pass"},
                 assessment={"sufficient": True, "next_step": "full_report"})
    assert nodes.store_memory_node(state)["trace"][0]["event"] == "memory_stored"
    assert len(saved) == 1
