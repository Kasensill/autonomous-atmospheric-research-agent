"""诊断材料来自真实图检查点；模型、工具和长期记忆均离线替换。"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from atmospheric_research_agent.agent import nodes
import atmospheric_research_agent.agent.graph as graph_module
import atmospheric_research_agent.agent.run as run_module


def setup_offline_run(monkeypatch, tmp_path, *, final_verdict="pass", clarify=False):
    monkeypatch.setattr(run_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_module, "_check_debug_directory_ignored", lambda _: None)
    records = [
        {"evidence_id": "old", "source_type": "weather", "content": "过期证据正文也要保留",
         "source_location": "weather-source", "published_or_observed_at": "2000-01-01",
         "valid_at": "2000-01-01T00:00:00+00:00", "fetched_at": "2000-01-01T00:01:00+00:00"},
        {"evidence_id": "current", "source_type": "local_knowledge", "content": "下沉运动抑制云雨。",
         "source_location": "theory.md", "source_display_name": "内部资料", "author_or_organization": "内部机构"},
    ]
    monkeypatch.setattr(graph_module, "recall_memory_node", lambda _: {"evidence": records})

    def agent(state):
        return {"messages": [AIMessage(content="{}")],
                "clarification_question": "哪个地区？" if clarify and not state["clarification_history"] else None}

    monkeypatch.setattr(graph_module, "agent_node", agent)
    monkeypatch.setattr(graph_module, "assess_evidence_node", lambda _: {
        "assessment": {"sufficient": True, "next_step": "full_report"}})

    class DraftModel:
        def invoke(self, messages):
            return AIMessage(content="原草稿全文，待修改。[E2]")

    class RevisionModel:
        def invoke(self, messages):
            return AIMessage(content="修订草稿全文。[E2]")

    reviews = []

    class Reviewer:
        def invoke(self, messages):
            verdict = "revise" if not reviews else final_verdict
            review = nodes.ReportReviewOutput(
                verdict=verdict, supported_points=["有证据"], unsupported_or_overstated_claims=[],
                citation_issues=[], required_changes=["收紧表述"] if verdict == "revise" else [],
                reason="完整审查理由",
            )
            reviews.append(review.model_dump())
            return review

    monkeypatch.setattr(graph_module, "generate_draft_node", lambda state: nodes.generate_draft_with_model(state, DraftModel()))
    monkeypatch.setattr(graph_module, "revise_report_node", lambda state: nodes.revise_report_with_model(state, RevisionModel()))
    monkeypatch.setattr(graph_module, "verify_report_node", lambda state: nodes.verify_report_with_model(state, Reviewer()))
    monkeypatch.setattr(graph_module, "store_memory_node", lambda _: {})
    return records, reviews


@pytest.mark.parametrize("final_verdict", ["pass", "revise"])
def test_debug_preserves_all_drafts_reviews_evidence_and_final_output(monkeypatch, tmp_path, final_verdict):
    records, reviews = setup_offline_run(monkeypatch, tmp_path, final_verdict=final_verdict, clarify=True)
    questions = []

    def answer(question):
        questions.append(question)
        return "西安"

    result = run_module.run_research("研究", answer_provider=answer, debug=True)
    directory = tmp_path / "docs" / "validation" / datetime.now().astimezone().date().isoformat() / result["run_id"]
    assert questions == ["哪个地区？"]
    assert (directory / "draft_01.md").read_text(encoding="utf-8") == "原草稿全文，待修改。[E2]"
    assert (directory / "draft_02.md").read_text(encoding="utf-8") == "修订草稿全文。[E2]"
    assert len(list(directory.glob("draft_*.md"))) == 2
    assert len(list(directory.glob("review_*.json"))) == 2
    assert json.loads((directory / "review_01.json").read_text(encoding="utf-8")) == reviews[0]
    assert json.loads((directory / "review_02.json").read_text(encoding="utf-8")) == reviews[1]
    saved_evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    for index, (saved, record) in enumerate(zip(saved_evidence, records), start=1):
        assert saved["citation_id"] == f"E{index}"
        assert all(saved[key] == value for key, value in record.items())
        assert "published_or_observed_at" in saved and "fetched_at" in saved
    assert len(saved_evidence) == 2
    assert (directory / "final_report.md").read_text(encoding="utf-8") == result["final_report"]
    output = (directory / "final_output.txt").read_text(encoding="utf-8")
    assert output == run_module.format_final_result(result, debug=True) + "\n"
    assert "clarification_received" in output and "run_started" in output and "run_completed" in output
    assert "原草稿全文" not in result["draft_report"]
    if final_verdict == "revise":
        assert "不输出该草稿" in result["final_report"]
        assert "修订草稿全文" not in result["final_report"]


@pytest.mark.parametrize("debug", [False, True])
def test_existing_cli_terminal_output_is_unchanged(monkeypatch, tmp_path, capsys, debug):
    setup_offline_run(monkeypatch, tmp_path)
    original = run_module.run_research
    received = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        received.append(result)
        return result

    monkeypatch.setattr(run_module, "run_research", capture)
    monkeypatch.setattr(run_module.sys, "argv", ["agent.run", "研究", *(["--debug"] if debug else [])])
    run_module.main()
    output = capsys.readouterr().out
    assert output == run_module.format_final_result(received[0], debug=debug) + "\n"
    assert bool(list(tmp_path.rglob("final_output.txt"))) == debug


def test_missing_ignore_confirmation_stops_before_model_or_files(monkeypatch, tmp_path):
    monkeypatch.setattr(run_module, "PROJECT_ROOT", tmp_path)
    commands = []

    def not_ignored(command, **kwargs):
        commands.append((command, kwargs))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(run_module.subprocess, "run", not_ignored)
    monkeypatch.setattr(nodes, "create_research_model", lambda: pytest.fail("忽略检查失败后不能调用模型"))
    with pytest.raises(RuntimeError, match="未确认被 Git 忽略"):
        run_module.run_research("研究", debug=True)
    assert commands[0][0][:4] == ["git", "check-ignore", "--quiet", "--no-index"]
    assert commands[0][1]["cwd"] == tmp_path
    assert not (tmp_path / "docs").exists()


def test_question_is_in_trace_only_for_ask_user_and_is_truncated(monkeypatch):
    question = "问" * 250

    class Model:
        def invoke(self, messages):
            return AIMessage(content="", tool_calls=[
                {"id": "a", "name": "ask_user", "args": {"question": question}},
                {"id": "b", "name": "search_local_knowledge", "args": {"query": "查" * 550, "question": "不应记录"}},
            ])

    monkeypatch.setattr(nodes, "create_research_model", Model)
    update = nodes.agent_node(run_module.create_initial_state("研究"))
    event = next(item for item in update["trace"] if item["event"] == "model_response")
    assert event["tool_calls"][0]["args"] == {"question": "问" * 200}
    assert event["tool_calls"][1]["args"] == {"query": "查" * 500}
    assert update["clarification_question"] == question
