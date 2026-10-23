"""核验 thinking 实验开关的离线测试；不连接服务端。"""

import json

import pytest

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.run import format_final_result
from atmospheric_research_agent.agent.state import create_initial_state


@pytest.fixture
def clients(monkeypatch):
    created = []
    monkeypatch.setattr(nodes, "load_dotenv", lambda: None)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-placeholder")
    monkeypatch.setenv("DEEPSEEK_MODEL", "offline-model")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://offline.invalid")

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.structured = None
            created.append(self)

        def bind_tools(self, tools):
            return self

        def with_structured_output(self, schema, **kwargs):
            self.structured = (schema, kwargs)
            return self

    monkeypatch.setattr(nodes, "ChatOpenAI", Client)
    return created


@pytest.mark.parametrize("setting", [None, "disabled", "enabled"])
def test_only_verification_client_can_enable_thinking(monkeypatch, clients, setting):
    if setting is None:
        monkeypatch.delenv("VERIFY_THINKING", raising=False)
    else:
        monkeypatch.setenv("VERIFY_THINKING", setting)
    nodes.create_research_model()
    nodes.create_assessment_model()
    nodes._create_base_model()  # 写作和修订也使用默认基础模型。
    verification = nodes.create_verification_model()
    baseline = {
        "model": "offline-model", "api_key": "offline-placeholder",
        "base_url": "https://offline.invalid", "temperature": 0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    assert all(client.kwargs == baseline for client in clients[:-1])
    assert verification.kwargs == {
        **baseline, "extra_body": {"thinking": {"type": setting or "disabled"}},
    }
    assert verification.structured == (nodes.ReportReviewOutput, {"method": "json_mode"})


@pytest.mark.parametrize("setting", ["", "true", "ENABLED", "invalid"])
def test_invalid_setting_fails_before_client_creation(monkeypatch, clients, setting):
    monkeypatch.setenv("VERIFY_THINKING", setting)
    with pytest.raises(ValueError, match="VERIFY_THINKING 仅允许 disabled 或 enabled"):
        nodes.create_verification_model()
    assert clients == []


def review_state():
    state = create_initial_state("下沉运动如何影响云雨？")
    state.update(
        evidence=[{"evidence_id": "one", "content": "下沉可抑制云雨。", "source_location": "theory.md"}],
        draft_report="下沉可抑制云雨。[E1]",
    )
    return state


@pytest.mark.parametrize("setting", ["disabled", "enabled"])
def test_node_trace_records_actual_mode_and_call_duration(monkeypatch, setting):
    monkeypatch.setattr(nodes, "load_dotenv", lambda: None)
    monkeypatch.setenv("VERIFY_THINKING", setting)
    captured = []

    class Model:
        reasoning_content = "hidden-reasoning-must-not-be-logged"

        def invoke(self, messages):
            return nodes.ReportReviewOutput(
                verdict="pass", supported_points=[], unsupported_or_overstated_claims=[],
                citation_issues=[], required_changes=[], reason="证据支持。",
            )

    def create_model(*, thinking):
        captured.append(thinking)
        return Model()

    ticks = iter([10.0, 10.125])
    monkeypatch.setattr(nodes, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(nodes, "create_verification_model", create_model)
    update = nodes.verify_report_node(review_state())
    assert captured == [setting]
    event = next(item for item in update["trace"] if item["event"] == "report_review_completed")
    assert f"thinking={setting}" in event["detail"]
    assert event["duration_ms"] == 125.0
    assert "hidden-reasoning" not in json.dumps(update)
    assert "reasoning_content" not in json.dumps(update)
    output = format_final_result({**review_state(), **update}, debug=True)
    assert f"thinking={setting}" in output
    assert "125.0 ms" in output


def test_json_mode_service_error_propagates_without_retry_or_fallback():
    original = RuntimeError("json_mode and thinking combination rejected by service")

    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise original

    model = Model()
    with pytest.raises(RuntimeError) as caught:
        nodes.verify_report_with_model(review_state(), model, thinking="enabled")
    assert caught.value is original
    assert model.calls == 1
    assert any("VERIFY_THINKING=enabled" in note and "json_mode" in note
               and "未静默回退" in note for note in original.__notes__)
