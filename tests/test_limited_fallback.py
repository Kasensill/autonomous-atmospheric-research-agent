from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.settings import MAX_RESEARCH_ROUNDS


def test_all_expired_weather_routes_to_failure_explanation():
    state = create_initial_state("当前天气")
    state.update(research_round=MAX_RESEARCH_ROUNDS, evidence=[{
        "source_type": "weather", "valid_at": "2000-01-01T00:00:00+00:00",
        "content": "过期快照", "source_location": "expired-weather",
    }])
    assert nodes.route_after_research_guard(state) == "finalize_limited_report"
    update = nodes.finalize_limited_report_node(state)
    assert "未取得可引用的正式证据" in update["final_report"]
    assert "expired-weather" not in update["final_report"]
    assert update["trace"][0]["event"] == "limited_report_without_evidence"
    assert state["evidence"][0]["content"] == "过期快照"


def test_unexpected_fallback_state_is_explained_without_exposing_draft():
    state = create_initial_state("研究")
    state.update(evidence=[{"source_type": "local_knowledge", "content": "资料"}],
                 draft_report="不可交付的草稿")
    update = nodes.finalize_limited_report_node(state)
    assert "异常状态" in update["final_report"]
    assert "不可交付的草稿" not in update["final_report"]
    assert update["trace"][0]["event"] == "unexpected_state"
