from datetime import UTC, datetime, timedelta
import json

from langchain_core.messages import ToolMessage

from atmospheric_research_agent.agent.evidence import is_stale_weather, usable_evidence
from atmospheric_research_agent.agent.retrieval.rerank import filter_relevance
from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.evaluation.retrieval_metrics import calculate_retrieval_metrics, SourceReference


def test_gate_can_return_zero_and_missing_scores_are_not_accepted():
    assert filter_relevance([{"retrieval": {"rerank_score": 0.01}}, {}], 0.07) == []
    assert len(filter_relevance([{"retrieval": {"rerank_score": 0.8}}], 0.07)) == 1


def test_weather_freshness_uses_data_time_not_pause_duration():
    now = datetime(2026, 10, 20, 3, tzinfo=UTC)
    stale = {"source_type": "weather", "valid_at": (now - timedelta(hours=3)).isoformat()}
    fresh = {"source_type": "weather", "valid_at": (now - timedelta(minutes=20)).isoformat()}
    theory = {"source_type": "local_knowledge", "content": "通用机制"}
    assert is_stale_weather(stale, now=now)
    assert not is_stale_weather(fresh, now=now)
    assert usable_evidence({"evidence": [stale, fresh, theory]}, now=now) == [fresh, theory]


def test_stale_tool_message_is_masked_without_mutating_saved_state():
    state = create_initial_state("当前天气")
    original = ToolMessage(name="get_current_weather", tool_call_id="w", content=json.dumps({
        "valid_at": "2000-01-01T00:00:00+00:00", "temperature": 999}))
    state["messages"] = [original]
    messages = nodes._messages_for_model(state)
    assert "stale" in messages[-1].content
    assert "999" not in messages[-1].content
    assert "999" in original.content
    assert messages[-1].tool_call_id == "w"


def test_stale_citations_keep_original_ids_for_remaining_evidence():
    state = create_initial_state("研究")
    state["evidence"] = [{"source_type": "weather", "valid_at": "2000-01-01T00:00:00+00:00"},
                         {"source_type": "local_knowledge", "content": "机制"}]
    assert nodes._numbered_evidence(state)[0]["citation_id"] == "E2"


def test_unlabelled_is_not_automatically_noise_and_rank_not_compressed():
    good, bad = SourceReference("good", "s"), SourceReference("bad", "s")
    candidates = [{}, {"source": {"location": "good", "section": "s"}},
                  {"source": {"location": "bad", "section": "s"}}]
    metrics = calculate_retrieval_metrics(candidates, [good], top_k=3, irrelevant_sources=[bad])
    assert metrics.reciprocal_rank == 0.5
    assert metrics.labelled_noise_at_k == 1 / 3
    assert metrics.judgement_coverage_at_k == 2 / 3


def test_agent_input_contains_assessment_feedback():
    state = create_initial_state("研究")
    state["assessment"] = {"missing_points": ["需要低层水汽输送证据"]}
    assert "需要低层水汽输送证据" in nodes._messages_for_model(state)[1].content


def test_program_budget_reason_is_not_model_inference():
    from atmospheric_research_agent.agent.graph import _instrument_node
    from atmospheric_research_agent.agent.settings import MAX_RESEARCH_ROUNDS
    from langchain_core.messages import AIMessage
    state = create_initial_state("研究")
    state["research_round"] = MAX_RESEARCH_ROUNDS
    node = _instrument_node("agent", lambda state: {"messages": [AIMessage(content="", tool_calls=[{
        "id": "x", "name": "get_current_weather", "args": {"location": "北京"}}])]})
    assert node(state)["program_limitations"] == ["research_rounds_exhausted"]
