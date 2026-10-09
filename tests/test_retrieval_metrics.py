import pytest

from atmospheric_research_agent.evaluation.retrieval_metrics import (
    SourceReference,
    calculate_retrieval_metrics,
    mean_metrics,
    source_reference_from_candidate,
)


def _candidate(location: str, section: str) -> dict[str, object]:
    return {"source": {"location": location, "section": section}}


def test_metrics_capture_first_relevant_rank_and_multi_source_coverage() -> None:
    expected = {
        SourceReference("dynamics/jet.md", "影响与表现"),
        SourceReference("hazards/rain.md", "形成机制"),
    }

    metrics = calculate_retrieval_metrics(
        [
            _candidate("other.md", "定义与背景"),
            _candidate("dynamics/jet.md", "影响与表现"),
            _candidate("hazards/rain.md", "形成机制"),
        ],
        expected,
        top_k=3,
    )

    assert metrics.hit_at_k is True
    assert metrics.first_relevant_rank == 2
    assert metrics.reciprocal_rank == 0.5
    assert metrics.coverage_at_k == 1.0


def test_metrics_ignore_candidates_without_complete_source_reference() -> None:
    expected = {SourceReference("dynamics/jet.md", "影响与表现")}

    metrics = calculate_retrieval_metrics(
        [{"source": {"location": "dynamics/jet.md"}}],
        expected,
        top_k=1,
    )

    assert metrics.hit_at_k is False
    assert metrics.reciprocal_rank == 0.0
    assert metrics.coverage_at_k == 0.0
    assert metrics.first_relevant_rank is None


def test_mean_metrics_calculates_strategy_level_summary() -> None:
    first = calculate_retrieval_metrics(
        [_candidate("a.md", "定义与背景")],
        {SourceReference("a.md", "定义与背景")},
        top_k=1,
    )
    second = calculate_retrieval_metrics(
        [_candidate("other.md", "定义与背景")],
        {SourceReference("b.md", "形成机制")},
        top_k=1,
    )

    assert mean_metrics([first, second]) == {
        "hit_rate": 0.5,
        "mrr": 0.5,
        "coverage": 0.5,
    }


def test_metrics_reject_invalid_evaluation_contract() -> None:
    with pytest.raises(ValueError, match="top_k"):
        calculate_retrieval_metrics([], {SourceReference("a.md", "s")}, top_k=0)
    with pytest.raises(ValueError, match="expected_sources"):
        calculate_retrieval_metrics([], set(), top_k=1)


def test_source_reference_rejects_incomplete_candidate() -> None:
    assert source_reference_from_candidate({"source": {"location": "a.md"}}) is None
