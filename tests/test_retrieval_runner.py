from atmospheric_research_agent.evaluation.retrieval_cases import RetrievalCase
from atmospheric_research_agent.evaluation.retrieval_metrics import SourceReference
from atmospheric_research_agent.evaluation.retrieval_runner import (
    evaluate_cases,
    render_markdown_report,
    summarize_results,
)


def _case() -> RetrievalCase:
    return RetrievalCase(
        case_id="demo",
        question="测试问题",
        category="demo",
        expected_sources=(SourceReference("demo.md", "正确章节"),),
        note="单元测试用题目。",
    )


def test_evaluate_cases_keeps_strategy_metrics_separate() -> None:
    def provider(query: str, top_k: int):
        assert query == "测试问题"
        assert top_k == 2
        correct = {"source": {"location": "demo.md", "section": "正确章节"}}
        wrong = {"source": {"location": "other.md", "section": "其他章节"}}
        return {
            "vector": [wrong, correct],
            "hybrid_rrf": [correct],
            "hybrid_rrf_rerank": [correct],
        }

    results = evaluate_cases([_case()], top_k=2, candidate_provider=provider)

    assert results[0].results[0].metrics is not None
    assert results[0].results[0].metrics.reciprocal_rank == 0.5
    assert results[0].results[1].metrics is not None
    assert results[0].results[1].metrics.reciprocal_rank == 1.0
    summary = summarize_results(results)
    assert summary["vector"]["mrr"] == 0.5
    assert summary["hybrid_rrf"]["hit_rate"] == 1.0


def test_evaluate_cases_records_provider_failure_without_stopping_batch() -> None:
    results = evaluate_cases(
        [_case()],
        top_k=1,
        candidate_provider=lambda query, top_k: (_ for _ in ()).throw(RuntimeError("API 不可用")),
    )

    assert all(result.metrics is None and result.error == "API 不可用" for result in results[0].results)
    report = render_markdown_report(results, top_k=1, generated_at="2026-09-18T00:00:00+08:00")
    assert "运行错误" in report
    assert "API 不可用" in report


def test_evaluate_cases_keeps_baselines_when_only_rerank_fails() -> None:
    correct = {"source": {"location": "demo.md", "section": "正确章节"}}
    results = evaluate_cases(
        [_case()],
        top_k=1,
        candidate_provider=lambda query, top_k: {
            "vector": [correct],
            "hybrid_rrf": [correct],
            "hybrid_rrf_rerank": RuntimeError("Rerank 配额不足"),
        },
    )

    vector_result, rrf_result, rerank_result = results[0].results
    assert vector_result.metrics is not None
    assert rrf_result.metrics is not None
    assert rerank_result.metrics is None
    assert rerank_result.error == "Rerank 配额不足"


def test_report_does_not_claim_a_gap_when_dataset_has_none() -> None:
    correct = {"source": {"location": "demo.md", "section": "正确章节"}}
    results = evaluate_cases(
        [_case()],
        top_k=1,
        candidate_provider=lambda query, top_k: {
            "vector": [correct],
            "hybrid_rrf": [correct],
            "hybrid_rrf_rerank": [correct],
        },
    )

    report = render_markdown_report(results, top_k=1, generated_at="2026-09-20T00:00:00+08:00")
    assert "当前题集没有标记为资料缺口" in report
