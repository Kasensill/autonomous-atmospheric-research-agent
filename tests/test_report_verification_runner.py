from atmospheric_research_agent.evaluation.report_verification_cases import ReportVerificationCase
from atmospheric_research_agent.evaluation.report_verification_runner import (
    evaluate_verification_cases,
    render_markdown_report,
    summarize_verification_results,
)


def _case(case_id: str, expected_verdict: str) -> ReportVerificationCase:
    return ReportVerificationCase(
        case_id=case_id,
        category="test",
        question="测试问题",
        evidence=(
            {
                "evidence_id": "e1",
                "content": "测试证据",
                "source_display_name": "测试来源",
                "source_location": "test.md",
                "source_type": "local_knowledge",
                "tool_name": "test",
            },
        ),
        draft_report="测试草稿 [E1]",
        expected_verdict=expected_verdict,  # type: ignore[arg-type]
        note="测试用例。",
    )


def test_verification_evaluation_counts_false_passes_separately() -> None:
    cases = [_case("good", "pass"), _case("bad", "revise")]
    results = evaluate_verification_cases(
        cases,
        reviewer=lambda case: {
            "verdict": "pass",
            "reason": "测试审查。",
            "citation_issues": [],
            "unsupported_or_overstated_claims": [],
        },
    )

    summary = summarize_verification_results(results)

    assert summary["matched_cases"] == 1
    assert summary["false_pass_count"] == 1
    assert summary["false_revise_count"] == 0
    report = render_markdown_report(results, generated_at="2026-09-20T00:00:00+08:00")
    assert "危险放行（应 revise 却 pass）：1" in report
    assert "需复盘：bad" in report


def test_verification_evaluation_keeps_running_after_one_reviewer_error() -> None:
    cases = [_case("broken", "revise"), _case("good", "pass")]

    def reviewer(case: ReportVerificationCase):
        if case.case_id == "broken":
            raise RuntimeError("模型不可用")
        return {
            "verdict": "pass",
            "reason": "测试审查。",
            "citation_issues": [],
            "unsupported_or_overstated_claims": [],
        }

    results = evaluate_verification_cases(cases, reviewer=reviewer)

    summary = summarize_verification_results(results)
    assert summary["failed_cases"] == 1
    assert summary["completed_cases"] == 1
    assert results[0].error == "模型不可用"
