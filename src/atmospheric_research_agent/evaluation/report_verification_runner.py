"""运行 V2 Reflection / Verification 评测并输出人工标注对照报告。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agent.nodes import create_verification_model, verify_report_with_model
from ..agent.state import ResearchState, create_initial_state
from .report_verification_cases import ReportVerificationCase, load_report_verification_cases


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evals" / "report_verification_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results"


@dataclass(frozen=True)
class VerificationEvaluationResult:
    """一份草稿的人工标签、实际审查结论和可诊断的错误信息。"""

    case_id: str
    category: str
    expected_verdict: str
    actual_verdict: str | None
    matched: bool | None
    reason: str | None
    citation_issues: tuple[str, ...]
    unsupported_claims: tuple[str, ...]
    error: str | None = None


Reviewer = Callable[[ReportVerificationCase], dict[str, Any]]


def review_case_with_production_model(case: ReportVerificationCase) -> dict[str, Any]:
    """将评测案例转换为最小 State，并真实调用当前生产核验逻辑。"""

    state: ResearchState = create_initial_state(case.question)
    state["evidence"] = list(case.evidence)
    state["draft_report"] = case.draft_report
    update = verify_report_with_model(state, create_verification_model())
    return update["report_review"]


def evaluate_verification_cases(
    cases: Iterable[ReportVerificationCase], *, reviewer: Reviewer
) -> list[VerificationEvaluationResult]:
    """逐题评测；单个模型调用失败不阻断其余案例。"""

    results: list[VerificationEvaluationResult] = []
    for case in cases:
        try:
            review = reviewer(case)
            actual = review.get("verdict")
            if actual not in {"pass", "revise"}:
                raise ValueError("审查结果没有合法 verdict。")
            results.append(
                VerificationEvaluationResult(
                    case_id=case.case_id,
                    category=case.category,
                    expected_verdict=case.expected_verdict,
                    actual_verdict=actual,
                    matched=actual == case.expected_verdict,
                    reason=review.get("reason"),
                    citation_issues=tuple(review.get("citation_issues", [])),
                    unsupported_claims=tuple(review.get("unsupported_or_overstated_claims", [])),
                )
            )
        except Exception as error:
            results.append(
                VerificationEvaluationResult(
                    case_id=case.case_id,
                    category=case.category,
                    expected_verdict=case.expected_verdict,
                    actual_verdict=None,
                    matched=None,
                    reason=None,
                    citation_issues=(),
                    unsupported_claims=(),
                    error=str(error),
                )
            )
    return results


def summarize_verification_results(
    results: Iterable[VerificationEvaluationResult],
) -> dict[str, int | float]:
    """汇总一致率及两种错误：危险放行和保守误拦。"""

    result_list = list(results)
    completed = [result for result in result_list if result.matched is not None]
    matched = [result for result in completed if result.matched]
    false_pass = [
        result
        for result in completed
        if result.expected_verdict == "revise" and result.actual_verdict == "pass"
    ]
    false_revise = [
        result
        for result in completed
        if result.expected_verdict == "pass" and result.actual_verdict == "revise"
    ]
    return {
        "case_count": len(result_list),
        "completed_cases": len(completed),
        "failed_cases": len(result_list) - len(completed),
        "matched_cases": len(matched),
        "verdict_accuracy": len(matched) / len(completed) if completed else 0.0,
        "false_pass_count": len(false_pass),
        "false_revise_count": len(false_revise),
    }


def render_markdown_report(
    results: list[VerificationEvaluationResult], *, generated_at: str
) -> str:
    """渲染一份便于人工复盘的核验器评测报告。"""

    summary = summarize_verification_results(results)
    lines = [
        "# Reflection / Verification 评测结果",
        "",
        f"- 生成时间：{generated_at}",
        f"- 案例数：{summary['case_count']}",
        f"- 完成案例：{summary['completed_cases']}",
        f"- Verdict 一致率：{summary['verdict_accuracy']:.3f}",
        f"- 危险放行（应 revise 却 pass）：{summary['false_pass_count']}",
        f"- 保守误拦（应 pass 却 revise）：{summary['false_revise_count']}",
        "",
        "## 逐题对照",
        "",
        "| 案例 | 类型 | 人工标注 | 实际 verdict | 是否一致 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        actual = result.actual_verdict or "ERROR"
        matched = "是" if result.matched else ("否" if result.matched is False else "运行失败")
        lines.append(
            f"| {result.case_id} | {result.category} | {result.expected_verdict} | {actual} | {matched} |"
        )

    for result in results:
        if result.matched and not result.error:
            continue
        lines.extend(["", f"## 需复盘：{result.case_id}", ""])
        if result.error:
            lines.append(f"- 运行错误：{result.error}")
            continue
        lines.append(f"- 人工标注：`{result.expected_verdict}`；实际：`{result.actual_verdict}`")
        if result.reason:
            lines.append(f"- 审查理由：{result.reason}")
        if result.unsupported_claims:
            lines.append("- 模型标出的过度主张：" + "；".join(result.unsupported_claims))
        if result.citation_issues:
            lines.append("- 引用问题：" + "；".join(result.citation_issues))
    return "\n".join(lines) + "\n"


def run_evaluation(*, cases_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """运行生产核验器并写出 Markdown 和 JSON 结果。"""

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    results = evaluate_verification_cases(
        load_report_verification_cases(cases_path), reviewer=review_case_with_production_model
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_path = output_dir / f"report_verification_evaluation_{timestamp}.md"
    json_path = output_dir / f"report_verification_evaluation_{timestamp}.json"
    markdown_path.write_text(render_markdown_report(results, generated_at=generated_at), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "summary": summarize_verification_results(results),
                "cases": [asdict(result) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Reflection / Verification 人工标注评测。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="案例 JSON 路径。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="报告输出目录。")
    arguments = parser.parse_args()
    markdown_path, json_path = run_evaluation(
        cases_path=arguments.cases,
        output_dir=arguments.output_dir,
    )
    print(f"评测完成。Markdown 报告：{markdown_path}")
    print(f"结构化结果：{json_path}")


if __name__ == "__main__":
    main()
