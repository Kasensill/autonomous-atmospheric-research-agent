"""运行 V1.6 本地知识库 A/B/C 检索评测，并生成可阅读的结果报告。

本模块刻意不评测 Agent 的工具选择或最终文笔；它只回答一个问题：在同一批
人工标注问题上，向量、RRF 与 Rerank 三条检索路径各自找回了多少正确资料。
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agent.tools.local_knowledge import compare_local_retrievers
from .retrieval_cases import RetrievalCase, load_retrieval_cases
from .retrieval_metrics import RetrievalMetrics, calculate_retrieval_metrics, mean_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASES_PATH = PROJECT_ROOT / "evals" / "retrieval_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evals" / "results"
STRATEGIES = ("vector", "hybrid_rrf", "hybrid_rrf_rerank")


@dataclass(frozen=True)
class StrategyCaseResult:
    """一条题在一条检索策略上的指标或运行失败信息。"""

    strategy: str
    metrics: RetrievalMetrics | None
    candidate_count: int
    error: str | None = None


@dataclass(frozen=True)
class RetrievalCaseResult:
    """单条评测题的三策略结果。"""

    case_id: str
    question: str
    category: str
    known_data_gap: bool
    results: tuple[StrategyCaseResult, ...]


def candidates_by_strategy(
    query: str, *, top_k: int
) -> dict[str, list[dict[str, Any]] | Exception]:
    """对同一问题真实执行一次双路召回与可选精排，并拆成 A/B/C 对照列表。"""

    comparison = compare_local_retrievers(query, include_rerank=True)
    if comparison.get("status") != "success":
        raise RuntimeError(str(comparison.get("message", "本地检索对比失败。")))
    candidates: dict[str, list[dict[str, Any]] | Exception] = {
        "vector": list(comparison["vector_candidates"])[:top_k],
        "hybrid_rrf": list(comparison["fused_candidates"])[:top_k],
    }
    if comparison.get("rerank_status") == "success":
        candidates["hybrid_rrf_rerank"] = list(comparison["reranked_candidates"])[:top_k]
    else:
        candidates["hybrid_rrf_rerank"] = RuntimeError(
            "Rerank 路径不可用：" + str(comparison.get("rerank_message", "未提供错误说明"))
        )
    return candidates


CandidateProvider = Callable[[str, int], Mapping[str, list[dict[str, Any]] | Exception]]


def evaluate_cases(
    cases: Iterable[RetrievalCase],
    *,
    top_k: int,
    candidate_provider: CandidateProvider,
) -> list[RetrievalCaseResult]:
    """运行给定题集；单题失败被记录，不让一次 API 故障中断整批评测。"""

    if top_k < 1:
        raise ValueError("top_k 必须至少为 1。")

    case_results: list[RetrievalCaseResult] = []
    for case in cases:
        try:
            candidates_for_case = candidate_provider(case.question, top_k)
        except Exception as error:  # 保留完整批次报告，错误会显式出现在各策略列。
            strategy_results = tuple(
                StrategyCaseResult(strategy=strategy, metrics=None, candidate_count=0, error=str(error))
                for strategy in STRATEGIES
            )
        else:
            strategy_results_list: list[StrategyCaseResult] = []
            for strategy in STRATEGIES:
                candidates = candidates_for_case.get(strategy, [])
                if isinstance(candidates, Exception):
                    strategy_results_list.append(
                        StrategyCaseResult(
                            strategy=strategy,
                            metrics=None,
                            candidate_count=0,
                            error=str(candidates),
                        )
                    )
                    continue
                strategy_results_list.append(
                    StrategyCaseResult(
                        strategy=strategy,
                        metrics=calculate_retrieval_metrics(
                            candidates, case.expected_sources, top_k=top_k
                        ),
                        candidate_count=len(candidates),
                    )
                )
            strategy_results = tuple(strategy_results_list)
        case_results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                question=case.question,
                category=case.category,
                known_data_gap=case.known_data_gap,
                results=strategy_results,
            )
        )
    return case_results


def summarize_results(results: Iterable[RetrievalCaseResult]) -> dict[str, dict[str, float | int]]:
    """汇总每条策略的成功题数和简单平均指标。"""

    grouped: dict[str, list[RetrievalMetrics]] = {strategy: [] for strategy in STRATEGIES}
    failures: dict[str, int] = {strategy: 0 for strategy in STRATEGIES}
    for case_result in results:
        for result in case_result.results:
            if result.metrics is None:
                failures[result.strategy] += 1
            else:
                grouped[result.strategy].append(result.metrics)

    summary: dict[str, dict[str, float | int]] = {}
    for strategy in STRATEGIES:
        successful_metrics = grouped[strategy]
        if successful_metrics:
            summary[strategy] = {
                "successful_cases": len(successful_metrics),
                "failed_cases": failures[strategy],
                **mean_metrics(successful_metrics),
            }
        else:
            summary[strategy] = {"successful_cases": 0, "failed_cases": failures[strategy]}
    return summary


def _metric_cell(result: StrategyCaseResult) -> str:
    if result.metrics is None:
        return "ERROR"
    return (
        f"Hit={int(result.metrics.hit_at_k)} | MRR={result.metrics.reciprocal_rank:.2f}"
        f" | Cov={result.metrics.coverage_at_k:.2f}"
    )


def render_markdown_report(
    results: list[RetrievalCaseResult], *, top_k: int, generated_at: str
) -> str:
    """渲染人能审阅的 Markdown 报告，不依赖任何第三方格式化库。"""

    summary = summarize_results(results)
    lines = [
        "# 本地检索 A/B/C 评测结果",
        "",
        f"- 生成时间：{generated_at}",
        f"- 评测题数：{len(results)}", 
        f"- 指标窗口：Top {top_k}",
    ]
    if any(case.known_data_gap for case in results):
        lines.append(
            "- 注意：带 `资料缺口` 的题用于诊断当前知识库覆盖范围，不能简单视为检索算法失败。"
        )
    else:
        lines.append("- 当前题集没有标记为资料缺口的案例；指标均按现有资料的应召回来源计算。")
    lines.extend(
        [
            "",
            "## 总体指标",
            "",
            "| 策略 | 成功题数 | 失败题数 | Hit@K | MRR@K | Coverage@K |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy, metrics in summary.items():
        if not metrics["successful_cases"]:
            lines.append(
                f"| {strategy} | 0 | {metrics['failed_cases']} | — | — | — |"
            )
            continue
        lines.append(
            f"| {strategy} | {metrics['successful_cases']} | {metrics['failed_cases']}"
            f" | {metrics['hit_rate']:.3f} | {metrics['mrr']:.3f} | {metrics['coverage']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 每题结果",
            "",
            "| 题目 ID | 类别 | 标记 | 纯向量 | RRF | RRF + Rerank |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for case in results:
        cells = [_metric_cell(result) for result in case.results]
        gap = "资料缺口" if case.known_data_gap else ""
        lines.append(
            f"| {case.case_id} | {case.category} | {gap} | " + " | ".join(cells) + " |"
        )

    errors = [
        (case.case_id, result.strategy, result.error)
        for case in results
        for result in case.results
        if result.error
    ]
    if errors:
        lines.extend(["", "## 运行错误", ""])
        lines.extend(f"- `{case_id}` / `{strategy}`：{error}" for case_id, strategy, error in errors)
    return "\n".join(lines) + "\n"


def _json_payload(
    results: list[RetrievalCaseResult], *, top_k: int, generated_at: str
) -> dict[str, Any]:
    return {
        "generated_at": generated_at,
        "top_k": top_k,
        "summary": summarize_results(results),
        "cases": [asdict(case) for case in results],
    }


def run_evaluation(*, cases_path: Path, output_dir: Path, top_k: int) -> tuple[Path, Path]:
    """读取题集、运行真实 A/B/C 检索，并落盘 JSON 与 Markdown 结果。"""

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    cases = load_retrieval_cases(cases_path)
    results = evaluate_cases(
        cases,
        top_k=top_k,
        candidate_provider=lambda query, window: candidates_by_strategy(query, top_k=window),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_path = output_dir / f"retrieval_evaluation_{timestamp}.md"
    json_path = output_dir / f"retrieval_evaluation_{timestamp}.json"
    markdown_path.write_text(
        render_markdown_report(results, top_k=top_k, generated_at=generated_at), encoding="utf-8"
    )
    json_path.write_text(
        json.dumps(_json_payload(results, top_k=top_k, generated_at=generated_at), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="运行本地知识库检索 A/B/C 评测。")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="评测题集 JSON 路径。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="报告输出目录。")
    parser.add_argument("--top-k", type=int, default=4, help="评测窗口大小，默认 4。")
    arguments = parser.parse_args()
    markdown_path, json_path = run_evaluation(
        cases_path=arguments.cases,
        output_dir=arguments.output_dir,
        top_k=arguments.top_k,
    )
    print(f"评测完成。Markdown 报告：{markdown_path}")
    print(f"结构化结果：{json_path}")


if __name__ == "__main__":
    main()
