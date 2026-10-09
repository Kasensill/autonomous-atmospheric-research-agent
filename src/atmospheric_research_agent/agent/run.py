"""命令行入口：接收研究问题并运行一次完整 LangGraph。"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from .graph import build_research_graph
from .settings import GRAPH_RECURSION_LIMIT
from .state import ResearchState, create_initial_state


def run_research(question: str) -> ResearchState:
    """从初始 State 开始执行整张研究图，并返回最终 State。"""

    graph = build_research_graph()
    initial_state = create_initial_state(question)
    started = perf_counter()
    result: ResearchState = graph.invoke(
        initial_state,
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )
    finished_at = datetime.now(UTC).isoformat()
    duration_ms = round((perf_counter() - started) * 1000, 2)
    result["finished_at"] = finished_at
    result["run_duration_ms"] = duration_ms
    result.setdefault("run_id", initial_state["run_id"])
    result.setdefault("started_at", initial_state["started_at"])
    result.setdefault("trace", []).append(
        {
            "node": "run",
            "event": "run_completed",
            "detail": "本次研究任务已到达图的结束节点。",
            "timestamp": finished_at,
            "status": "success",
            "duration_ms": duration_ms,
        }
    )
    return result


def format_final_result(result: ResearchState, *, debug: bool = False) -> str:
    """把图的终态转换成适合命令行阅读的文本。"""

    clarification_question = result.get("clarification_question")
    if clarification_question:
        output = f"# 需要补充信息\n\n{clarification_question}"
    elif result.get("final_report"):
        output = result["final_report"]
    else:
        output = "# 运行未产生报告\n\n请使用 --debug 查看运行轨迹。"

    if debug:
        trace = result.get("trace", [])
        trace_lines = []
        for item in trace:
            duration = item.get("duration_ms")
            duration_text = f" / {duration} ms" if duration is not None else ""
            trace_lines.append(
                f"- [{item.get('timestamp', 'unknown')}] "
                f"{item.get('node', 'unknown')} / {item.get('event', 'unknown')}"
                f" / {item.get('status', 'success')}{duration_text}：{item.get('detail', '')}"
            )
        debug_parts = [
            "## 运行摘要",
            "",
            f"- 运行 ID：{result.get('run_id', 'unknown')}",
            f"- 开始时间（UTC）：{result.get('started_at', 'unknown')}",
            f"- 总耗时：{result.get('run_duration_ms', 'unknown')} ms",
            f"- 研究轮次：{result.get('research_round', 0)}",
            f"- 工具调用数：{result.get('tool_call_count', 0)}",
            f"- 正式证据数：{len(result.get('evidence', []))}",
        ]
        review = result.get("report_review")
        if review:
            debug_parts.extend(
                [
                    "",
                    "## 报告核验",
                    "",
                    f"- 审查结论：{review.get('verdict', 'unknown')}",
                    f"- 已修订次数：{result.get('report_revision_count', 0)}",
                    f"- 审查理由：{review.get('reason', '未提供')}",
                ]
            )
            for title, field_name in (
                ("已支持的关键点", "supported_points"),
                ("过度主张或缺证据项", "unsupported_or_overstated_claims"),
                ("引用问题", "citation_issues"),
                ("要求修改", "required_changes"),
            ):
                items = review.get(field_name, [])
                if items:
                    debug_parts.extend(["", f"### {title}", ""])
                    debug_parts.extend(f"- {item}" for item in items)
        debug_parts.extend(
            [
                "",
                "## 运行轨迹",
                "",
                *(trace_lines or ["- 本次没有记录轨迹。"]),
            ]
        )
        debug_section = "\n".join(debug_parts)
        output = f"{output}\n\n{debug_section}"

    return output


def main() -> None:
    """解析命令行参数并打印本轮研究的终态输出。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="运行一次大气科学 Autonomous Research Agent。")
    parser.add_argument("question", help="需要研究的大气科学问题。")
    parser.add_argument("--debug", action="store_true", help="额外显示研究轮次、工具数和运行轨迹。")
    arguments = parser.parse_args()

    result = run_research(arguments.question)
    print(format_final_result(result, debug=arguments.debug))


if __name__ == "__main__":
    main()
