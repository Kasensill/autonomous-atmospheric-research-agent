"""命令行入口：接收研究问题并运行一次完整 LangGraph。"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID
from collections.abc import Callable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .graph import build_research_graph
from .settings import GRAPH_RECURSION_LIMIT
from .state import ResearchState, create_initial_state

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _check_debug_directory_ignored(directory: Path) -> None:
    """只读确认诊断材料不会进入仓库；不修改任何忽略规则。"""
    relative_probe = (directory / "final_output.txt").relative_to(PROJECT_ROOT).as_posix()
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", relative_probe],
            cwd=PROJECT_ROOT, capture_output=True, check=False,
        )
    except OSError as error:
        raise RuntimeError("无法确认诊断目录已被 Git 忽略，停止保存；请先确认忽略规则。") from error
    if result.returncode != 0:
        raise RuntimeError("诊断目录未确认被 Git 忽略，停止保存；请先确认忽略规则。")


def _save_debug_materials(graph: Any, config: dict, result: ResearchState, directory: Path) -> None:
    """在运行完成后从检查点取出每个写作/核验节点的原始输出，不调用模型。"""
    _check_debug_directory_ignored(directory)
    # 当前检查点保留每步 task.result；不能只读终态，否则会丢失旧稿和旧审查。
    files: dict[str, str] = {}
    drafts = reviews = 0
    seen_tasks: set[str] = set()
    for snapshot in reversed(list(graph.get_state_history(config))):
        for task in snapshot.tasks:
            if task.id in seen_tasks or not isinstance(task.result, dict):
                continue
            seen_tasks.add(task.id)
            if task.name in {"generate_draft", "revise_report"} and "draft_report" in task.result:
                drafts += 1
                files[f"draft_{drafts:02d}.md"] = task.result["draft_report"]
            if task.name == "verify_report" and "report_review" in task.result:
                reviews += 1
                files[f"review_{reviews:02d}.json"] = json.dumps(
                    task.result["report_review"], ensure_ascii=False, indent=2,
                )
    # 不过滤过期资料、不重编号；保留原始证据正文、来源及已有时间元数据供复盘。
    evidence = [
        {"citation_id": f"E{index}", "published_or_observed_at": None,
         "valid_at": None, "fetched_at": None, **record}
        for index, record in enumerate(result.get("evidence", []), start=1)
    ]
    files["evidence.json"] = json.dumps(evidence, ensure_ascii=False, indent=2)
    files["final_report.md"] = result.get("final_report") or ""
    files["final_output.txt"] = format_final_result(result, debug=True) + "\n"
    directory.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")


def run_research(
    question: str, *, answer_provider: Callable[[str], str] | None = None, debug: bool = False,
) -> ResearchState:
    """同进程运行与恢复；没有回答接收器时将暂停结果返回给调用者。"""

    graph = build_research_graph(checkpointer=InMemorySaver())
    initial_state = create_initial_state(question)
    debug_directory = None
    if debug:
        # run_id 由程序创建；额外校验避免它被当作任意输出路径。
        run_id = str(UUID(initial_state["run_id"]))
        debug_directory = PROJECT_ROOT / "docs" / "validation" / datetime.now().astimezone().date().isoformat() / run_id
        _check_debug_directory_ignored(debug_directory)
    started = perf_counter()
    config = {"recursion_limit": GRAPH_RECURSION_LIMIT,
              "configurable": {"thread_id": initial_state["run_id"]}}
    result = graph.invoke(initial_state, config=config)
    waiting_seconds = 0.0
    while result.get("__interrupt__"):
        if answer_provider is None:
            # 未完成任务不能写 finished_at 或 run_completed。
            return result
        prompt = result["__interrupt__"][0].value["question"]
        waiting_started = perf_counter()
        answer = answer_provider(prompt)
        while not isinstance(answer, str) or not answer.strip():
            answer = answer_provider("回答不能为空。" + prompt)
        waiting_seconds += perf_counter() - waiting_started
        result = graph.invoke(Command(resume=answer.strip()), config=config)
    finished_at = datetime.now(UTC).isoformat()
    duration_ms = round((perf_counter() - started - waiting_seconds) * 1000, 2)
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
    if debug_directory is not None:
        _save_debug_materials(graph, config, result, debug_directory)
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
            diagnostics = {key: item[key] for key in ("tool_calls", "tool_name", "returned_count", "accepted_count", "duplicate_count", "relevance_gate", "top_scores", "clarification_quota") if key in item}
            diagnostics_text = f" {json.dumps(diagnostics, ensure_ascii=False)}" if diagnostics else ""
            trace_lines.append(
                f"- [{item.get('timestamp', 'unknown')}] "
                f"{item.get('node', 'unknown')} / {item.get('event', 'unknown')}"
                f" / {item.get('status', 'success')}{duration_text}：{item.get('detail', '')}{diagnostics_text}"
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
            f"- 程序确认的限制：{result.get('program_limitations', [])}",
            f"- 模型判断的限制：{result.get('assessment', {}).get('limitation_reasons', [])}",
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

    def read_answer(question: str) -> str:
        print(f"\n# 等待补充信息\n{question}\n当前任务已暂停，已有资料和预算保留。")
        return input("你的回答：")

    try:
        result = run_research(arguments.question, answer_provider=read_answer, debug=arguments.debug)
    except (EOFError, KeyboardInterrupt):
        print("\n已退出。本版检查点仅在内存中，关闭进程后不能恢复。")
        return
    print(format_final_result(result, debug=arguments.debug))


if __name__ == "__main__":
    main()
