"""命令行入口：接收研究问题并运行一次完整 LangGraph。"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .graph import build_research_graph
from .settings import GRAPH_RECURSION_LIMIT
from .state import ResearchState, create_initial_state


def run_research(question: str) -> ResearchState:
    """从初始 State 开始执行整张研究图，并返回最终 State。"""

    graph = build_research_graph()
    return graph.invoke(
        create_initial_state(question),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
    )


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
        trace_lines = [
            f"- {item.get('node', 'unknown')} / {item.get('event', 'unknown')}："
            f"{item.get('detail', '')}"
            for item in trace
        ]
        debug_section = "\n".join(
            [
                "## 运行摘要",
                "",
                f"- 研究轮次：{result.get('research_round', 0)}",
                f"- 工具调用数：{result.get('tool_call_count', 0)}",
                f"- 正式证据数：{len(result.get('evidence', []))}",
                "",
                "## 运行轨迹",
                "",
                *(trace_lines or ["- 本次没有记录轨迹。"]),
            ]
        )
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
