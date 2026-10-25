"""供用户手动核查实际进程重启；使用生产图/内存检查点，模型I/O全部固定，无API。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ..agent import graph as graphs
from ..agent.state import create_initial_state


def build_probe():
    # 保留真实图、request_clarification、interrupt、InMemorySaver和定稿节点。
    # 不创建真实模型/工具，不接触用户长期记忆。
    stack = ExitStack()

    def agent(state):
        update = {"messages": [AIMessage(content='{"decision":"assess"}')]}
        if not state["clarification_history"]:
            update["clarification_question"] = "请提供地点（手动检查点试验）"
        return update

    replacements = {
        "agent_node": agent,
        "recall_memory_node": lambda s: {},
        "assess_evidence_node": lambda s: {"assessment": {"sufficient": True, "next_step": "full_report"}},
        "generate_draft_node": lambda s: {"draft_report": "仅用于检查点试验的固定资料 [E1]。", "report_mode": "full"},
        "verify_report_node": lambda s: {"report_review": {"verdict": "pass"}},
        "store_memory_node": lambda s: {},
    }
    for name, value in replacements.items():
        stack.enter_context(patch.object(graphs, name, value))
    return graphs.build_research_graph(checkpointer=InMemorySaver()), stack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pause", "inspect"])
    parser.add_argument("--thread", default="restart-check")
    parser.add_argument("--output", type=Path, default=Path("docs/validation/2026-10-25/agent-supplement/manual-restart"))
    args = parser.parse_args()
    graph, patches = build_probe()
    config = {"configurable": {"thread_id": args.thread}}
    with patches:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.mode == "inspect":
            checkpoint = graph.get_state(config)
            result = {"pid": os.getpid(), "thread": args.thread, "checkpoint_present": bool(checkpoint.values),
                      "next_nodes": list(checkpoint.next), "interpretation": "新进程新InMemorySaver，不能恢复旧任务；磁盘诊断文件不是检查点。"}
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            (args.output / "new_process_inspection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        initial = create_initial_state("实际进程重启手动试验")
        initial.update(research_round=1, tool_call_count=2, evidence=[{
            "evidence_id": "manual-e1", "content": "检查点试验固定资料。", "source_type": "local_knowledge",
            "source_display_name": "手动试验固定资料", "source_location": "manual-fixture.md",
        }])
        paused = graph.invoke(initial, config)
        result = {"pid": os.getpid(), "thread": args.thread, "run_id": paused["run_id"],
                  "research_round": paused["research_round"], "tool_count": paused["tool_call_count"],
                  "evidence_count": len(paused["evidence"]), "checkpoint_present": bool(graph.get_state(config).values),
                  "next_nodes": list(graph.get_state(config).next), "note": "该JSON只是观测记录，不是持久检查点。"}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        (args.output / "paused_observation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            answer = input("同进程对照：输入北京并回车；重启试验：保持此处等待，在另一终端结束上方PID。\n")
        except (KeyboardInterrupt, EOFError):
            print("进程退出，内存检查点随之消失。", flush=True)
            return
        final = graph.invoke(Command(resume=answer), config)
        checks = {"same_run_id": final["run_id"] == initial["run_id"],
                  "round_preserved": final["research_round"] == 1,
                  "tools_preserved": final["tool_call_count"] == 2,
                  "evidence_preserved": final["evidence"] == initial["evidence"],
                  "one_answer": len(final["clarification_history"]) == 1,
                  "finished_report": bool(final["final_report"])}
        print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
        (args.output / "same_process_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
