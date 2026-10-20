"""离线演示暂停恢复：真实检查点与项目澄清节点，模拟研究，不调用 API。"""

import argparse
import sys

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from atmospheric_research_agent.agent.nodes import request_clarification_node
from atmospheric_research_agent.agent.state import ResearchState, create_initial_state


def prepare_demo(state: ResearchState) -> dict:
    """只准备一次模拟资料，让暂停前后有可比较的数据。"""
    print("[准备] 写入一条模拟证据，提出年份追问。")
    return {
        "evidence": [{"evidence_id": "demo_1", "content": "演示占位资料，不是科学证据。"}],
        "research_round": 1,
        "tool_call_count": 2,
        "clarification_question": "你想研究哪一年的事件？",
        "messages": [AIMessage(content="请补充事件年份。")],
    }


def continue_demo(state: ResearchState) -> dict:
    print("[继续] 收到回答：", state["clarification_history"][-1]["answer"])
    return {"final_report": "演示完成：这里只验证恢复机制，没有生成真实研究报告。"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", help="自动提供回答，用于复现；不传则等待手动输入")
    args = parser.parse_args()

    builder = StateGraph(ResearchState)
    builder.add_node("prepare", prepare_demo)
    builder.add_node("request_clarification", request_clarification_node)
    builder.add_node("continue_research", continue_demo)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "request_clarification")
    builder.add_edge("request_clarification", "continue_research")
    builder.add_edge("continue_research", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    initial = create_initial_state("研究某次暴雨的成因（演示）")
    config = {"configurable": {"thread_id": initial["run_id"]}}
    paused = graph.invoke(initial, config)
    print("\n[暂停]", paused["__interrupt__"][0].value["question"])
    print("任务 ID：", paused["run_id"])
    print("证据数：", len(paused["evidence"]))
    print("研究轮次 / 工具次数：", paused["research_round"], "/", paused["tool_call_count"])
    print("后续节点尚未执行：", paused["final_report"] is None)
    print("输入前任务处于暂停状态，内存检查点仍保留数据。")
    answer = args.answer if args.answer is not None else input("你的回答：")
    while not answer.strip():
        answer = input("回答不能为空，请重新输入：")

    resumed = graph.invoke(Command(resume=answer.strip()), config)
    checks = {
        "同一任务 ID": resumed["run_id"] == paused["run_id"],
        "原有证据完整保留且未重复追加": resumed["evidence"] == paused["evidence"],
        "研究轮次保留": resumed["research_round"] == paused["research_round"],
        "工具次数保留": resumed["tool_call_count"] == paused["tool_call_count"],
        "原始问题保留": resumed["question"] == initial["question"],
        "回答只记录一次": len(resumed["clarification_history"]) == 1,
        "追问已清除": resumed["clarification_question"] is None,
        "已到达 END": not graph.get_state(config).next,
    }
    print("\n[恢复验证]")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}：{name}")
    if not all(checks.values()):
        raise RuntimeError("暂停恢复验证失败。")
    print(resumed["final_report"])


if __name__ == "__main__":
    main()
