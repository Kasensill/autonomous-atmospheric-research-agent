"""完整研究流程的人工交互测试：问题和澄清回答均由用户输入。

直接调用项目正式 run_research / LangGraph，不替换模型、工具或报告节点。
会消耗已配置 API 的额度，并沿用正式流程的项目记忆读写规则。
"""

from __future__ import annotations

import sys

from atmospheric_research_agent.agent.run import format_final_result, run_research


def read_nonempty(prompt: str) -> str:
    """空输入留在当前提示，不触发模型或恢复图。"""
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("输入不能为空，请重新输入。")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("完整研究流程 · 人工交互测试")
    print("使用真实模型和工具；研究问题与后续补充均由你输入。")
    print("需要澄清时自动暂停，回答后继续原任务。无需澄清时直接研究。")
    print("本次使用内存检查点，退出进程后不能恢复。\n")

    def answer_provider(question: str) -> str:
        print(f"\n[任务暂停 · 等待你的补充]\n{question}", flush=True)
        print("已获得的资料和已用预算保留，等待期间不继续执行研究工具。")
        answer = read_nonempty("你的补充：")
        print("\n[恢复原任务 · 继续研究]", flush=True)
        return answer

    try:
        question = read_nonempty("请输入研究问题：")
        print("\n[开始研究]", flush=True)
        result = run_research(question, answer_provider=answer_provider)
    except (EOFError, KeyboardInterrupt):
        print("\n已退出交互测试。本进程内的任务检查点不会保留到下次启动。")
        return

    print("\n[研究流程结束]\n")
    print(format_final_result(result, debug=True))
    print("\n## 本次人工补充记录")
    history = result.get("clarification_history", [])
    if not history:
        print("本次任务没有触发澄清。")
    for index, item in enumerate(history, start=1):
        print(f"{index}. 系统追问：{item['question']}")
        print(f"   你的回答：{item['answer']}")


if __name__ == "__main__":
    main()
