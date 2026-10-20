"""真实模型 HITL 验收：会调用配置的模型和工具，并按原流程保存合格报告记忆。"""

import argparse
import sys

from atmospheric_research_agent.agent.run import format_final_result, run_research


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", help="自动回答地点；不填则等待键盘输入")
    args = parser.parse_args()
    answers = []

    def answer(question: str) -> str:
        print(f"\n模型追问：{question}", flush=True)
        if answers:
            raise RuntimeError("已补充地点却再次追问；本次验收停止，需检查原因。")
        value = args.answer if args.answer is not None else input("你的回答：")
        if value.strip():
            answers.append(value.strip())
        print(f"回答：{value}", flush=True)
        return value

    result = run_research("请查询当前天气并给出带来源的简要分析。", answer_provider=answer)
    print(format_final_result(result, debug=True))
    checks = {
        "实际追问一次": len(answers) == 1,
        "回答记录保留": len(result.get("clarification_history", [])) == 1,
        "待回答问题清空": result.get("clarification_question") is None,
        "交付最终结果": bool(result.get("final_report")),
        "获取天气证据": any(item.get("source_type") == "weather" for item in result.get("evidence", [])),
    }
    print("\n真实模型验收：")
    for name, passed in checks.items():
        print(f"{'通过' if passed else '失败'}：{name}")
    if not all(checks.values()):
        raise RuntimeError("真实模型验收未全部通过，请检查上方运行轨迹。")


if __name__ == "__main__":
    main()
