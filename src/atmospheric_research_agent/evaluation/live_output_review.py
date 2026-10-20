"""真实端到端验收：缺地点追问→回答北京→天气证据→简要报告→记忆策略。

会调用项目配置的外部 API，保留实际报告与验收摘要；失败不能当成通过。
"""

import json
from datetime import datetime
from pathlib import Path

from ..agent.run import run_research, format_final_result
from ..agent.nodes import _numbered_evidence
from ..agent.settings import BRIEF_REPORT_MAX_CHARACTERS

ROOT = Path(__file__).resolve().parents[3]


def main():
    answers = []
    def answer(question):
        if answers:
            raise RuntimeError("已回答地点却再次追问，人工检查；不自动重复回答")
        answers.append({"question": question, "answer": "北京"})
        return "北京"
    result = run_research("请查询当前天气，并结合大气科学知识解释目前的天气状况，给出带来源的简要报告。", answer_provider=answer)
    weather = [item for item in result.get("evidence", []) if item.get("source_type") == "weather"]
    checks = {
        "clarified_once": len(answers) == 1,
        "history_retained": len(result.get("clarification_history", [])) == 1,
        "question_cleared": result.get("clarification_question") is None,
        "weather_evidence_obtained": bool(weather),
        "cloud_and_direction_in_evidence": bool(weather) and all("云量" in item["content"] and "风向" in item["content"] for item in weather),
        "verified_pass": result.get("report_review", {}).get("verdict") == "pass",
        "brief_length": len(result.get("draft_report") or "") <= BRIEF_REPORT_MAX_CHARACTERS,
        "weather_memory_not_stored": not any(t.get("event") == "memory_stored" for t in result.get("trace", [])),
        "report_delivered": bool(result.get("final_report")),
    }
    prefix = ROOT / "docs/validation/2026-10-20" / ("live_output_" + datetime.now().strftime("%H%M%S"))
    prefix.with_suffix(".md").write_text(format_final_result(result, debug=True), encoding="utf-8")
    payload = {"generated_at": datetime.now().astimezone().isoformat(), "run_id": result.get("run_id"),
               "checks": checks, "all_passed": all(checks.values()), "answers": answers,
               "body_characters": len(result.get("draft_report") or ""), "evidence_count": len(result.get("evidence", [])),
               "usable_evidence_count": len(_numbered_evidence(result)), "tool_calls": result.get("tool_call_count"),
               "research_round": result.get("research_round"), "report_revision_count": result.get("report_revision_count"),
               "draft_report": result.get("draft_report"), "report_review": result.get("report_review"),
               "scope": "一次运行的机制验收，非科学内容人工盲审；不与昨日不同时间天气逐数值比较。"}
    prefix.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
