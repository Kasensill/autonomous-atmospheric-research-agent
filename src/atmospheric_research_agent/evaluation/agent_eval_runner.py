"""小规模 Agent 验收：冻结工具/真实工具共用生产图，隔离记忆并追加保存尝试。

不修改生产提示词或路由。自动协议检查与逐题事实判定分开保存。
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import monotonic
from typing import Any
from unittest.mock import patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from ..agent import evidence as evidence_module, nodes
from ..agent.graph import build_research_graph
from ..agent.memory import recall_project_memory, save_project_memory
from ..agent.settings import GRAPH_RECURSION_LIMIT, MAX_CLARIFICATIONS, MAX_RESEARCH_ROUNDS, MAX_TOOL_CALLS
from ..agent.state import create_initial_state
from ..agent.tools import RESEARCH_TOOLS
from .report_verification_cases import load_report_verification_cases

ROOT = Path(os.getenv("AGENT_EVAL_PROJECT_ROOT", str(Path(__file__).resolve().parents[3]))).resolve()
DEFAULT_CASES = ROOT / "evals/agent_behavior_cases.json"


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def append_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


def message_data(message: Any) -> dict:
    # 不落盘 additional_kwargs：提供商可能把隐藏推理放在其中。
    return {"type": message.type, "content": message.content,
            "tool_calls": getattr(message, "tool_calls", []),
            "tool_call_id": getattr(message, "tool_call_id", None),
            "name": getattr(message, "name", None)}


def controlled_model(factory, recorder, **kwargs):
    model = factory(**kwargs)
    if model.model_name != "deepseek-flash" or str(model.openai_api_base).rstrip("/") != "https://api.deepseek.com":
        raise BudgetStop("当前预算仅核对了官方deepseek-flash，模型或端点变化需重新估价")
    # 重新构造客户端：model_copy修改字段不会重建已初始化的HTTP客户端。
    return nodes.ChatOpenAI(model=model.model_name, api_key=model.openai_api_key,
                            base_url=model.openai_api_base, temperature=model.temperature,
                            extra_body=model.extra_body, callbacks=[recorder],
                            max_retries=0, timeout=60, max_tokens=4096)


class BudgetStop(RuntimeError):
    pass


class ModelRecorder(BaseCallbackHandler):
    """保存可见模型输入输出；关闭SDK重试，usage缺失保留保守预留。

    保守预算按官网Flash峰值美元价×10折算：输入3元/百万，输出12元/百万。
    这不是实际汇率或最终账单；不使用缓存优惠。工具另预留2元。
    """

    raise_error = True

    def __init__(self, directory: Path, limit: float, request_limit: int = 1000):
        self.directory, self.limit, self.request_limit = directory, limit, request_limit
        self.cost = 0.0
        self.calls = 0
        self.active: dict[str, dict] = {}
        self.deadline = float("inf")

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        payload = [[message_data(message) for message in batch] for batch in messages]
        byte_bound = len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 20000
        if byte_bound > 120000:
            raise BudgetStop("单次输入超过评测控制上限，停止而非截断资料")
        reservation = (byte_bound * 3 + 4096 * 12) / 1e6
        if self.calls >= self.request_limit or self.cost + reservation > self.limit or monotonic() > self.deadline:
            raise BudgetStop("达到模型费用预留、请求数或每任务时间上限")
        self.cost += reservation
        self.calls += 1
        self.active[str(run_id)] = {"reservation": reservation, "start": monotonic(), "number": self.calls}
        append_json(self.directory / "model_calls.jsonl", {
            "event": "start", "run_id": str(run_id), "number": self.calls,
            "at": datetime.now(UTC).isoformat(), "messages": payload,
            "reservation_cny": reservation,
        })

    def on_llm_end(self, response, *, run_id, **kwargs):
        active = self.active.pop(str(run_id), {})
        output = []
        input_tokens = output_tokens = 0
        usage_present = False
        for batch in response.generations:
            for generation in batch:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                output.append({"message": message_data(message) if message else {"content": generation.text}, "usage": usage})
                if usage:
                    usage_present = True
                    input_tokens += usage.get("input_tokens", 0)
                    output_tokens += usage.get("output_tokens", 0)
        if usage_present:
            self.cost += (input_tokens * 3 + output_tokens * 12) / 1e6 - active.get("reservation", 0)
        append_json(self.directory / "model_calls.jsonl", {
            "event": "end", "run_id": str(run_id), "number": active.get("number"),
            "duration_seconds": monotonic() - active.get("start", monotonic()),
            "outputs": output, "conservative_total_cny": self.cost,
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        self.active.pop(str(run_id), None)
        append_json(self.directory / "model_calls.jsonl", {
            "event": "error", "run_id": str(run_id), "error_type": type(error).__name__,
            "message": str(error)[:1000], "conservative_total_cny": self.cost,
        })


def prepare(output: Path, cases_path: Path = DEFAULT_CASES, verification_path: Path | None = None) -> dict:
    """冻结题集和本地资料；同一目录不得悄悄更新基线。"""
    target = output / "cases.json"
    if not (output / "live_cases.json").exists():
        save_json(output / "live_cases.json", json.loads((ROOT / "evals/agent_live_cases.json").read_text(encoding="utf-8")))
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    for case in data["cases"]:
        case["local_records"] = []
        for relative in case.get("local_paths", []):
            content = (ROOT / relative).read_text(encoding="utf-8")
            case["local_records"].append({"content": content, "sha256": sha256(content.encode()).hexdigest(),
                                          "source": {"display_name": "知识库：" + Path(relative).stem, "location": relative}})
    save_json(target, data)
    save_json(output / "verification_cases.json", json.loads((verification_path or ROOT / "evals/report_verification_cases.json").read_text(encoding="utf-8")))
    tracked_diff = subprocess.run(["git", "diff", "HEAD"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    save_json(output / "manifest.json", {
        "created_at": datetime.now(UTC).isoformat(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "tracked_diff_sha256": sha256(tracked_diff.encode()).hexdigest(),
        "cases_sha256": sha256(target.read_bytes()).hexdigest(),
        "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "label_status": data["label_status"], "split": "pilot开发/回归，无独立留出结论",
        "execution_controls": {"temperature": 0, "max_tokens": 4096, "max_retries": 0, "timeout_seconds": 60,
                               "per_task_cooperative_deadline_seconds": 600, "tool_reserve_cny": 2},
        "price_reference": "https://api-docs.deepseek.com/quick_start/pricing/",
        "pricing_assumption": "Flash峰值USD输入0.3/输出1.2每百万，×10作保守人民币预算，非真实账单",
    })
    baseline_files = [*sorted((ROOT / "src/atmospheric_research_agent/agent").rglob("*.py")),
                      *sorted((ROOT / "data/knowledge_base").rglob("*.md"))]
    save_json(output / "baseline_files.json", {
        str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest() for p in baseline_files
    })
    return data


def frozen_tools(case: dict, clock: str) -> list:
    url = "https://fixtures.example.invalid/" + case["id"]

    def local(query: str):
        return {"status": "success", "results": deepcopy(case.get("local_records", [])), "query": query}

    def search(query: str):
        web = case.get("web")
        pages = case.get("pages", [web] if web else [])
        candidates = [{"title": page["title"], "url": url + (f"/{i}" if case.get("pages") else ""),
                       "summary": page.get("summary", page["title"])} for i, page in enumerate(pages)]
        return {"status": "success", "query": query, "results": candidates, "result_count": len(candidates)}

    def read(url: str):
        web = case.get("web")
        if case.get("pages"):
            web = next((page for i, page in enumerate(case["pages"])
                        if url == "https://fixtures.example.invalid/" + case["id"] + f"/{i}"), None)
        elif url != "https://fixtures.example.invalid/" + case["id"]:
            web = None
        if not web or web.get("read_error"):
            return {"status": "error", "message": (web or {}).get("read_error", "冻结资料无该URL"), "content": ""}
        return {"status": "success", "content": web["content"],
                "source": {"display_name": web["title"], "location": url, "organization": "合成评测资料"}}

    def weather(location: str):
        if not case.get("weather") or not any(term in location.lower() for term in ("北京", "beijing")):
            return {"status": "error", "message": "冻结资料无该地点当前天气"}
        valid_at = case.get("weather_valid_at", clock)
        return {"status": "success", "observed_at": valid_at, "valid_at": valid_at, "fetched_at": clock,
                "evidence_text": f"【评测回放快照，非真实天气】北京，{valid_at}，Open-Meteo模型当前条件：气温12°C，相对湿度55%，风速8 km/h，降水0毫米。不是地面站实测。不含气候基准、空气质量和预警信息。",
                "source": {"display_name": "合成Open-Meteo模型条件", "location": "https://fixtures.example.invalid/weather/beijing", "organization": "合成测试"}}

    functions = {"search_local_knowledge": local, "search_web": search, "read_source": read, "get_current_weather": weather}
    # 沿用生产工具名称、描述和参数Schema，避免测试工具提示改变选择行为。
    return [StructuredTool.from_function(functions[t.name], name=t.name, description=t.description, args_schema=t.args_schema) for t in RESEARCH_TOOLS]


def objective_checks(state: dict, case: dict) -> dict:
    checks = {
        "allowed_ending": ("paused" if state.get("__interrupt__") else "completed") in case.get("allowed_endings", ["completed"]),
        "report_when_completed": bool(state.get("final_report")) or bool(state.get("__interrupt__")),
        "research_budget": state.get("research_round", 0) <= MAX_RESEARCH_ROUNDS,
        "tool_budget": state.get("tool_call_count", 0) <= MAX_TOOL_CALLS,
        "clarification_budget": len(state.get("clarification_history", [])) <= MAX_CLARIFICATIONS,
        "no_search_evidence": all(e.get("tool_name") != "search_web" for e in state.get("evidence", [])),
    }
    if any(t.get("event") == "report_finalized" for t in state.get("trace", [])):
        checks["verification_pass"] = state.get("report_review", {}).get("verdict") == "pass"
        checks["citations_exist"] = not nodes.check_citation_integrity(state)
    messages = state.get("messages", [])
    requested = [call for m in messages for call in getattr(m, "tool_calls", [])]
    answers = [getattr(m, "tool_call_id", None) for m in messages if m.type == "tool"]
    checks["tool_protocol"] = all(answers.count(call["id"]) == 1 for call in requested)
    if case["id"] == "Q07" or case.get("clarify_first"):
        checks["clarify_before_research"] = bool(requested) and requested[0]["name"] == "ask_user"
    executed = {m.name for m in messages if m.type == "tool" and not m.additional_kwargs.get("protocol_only")}
    for name in case.get("required_tools", []):
        checks["executed_" + name] = name in executed
    return checks


def write_card(directory: Path, case: dict, state: dict, result: dict) -> None:
    lines = [f"# {result['attempt_id']}", "", case["question"], "", f"运行状态：{result['status']}；语义判定：待本轮逐项证据检查。", "",
             "## 预定必需行为", "", *["- " + item for item in case["required"]], "", "## 禁止行为", "",
             *["- " + item for item in case["forbidden"]], "", "## 自动检查", "",
             *[f"- {name}：{'Pass' if passed else 'Fail'}" for name, passed in result.get("checks", {}).items()],
             "", "## 追问与回答", "", *[f"- 问：{item['question']}；答：{item['answer']}" for item in state.get("clarification_history", [])],
             "", "## 研究工具轨迹", ""]
    for message in state.get("messages", []):
        for call in getattr(message, "tool_calls", []):
            lines.append(f"- {call['name']}：{json.dumps(call['args'], ensure_ascii=False)}")
    lines += ["", "## 交付结果", "", state.get("final_report") or result.get("error") or "没有交付报告。", "", "## 正式证据", ""]
    for i, item in enumerate(state.get("evidence", []), 1):
        lines += [f"### E{i}：{item.get('source_display_name')}", "", item.get("source_location", ""), "", item.get("content", ""), ""]
    lines += ["## 证据判定", "", "- 判定：见本批次汇总中的逐题判定，用户可抽验。", "- 原始运行记录不预先填通过。", "", "完整可见模型输入输出见 model_calls.jsonl；各版草稿及核验见 checkpoints.json。", ""]
    (directory / "card.md").write_text("\n".join(lines), encoding="utf-8")


def run_case(case: dict, mode: str, attempt_id: str, output: Path, clock: str, recorder: ModelRecorder) -> dict:
    directory = output / attempt_id
    directory.mkdir(exist_ok=False)
    recorder.directory = directory
    recorder.deadline = monotonic() + 600
    started, calls_before, cost_before = monotonic(), recorder.calls, recorder.cost
    graph = None
    state = create_initial_state(case["question"])
    config = {"recursion_limit": GRAPH_RECURSION_LIMIT, "configurable": {"thread_id": state["run_id"]}}
    result = {"attempt_id": attempt_id, "case_id": case["id"], "mode": mode, "status": "STARTED",
              "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
    append_json(output / "runs.jsonl", result)
    memory_path = directory / "isolated_memory.json"
    original_factory = nodes._create_base_model
    frozen_now = datetime.fromisoformat(clock)

    def model_factory(**kwargs):
        return controlled_model(original_factory, recorder, **kwargs)

    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(nodes, "_create_base_model", model_factory))
            stack.enter_context(patch.object(nodes, "recall_project_memory", lambda question: recall_project_memory(question, store_path=memory_path)))
            stack.enter_context(patch.object(nodes, "save_project_memory", lambda record: save_project_memory(record, store_path=memory_path)))
            if mode == "frozen":
                stack.enter_context(patch.object(nodes, "RESEARCH_TOOLS", frozen_tools(case, clock)))
                original_stale = evidence_module.is_stale_weather
                stale = lambda record, **kw: original_stale(record, now=frozen_now)
                stack.enter_context(patch.object(evidence_module, "is_stale_weather", stale))
                stack.enter_context(patch.object(nodes, "is_stale_weather", stale))
            graph = build_research_graph(checkpointer=InMemorySaver())
            state = graph.invoke(state, config)
            answers = iter(case.get("answers", []))
            while state.get("__interrupt__"):
                answer = next(answers, None)
                if answer is None:
                    break
                state = graph.invoke(Command(resume=answer), config)
            result["checks"] = objective_checks(state, case)
            result["status"] = "COMPLETED" if not state.get("__interrupt__") else "PAUSED"
    except Exception as error:
        result.update(status="ERROR", error_type=type(error).__name__, error=str(error)[:2000])
        if graph:
            state = graph.get_state(config).values or state
    finally:
        result.update(duration_seconds=round(monotonic() - started, 3), model_requests=recorder.calls - calls_before,
                      conservative_cny=round(recorder.cost - cost_before, 6), semantic_verdict="AWAITING_EVIDENCE_REVIEW",
                      research_round=state.get("research_round", 0), tool_count=state.get("tool_call_count", 0),
                      revisions=state.get("report_revision_count", 0), evidence_count=len(state.get("evidence", [])))
        save_json(directory / "state.json", {**state, "messages": [message_data(m) for m in state.get("messages", [])]})
        if graph:
            checkpoints = []
            for snapshot in reversed(list(graph.get_state_history(config))):
                checkpoints.append({"step": (snapshot.metadata or {}).get("step"), "draft": snapshot.values.get("draft_report"),
                                    "review": snapshot.values.get("report_review"), "next": snapshot.next})
            save_json(directory / "checkpoints.json", checkpoints)
        save_json(directory / "result.json", result)
        write_card(directory, case, state, result)
        append_json(output / "runs.jsonl", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "frozen", "live", "judge"], default="prepare")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--verification-cases", type=Path)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--budget-cny", type=float, default=None, help="可选费用控制；默认仅记录用量，不设置费用门槛")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "docs/validation"):
        parser.error("输出必须位于项目docs/validation内")
    output.mkdir(parents=True, exist_ok=True)
    data = prepare(output, args.cases, args.verification_cases)
    if args.phase == "prepare":
        print(f"冻结{len(data['cases'])}题；未调用API。", flush=True)
        return
    previous = [json.loads(line) for line in (output / "runs.jsonl").read_text(encoding="utf-8").splitlines()] if (output / "runs.jsonl").exists() else []
    recorder = ModelRecorder(output, max(0, args.budget_cny - 2) if args.budget_cny is not None else float("inf"))
    finished = [item for item in previous if item["status"] != "STARTED"]
    recorder.cost = sum(item.get("conservative_cny", 0) for item in finished)
    recorder.calls = sum(item.get("model_requests", 0) for item in finished)
    if args.phase == "judge":
        run_judges(output, recorder, args.verification_cases)
        return
    if args.phase == "live":
        data["cases"] = json.loads((output / "live_cases.json").read_text(encoding="utf-8"))["cases"]
    cases = [case for case in data["cases"] if not args.ids or case["id"] in args.ids]
    if args.ids and set(args.ids) - {case["id"] for case in cases}:
        parser.error("存在未知ID")
    consecutive_errors = 0
    for case in cases:
        for _ in range(args.repeats):
            ordinal = 1 + len(list(output.glob(f"{args.phase}-{case['id']}-*")))
            attempt = f"{args.phase}-{case['id']}-{ordinal:02}"
            result = run_case(case, args.phase, attempt, output, data["clock"], recorder)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            consecutive_errors = consecutive_errors + 1 if result["status"] == "ERROR" else 0
            if consecutive_errors >= 3 or result.get("error_type") == "BudgetStop":
                return


def run_judges(output: Path, recorder: ModelRecorder, cases_path: Path | None = None):
    """保留原始模型审查和程序合并审查，避免把复合结果误称纯模型能力。"""
    original_factory = nodes._create_base_model
    results = []
    errors = 0
    for case in load_report_verification_cases(cases_path or output / "verification_cases.json"):
        directory = output / ("judge-" + case.case_id)
        directory.mkdir(exist_ok=False)
        recorder.directory, recorder.deadline = directory, monotonic() + 600
        state = create_initial_state(case.question)
        state.update(evidence=deepcopy(list(case.evidence)), draft_report=case.draft_report)
        for evidence in state["evidence"]:
            if evidence.get("source_type") == "weather":
                evidence["valid_at"] = datetime.now(UTC).isoformat()
        before, cost_before = recorder.calls, recorder.cost
        result = {"attempt_id": directory.name, "case_id": case.case_id, "expected": case.expected_verdict,
                  "runner_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}
        try:
            model = controlled_model(original_factory, recorder)
            structured = model.with_structured_output(nodes.ReportReviewOutput, method="json_mode")
            captured = {}

            class Capture:
                def invoke(self, messages):
                    answer = structured.invoke(messages)
                    captured.update(answer.model_dump())
                    return answer

            update = nodes.verify_report_with_model(state, Capture())
            result.update(status="COMPLETED", raw_model_review=captured, composite_review=update["report_review"],
                          matched=update["report_review"]["verdict"] == case.expected_verdict,
                          program_citation_issues=nodes.check_citation_integrity(state))
            errors = 0
        except Exception as error:
            result.update(status="ERROR", error_type=type(error).__name__, error=str(error)[:2000])
            errors += 1
        result.update(model_requests=recorder.calls - before, conservative_cny=recorder.cost - cost_before)
        save_json(directory / "result.json", result)
        append_json(output / "runs.jsonl", result)
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"raw_model_review", "composite_review"}}, ensure_ascii=False), flush=True)
        if errors >= 3 or result.get("error_type") == "BudgetStop":
            break
    save_json(output / "judge_results.json", results)


if __name__ == "__main__":
    main()
