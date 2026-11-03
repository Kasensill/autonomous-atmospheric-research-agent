"""历史查询在真实 SQLite 和研究图上的离线验证。"""
import json
from uuid import uuid4

from langchain_core.messages import AIMessage

from asgi_helpers import MemoryClient
from test_service_api import wait
from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.history import archived_history, select_evidence
from atmospheric_research_agent.service.api import create_app
from atmospheric_research_agent.service.manager import RunManager, RunRecord


def model_with(monkeypatch, *decisions):
    responses = iter(decisions)

    class Model:
        def invoke(self, messages):
            return AIMessage(content=json.dumps(next(responses), ensure_ascii=False))

    monkeypatch.setattr(nodes, "create_research_model", Model)
    monkeypatch.setattr(nodes, "recall_project_memory", lambda question: [])


def seed(manager, *, status="completed", mode="limited_report", question="副高采样研究"):
    run_id = str(uuid4())
    record = RunRecord(run_id, question, run_id, status=status, delivery_type=mode,
                       conversation_id=run_id)
    record.state = {"final_report": "上次使用了甲每小时一次、乙每三小时一次的设定。",
                    "evidence": [{"evidence_id": "original-evidence", "source_type": "web",
                                  "source_location": "https://example.org/observation",
                                  "content": "甲每小时采样一次，乙每三小时采样一次。"}]}
    manager._persist(record)
    return record


def request(purpose, target="latest", **kwargs):
    return {"decision": "query_history",
            "history_request": {"purpose": purpose, "target": target, **kwargs}}


def create(client, question, key="question"):
    response = client.post("/runs", json={"question": question}, headers={"Idempotency-Key": key})
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def test_history_answer_bypasses_research_and_memory(tmp_path, monkeypatch):
    model_with(monkeypatch, request("answer_history", include_evidence=True))
    manager = RunManager(db_path=tmp_path / "runs.sqlite3")
    old = seed(manager)
    with MemoryClient(create_app(manager)) as client:
        run_id = create(client, "上次报告的结论和依据是什么？")
        view = wait(client, run_id, "completed")
        assert view["delivery_type"] == "history_answer"
        assert view["research_round"] == view["tool_call_count"] == 0
        result = client.get(f"/runs/{run_id}/result").json()["state"]
        assert result["evidence"] == []
        assert old.run_id in result["final_report"] and "档案条目 #1" in result["final_report"]
        trace = client.get(f"/runs/{run_id}/trace").json()["events"]
        assert not {"assess_evidence", "store_memory", "tools"} & {event["node"] for event in trace}


def test_ambiguous_archive_asks_then_reads_explicit_choice(tmp_path, monkeypatch):
    manager = RunManager(db_path=tmp_path / "runs.sqlite3")
    first, other = seed(manager), seed(manager)
    model_with(monkeypatch, request("answer_history", "topic", value="副高"),
               request("answer_history", "run_id", value=first.run_id))
    with MemoryClient(create_app(manager)) as client:
        run_id = create(client, "想看之前一份副高报告")
        waiting = wait(client, run_id, "waiting_user")
        assert first.run_id in waiting["clarification"]["question"]
        assert other.run_id in waiting["clarification"]["question"]
        response = client.post(f"/runs/{run_id}/answers", json={
            "question_id": waiting["clarification"]["question_id"], "answer": first.run_id})
        assert response.status_code == 202
        wait(client, run_id, "completed")
        result = client.get(f"/runs/{run_id}/result").json()["state"]
        assert first.run_id in result["final_report"] and other.run_id not in result["final_report"]
        assert result["research_round"] == 0


def test_selected_limited_report_evidence_enters_normal_review(tmp_path, monkeypatch):
    import importlib
    graph = importlib.import_module("atmospheric_research_agent.agent.graph")
    manager = RunManager(db_path=tmp_path / "runs.sqlite3")
    old = seed(manager)
    model_with(monkeypatch, request("support_research"),
               {"decision": "use_history", "evidence_indices": [99, 1, 1]})

    def assess(state):
        assert len(state["evidence"]) == 1
        assert state["evidence"][0]["historical_run_id"] == old.run_id
        return {"assessment": {"next_step": "full_report"}}

    monkeypatch.setattr(graph, "assess_evidence_node", assess)
    monkeypatch.setattr(graph, "generate_draft_node",
                        lambda state: {"draft_report": "采样次数比为 3:1 [E1]。", "report_mode": "full"})
    monkeypatch.setattr(graph, "verify_report_node", lambda state: {"report_review": {"verdict": "pass"}})
    monkeypatch.setattr(graph, "store_memory_node", lambda state: {})
    with MemoryClient(create_app(manager)) as client:
        run_id = create(client, "用上次的数据算采样次数比")
        view = wait(client, run_id, "completed")
        assert view["delivery_type"] == "full_report" and view["tool_call_count"] == 0
        result = client.get(f"/runs/{run_id}/result").json()["state"]
        assert result["evidence"][0]["historical_evidence_index"] == 1
        assert "3:1" in result["final_report"]


def test_failed_archive_is_only_lead_and_mismatched_request_cannot_import():
    run_id = str(uuid4())
    request_data = request("support_research", "run_id", value=run_id)["history_request"]
    state = {"question": run_id, "history_request": request_data}
    row = {"run_id": run_id, "status": "failed", "delivery_type": None, "question": "旧研究",
           "state": {"evidence": [{"evidence_id": "old", "content": "片段", "source_location": "https://example.org",
                                  "source_type": "web"}]}}
    result = archived_history(request_data, state, [row], parent_run_id=None)
    state["history_result"] = {**result, "request": request_data}
    assert result["status"] == "matched" and not result["evidence"][0]["eligible"]
    assert select_evidence(state, [1]) == []
    state["history_request"] = {**request_data, "purpose": "answer_history"}
    assert select_evidence(state, [1]) == []


def test_history_miss_finishes_without_empty_research_loop(tmp_path, monkeypatch):
    model_with(monkeypatch, request("answer_history", "topic", value="无记录"))
    with MemoryClient(create_app(db_path=tmp_path / "runs.sqlite3")) as client:
        run_id = create(client, "以前是否研究过无记录主题？")
        view = wait(client, run_id, "completed")
        assert view["delivery_type"] == "history_answer" and view["research_round"] == 0
        assert "没有找到" in client.get(f"/runs/{run_id}/result").json()["state"]["final_report"]


def test_memory_fallback_is_history_only(tmp_path, monkeypatch):
    memory = {"memory_id": "memory-card", "status": "verified_report",
              "summary": "副高边缘当时被解释为水汽输送路径。", "question": "副高边缘"}
    model_with(monkeypatch, request("answer_history", "topic", value="副高边缘"),
               {"decision": "stop", "stop_reason": "未取得本轮事件资料"})
    monkeypatch.setattr(nodes, "recall_project_memory", lambda question: [memory])
    with MemoryClient(create_app(db_path=tmp_path / "runs.sqlite3")) as client:
        first = create(client, "以前怎么解释副高边缘？", "history")
        assert wait(client, first, "completed")["delivery_type"] == "history_answer"
        historical = client.get(f"/runs/{first}/result").json()["state"]
        assert "[Memory]" in historical["final_report"] and not historical["evidence"]
        second = create(client, "上次的解释能证明今天这场雨的成因吗？", "research")
        assert wait(client, second, "completed")["delivery_type"] == "failure_notice"
        research = client.get(f"/runs/{second}/result").json()["state"]
        assert not research["evidence"]
        trace = client.get(f"/runs/{second}/trace").json()["events"]
        assert any(event["event"] == "research_stopped_without_evidence" for event in trace)


def test_scope_blocks_invented_uuid_and_user_material_is_only_candidate(tmp_path, monkeypatch):
    manager = RunManager(db_path=tmp_path / "runs.sqlite3")
    old = seed(manager)
    old.state["evidence"][0].update(source_type="user_material", role="analysis")
    manager._persist(old)
    model_with(monkeypatch, request("answer_history", "run_id", value=old.run_id),
               request("support_research", "run_id", value=old.run_id),
               {"decision": "use_history", "evidence_indices": [1]},
               {"decision": "stop", "stop_reason": "未获得独立证据"})
    with MemoryClient(create_app(manager)) as client:
        first = create(client, "查询不存在的指定任务", "history")
        wait(client, first, "completed")
        historical = client.get(f"/runs/{first}/result").json()["state"]
        assert "没有找到" in historical["final_report"]
        second = create(client, f"用任务 {old.run_id} 的资料核实新事实", "research")
        assert wait(client, second, "completed")["delivery_type"] == "failure_notice"
        research = client.get(f"/runs/{second}/result").json()["state"]
        assert not research["evidence"]
