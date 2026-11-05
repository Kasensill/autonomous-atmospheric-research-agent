from atmospheric_research_agent.agent import decision as _split_decision
from atmospheric_research_agent.agent import memory_nodes as _split_memory_nodes
from atmospheric_research_agent.agent import model_factory as _split_model_factory
"""历史查询在真实 SQLite 和研究图上的离线验证。"""
import json
from uuid import uuid4

from langchain_core.messages import AIMessage

from asgi_helpers import MemoryClient
from test_service_api import wait
from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.history import history_payload, lookup_history_node, select_evidence
import pytest
from atmospheric_research_agent.service.api import create_app
from atmospheric_research_agent.service.manager import RunManager, RunRecord


def model_with(monkeypatch, *decisions):
    responses = iter(decisions)

    class Model:
        def invoke(self, messages):
            return AIMessage(content=json.dumps(next(responses), ensure_ascii=False))

    monkeypatch.setattr(_split_decision, "create_research_model", Model)
    monkeypatch.setattr("atmospheric_research_agent.agent.history._targeted_answer",
                        lambda question, record: "旧记录显示甲每小时一次、乙每三小时一次。")
    monkeypatch.setattr(_split_memory_nodes, "recall_project_memory", lambda question: [])


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
        assert old.run_id in result["final_report"] and "乙每三小时一次" in result["final_report"]
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
    result = history_payload(request_data, state, row)
    state["history_result"] = {**result, "request": request_data}
    assert not result["evidence"][0]["eligible"]
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
    monkeypatch.setattr(_split_memory_nodes, "recall_project_memory", lambda question: [memory])
    with MemoryClient(create_app(db_path=tmp_path / "runs.sqlite3")) as client:
        first = create(client, "以前怎么解释副高边缘？", "history")
        assert wait(client, first, "completed")["delivery_type"] == "history_answer"
        historical = client.get(f"/runs/{first}/result").json()["state"]
        assert "没有找到" in historical["final_report"] and not historical["evidence"]
        from types import SimpleNamespace
        cli_request = request("answer_history", "topic", value="副高边缘")["history_request"]
        cli = lookup_history_node({"question": "以前怎么解释副高边缘？",
                                   "history_request": cli_request, "recalled_memories": [memory]},
                                  SimpleNamespace(context=None))
        assert cli["history_result"]["memory"]["memory_id"] == "memory-card"
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


def test_recent_report_reference_keeps_model_topic(tmp_path, monkeypatch):
    manager = RunManager(db_path=tmp_path / "runs.sqlite3")
    seed(manager, question="较早的郑州暴雨研究")
    previous = seed(manager, mode="full_report", question="2012 年 7 月 21 日北京暴雨是怎么形成的？")
    seed(manager, question="更新的无关任务")
    model_with(monkeypatch, request("answer_history", "topic",
                                    value="2012年7月21日北京暴雨", include_evidence=True))
    with MemoryClient(create_app(manager)) as client:
        run_id = create(client, "刚才那份关于北京 7·21 暴雨成因的报告，结论和依据是什么？")
        view = wait(client, run_id, "completed")
        assert view["delivery_type"] == "history_answer"
        assert view["research_round"] == view["tool_call_count"] == 0
        result = client.get(f"/runs/{run_id}/result").json()["state"]
        assert previous.run_id in result["final_report"]
        assert "乙每三小时一次" in result["final_report"]
        trace = client.get(f"/runs/{run_id}/trace").json()["events"]
        assert any(t["event"] == "history_lookup" and "topic → topic" in t["detail"] for t in trace)


def test_conclusions_and_sources_answer_omits_report_body(tmp_path, monkeypatch):
    manager = RunManager(db_path=tmp_path / "runs.sqlite3")
    old = seed(manager, mode="full_report")
    old.state["final_report"] = (
        "# 历史报告\n\n## 1. 结论\n\n- 结论甲 [E1]。\n- 结论乙 [E2]。\n\n"
        "## 2. 分析正文\n\n" + "不应输出的分析正文。" * 700 +
        "\n\n## 资料来源\n\n"
        "- [E1] 文献甲（[原文](https://example.org/a)）\n"
        "- [E2] 文献甲另一页（[原文](https://example.org/a)）\n"
        "- [E3] 文献乙（[原文](https://example.org/b)）"
    )
    manager._persist(old)
    model_with(monkeypatch, request("answer_history", include_evidence=True))
    with MemoryClient(create_app(manager)) as client:
        run_id = create(client, "刚才那份报告结论有几条，分别是什么，引用了哪几篇文献？")
        wait(client, run_id, "completed")
        report = client.get(f"/runs/{run_id}/result").json()["state"]["final_report"]
        assert "**2 条**" in report and "结论甲" in report and "结论乙" in report
        assert "**2 个来源**" in report and "E1、E2" in report and "文献乙" in report
        assert "不应输出的分析正文" not in report and "档案条目" not in report


def locate(manager, question="刚才", *, target="latest", value="", purpose="answer_history"):
    from atmospheric_research_agent.agent.history import HistoryRequest, effective_history_request
    run_id = str(uuid4())
    current = RunRecord(run_id, question, run_id)
    manager._persist(current)
    req = HistoryRequest.model_validate(request(purpose, target, value=value)["history_request"]).model_dump()
    state = {"question": question}
    return manager._history_context(current).lookup(effective_history_request(req, state), state)


def test_only_history_answer_is_searchable_and_persisted(tmp_path, monkeypatch):
    manager = RunManager(db_path=tmp_path / "runs.sqlite3")
    old = seed(manager, mode="history_answer")
    model_with(monkeypatch, request("answer_history"))
    with MemoryClient(create_app(manager)) as client:
        current = create(client, "刚才那条回答是什么？")
        wait(client, current, "completed")
        saved = manager._store.get(current)
        assert saved["history_source_run_id"] == old.run_id
        assert saved["history_lookup"]["selected_run_id"] == old.run_id
        trace = next(t for t in saved["trace"] if t["event"] == "history_lookup")
        assert trace["history_lookup"] == saved["history_lookup"]
        assert trace["origin_status"] == "missing_origin"


@pytest.mark.parametrize("link_name", ["history_source_run_id", "parent_run_id"])
def test_history_answer_backtracks_to_report_without_changing_direct_hit(tmp_path, link_name):
    with MemoryClient(create_app(manager := RunManager(db_path=tmp_path / "runs.sqlite3"))):
        origin = seed(manager, mode="full_report")
        answer = seed(manager, mode="history_answer")
        setattr(answer, link_name, origin.run_id)
        manager._persist(answer)
        result = locate(manager, "刚才那条回答")
        assert result["history_lookup"]["selected_run_id"] == answer.run_id
        assert result["origin_run_id"] == origin.run_id
        assert result["origin_report"] == origin.state["final_report"]
        assert result["origin_evidence"] == origin.state["evidence"]


def test_history_origin_cycle_stops_and_keeps_match(tmp_path):
    with MemoryClient(create_app(manager := RunManager(db_path=tmp_path / "runs.sqlite3"))):
        a, b = seed(manager, mode="history_answer"), seed(manager, mode="history_answer")
        a.history_source_run_id, b.history_source_run_id = b.run_id, a.run_id
        manager._persist(a)
        manager._persist(b)
        result = locate(manager, "刚才那条回答")
        assert result["status"] == "matched" and result["origin_status"] == "cycle"
        assert result["run_id"] == b.run_id


def test_latest_orders_by_finished_at_with_utc_and_updated_at_fallback(tmp_path):
    with MemoryClient(create_app(manager := RunManager(db_path=tmp_path / "runs.sqlite3"))):
        earlier_created = seed(manager)
        later_created = seed(manager)
        fallback = seed(manager)
        earlier_created.state["finished_at"] = "2026-11-03T09:45:00+08:00"
        later_created.state["finished_at"] = "2026-11-03T01:30:00"
        later_created.updated_at = "2026-11-03T23:30:00+00:00"
        fallback.updated_at = "2026-11-03T01:40:00Z"
        for row in (earlier_created, later_created, fallback):
            manager._persist(row)
        assert locate(manager)["run_id"] == earlier_created.run_id
        earlier_created.state.pop("finished_at")
        earlier_created.updated_at = "invalid"
        earlier_created.state["started_at"] = "2026-11-03T01:20:00"
        manager._persist(earlier_created)
        assert locate(manager)["run_id"] == fallback.run_id


def test_normalized_topic_and_explicit_record_type(tmp_path):
    with MemoryClient(create_app(manager := RunManager(db_path=tmp_path / "runs.sqlite3"))):
        old = seed(manager, mode="full_report", question="北京｜WRF 试验")
        answer = seed(manager, mode="history_answer", question="北京 WRF 试验")
        assert locate(manager)["run_id"] == answer.run_id
        assert locate(manager, "刚才那份关于北京 Wrf 试验的报告", target="topic",
                      value="北 京，wrf！试验")["run_id"] == old.run_id
        assert locate(manager, purpose="support_research")["run_id"] == old.run_id


def test_no_completed_records_and_scan_limit_reasons(tmp_path, monkeypatch):
    monkeypatch.setattr("atmospheric_research_agent.service.manager.HISTORY_ARCHIVE_SCAN_LIMIT", 2)
    with MemoryClient(create_app(manager := RunManager(db_path=tmp_path / "runs.sqlite3"))):
        seed(manager, status="interrupted", mode=None)
        assert locate(manager)["history_lookup"]["not_found_reason"] == "no_candidates"
        seed(manager, mode="failure_notice")
        assert locate(manager)["history_lookup"]["not_found_reason"] == "filtered_out"
        seed(manager, question="窗口之外的甲主题")
        seed(manager, question="无关乙")
        seed(manager, question="无关丙")
        result = locate(manager, "刚才关于甲主题的内容", target="topic", value="甲主题")
        summary = result["history_lookup"]
        assert result["status"] == "not_found" and summary["not_found_reason"] == "truncated"
        assert summary["truncated"] and summary["candidate_count_after_base_filter"] == 3
        assert summary["candidate_count_after_topic"] == 0


def test_literal_topic_does_not_expand_to_synonyms(tmp_path):
    with MemoryClient(create_app(manager := RunManager(db_path=tmp_path / "runs.sqlite3"))):
        seed(manager, question="北京特大暴雨")
        result = locate(manager, "刚才关于北京暴雨的内容", target="topic", value="北京暴雨")
        assert result["status"] == "not_found"
        assert result["history_lookup"]["not_found_reason"] == "no_match"


@pytest.mark.xfail(strict=True, reason="已知限制：归一化子串匹配不跨过插入的“特大”")
def test_semantic_topic_would_find_beijing_heavy_rain(tmp_path):
    with MemoryClient(create_app(manager := RunManager(db_path=tmp_path / "runs.sqlite3"))):
        old = seed(manager, question="北京特大暴雨")
        result = locate(manager, "刚才关于北京暴雨的内容", target="topic", value="北京暴雨")
        assert result["status"] == "matched" and result["run_id"] == old.run_id


def test_missing_origin_targeted_answer_uses_only_selected_answer(monkeypatch):
    from atmospheric_research_agent.agent.history import HistoricalAnswer, answer_history_node

    class Model:
        def with_structured_output(self, schema, **kwargs):
            assert schema is HistoricalAnswer
            return self

        def invoke(self, messages):
            payload = json.loads(messages[-1].content)
            assert payload["record"] == "上次回答说乙站为 10 毫米。长段落不应照贴。"
            return HistoricalAnswer(answer="10 毫米", quotes=["乙站为 10 毫米"])

    monkeypatch.setattr(_split_model_factory, "_create_base_model", Model)
    req = request("answer_history")["history_request"]
    result = answer_history_node({
        "question": "上次回答里乙站是多少？只告诉我数字和单位。",
        "history_request": req,
        "history_result": {"request": req, "status": "matched", "run_id": "old-answer",
                           "delivery_type": "history_answer", "origin_status": "missing_origin",
                           "report": "上次回答说乙站为 10 毫米。长段落不应照贴。"},
    })
    assert result["final_report"] == "原报告已不存在，以下内容来自历史回答存档。\n\n10 毫米"


def test_targeted_answer_rejects_unbacked_number_or_citation(monkeypatch):
    from atmospheric_research_agent.agent.history import HistoricalAnswer, _targeted_answer

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return HistoricalAnswer(answer="10 毫米 [E2]", quotes=["乙站为 110 毫米 [E1]"])

    monkeypatch.setattr(_split_model_factory, "_create_base_model", Model)
    result = _targeted_answer("乙站是多少？", "乙站为 110 毫米 [E1]。")
    assert result == "这条历史记录没有足够信息回答这个问题。"


@pytest.mark.parametrize(("answer", "quote", "accepted"), [
    ("3小时", "乙站每三小时采样一次", True),
    ("三小时", "乙站每3小时采样一次", True),
    ("13小时", "乙站每十三小时采样一次", True),
    ("-3℃", "温度为负三℃", True),
    ("4小时", "乙站每三小时采样一次", False),
    ("四小时", "乙站每三小时采样一次", False),
    ("3小时", "乙站每十三小时采样一次", False),
    ("3分钟", "乙站每三小时采样一次", False),
    ("3毫米/小时", "降水量为3毫米", False),
    ("三百小时", "乙站每三小时采样一次", False),
])
def test_targeted_answer_quantity_requires_equivalent_quoted_value(monkeypatch, answer, quote, accepted):
    from atmospheric_research_agent.agent.history import HistoricalAnswer, _targeted_answer

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return HistoricalAnswer(answer=answer, quotes=[quote])

    monkeypatch.setattr(_split_model_factory, "_create_base_model", Model)
    result = _targeted_answer("是多少？", quote)
    assert (result == answer) is accepted


def test_targeted_answer_does_not_join_separate_quotes_into_quantity(monkeypatch):
    from atmospheric_research_agent.agent.history import HistoricalAnswer, _targeted_answer

    class Model:
        def with_structured_output(self, schema, **kwargs):
            return self

        def invoke(self, messages):
            return HistoricalAnswer(answer="3小时", quotes=["3", "小时"])

    monkeypatch.setattr(_split_model_factory, "_create_base_model", Model)
    assert _targeted_answer("多久？", "次数为3，时间单位为小时。") == "这条历史记录没有足够信息回答这个问题。"
