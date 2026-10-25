"""评测装置的关键契约：冻结不联网、真实图运行、预算拦截、产物可复核。"""

import json

import pytest
from langchain_core.messages import AIMessage

from atmospheric_research_agent.agent import graph as graph_module
from atmospheric_research_agent.evaluation import agent_eval_runner as runner


def test_frozen_tools_keep_production_schema_and_do_not_read_arbitrary_urls():
    case = {"id": "test", "web": {"title": "合成", "content": "正文"}, "local_records": []}
    tools = {tool.name: tool for tool in runner.frozen_tools(case, "2026-10-25T04:00:00+00:00")}
    assert set(tools) == {tool.name for tool in runner.RESEARCH_TOOLS}
    candidate = tools["search_web"].invoke({"query": "任何问题"})["results"][0]["url"]
    assert tools["read_source"].invoke({"url": candidate})["content"] == "正文"
    assert tools["read_source"].invoke({"url": "http://127.0.0.1/"})["status"] == "error"


def test_budget_refuses_request_before_it_starts(tmp_path):
    recorder = runner.ModelRecorder(tmp_path, 0)
    with pytest.raises(runner.BudgetStop):
        recorder.on_chat_model_start({}, [[AIMessage(content="test")]], run_id="one")
    assert recorder.calls == 0
    assert not (tmp_path / "model_calls.jsonl").exists()


def test_controlled_model_rebuilds_http_client_with_retry_and_timeout_limits(tmp_path):
    original = runner.nodes.ChatOpenAI(model="deepseek-flash", api_key="test-placeholder",
                                      base_url="https://api.deepseek.com", max_retries=2)
    controlled = runner.controlled_model(lambda: original, runner.ModelRecorder(tmp_path, 1))
    assert controlled.root_client is not original.root_client
    assert controlled.root_client.max_retries == 0
    assert controlled.root_client.timeout == 60
    assert controlled.max_tokens == 4096


def test_case_ids_and_repeated_families_are_fixed():
    data = json.loads(runner.DEFAULT_CASES.read_text(encoding="utf-8"))
    assert len({case["id"] for case in data["cases"]}) == 10
    assert sum(case["repeat"] for case in data["cases"]) == 3
    assert {case["category"] for case in data["cases"]} == {"明确可答", "缺关键信息", "证据不足", "冻结环境无结果", "时效性强"}
    for case in data["cases"]:
        assert case["required"] and case["forbidden"]


def test_real_graph_with_mock_model_writes_human_review_card(monkeypatch, tmp_path):
    monkeypatch.setattr(graph_module, "agent_node", lambda state: {"messages": [AIMessage(content='{"decision":"assess"}')]})
    case = {"id": "test", "question": "没有证据", "answers": [], "required": ["说明缺口"], "forbidden": ["伪造"]}
    recorder = runner.ModelRecorder(tmp_path, 1)
    result = runner.run_case(case, "frozen", "frozen-test-01", tmp_path, "2026-10-25T04:00:00+00:00", recorder)
    assert result["status"] == "COMPLETED"
    assert result["model_requests"] == 0
    assert all(result["checks"].values())
    card = (tmp_path / "frozen-test-01/card.md").read_text(encoding="utf-8")
    assert "不预先填通过" in card and "未取得可引用" in card
    assert (tmp_path / "frozen-test-01/checkpoints.json").exists()
