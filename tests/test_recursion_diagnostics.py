"""真实图、mock 模型：仅复现循环并检验诊断，不改变路由或额度。"""

import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from atmospheric_research_agent.agent import nodes
import atmospheric_research_agent.agent.graph as graph_module
import atmospheric_research_agent.agent.run as run_module
from atmospheric_research_agent.agent.settings import MAX_CLARIFICATIONS, GRAPH_RECURSION_LIMIT


def setup_loop(monkeypatch, tmp_path, phase):
    monkeypatch.setattr(run_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_module, "_check_debug_directory_ignored", lambda _: None)
    completed = MAX_CLARIFICATIONS - 1 if phase == "resume" else MAX_CLARIFICATIONS
    monkeypatch.setattr(graph_module, "recall_memory_node", lambda _: {
        "evidence": [{"evidence_id": "one", "content": "一般机制。", "source_type": "local_knowledge",
                      "source_location": "test.md"}],
        "clarification_history": [{"question": "条件？", "answer": "已有补充"} for _ in range(completed)],
    })
    calls = []

    class AgentModel:
        def invoke(self, messages):
            calls.append(len(calls) + 1)
            # 此刻节点已经开始，证明记录不是运行结束后补写的。
            paths = list(tmp_path.rglob("node_trace.jsonl"))
            if paths:
                entries = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
                assert entries[-1]["node"] == "agent"
                assert entries[-1]["event"] == "node_started"
            return AIMessage(content="", tool_calls=[{
                "name": "ask_user", "id": f"ask-{len(calls)}", "args": {"question": "地点和日期？"},
            }])

    class AssessmentModel:
        def invoke(self, messages):
            return nodes.EvidenceAssessmentOutput(
                sufficient=False, covered_points=[], missing_points=["事件实况"], potential_conflicts=[],
                next_step="continue", reason="资料仍不足。",
            )

    monkeypatch.setattr(nodes, "create_research_model", AgentModel)
    monkeypatch.setattr(nodes, "create_assessment_model", AssessmentModel)
    monkeypatch.setattr(graph_module, "tools_node", lambda _: pytest.fail("不应执行研究工具"))
    monkeypatch.setattr(graph_module, "store_memory_node", lambda _: pytest.fail("不应存记忆"))
    return calls


@pytest.mark.parametrize("phase", ["initial", "resume"])
def test_recursion_error_is_explained_and_checkpoint_is_saved(monkeypatch, tmp_path, capsys, phase):
    calls = setup_loop(monkeypatch, tmp_path, phase)
    answers = []

    def answer(prompt):
        answers.append(prompt)
        return "济宁，指定日期。"

    monkeypatch.setattr("builtins.input", answer)
    monkeypatch.setattr(run_module.sys, "argv", ["agent.run", "研究", "--debug"])
    with pytest.raises(SystemExit) as stopped:
        run_module.main()
    assert stopped.value.code != 0
    output = capsys.readouterr().out
    assert "流程超过步数上限被中止" in output
    assert "不是系统崩溃" in output
    assert "超步数诊断已保存" in output
    assert "Traceback" not in output
    assert len(answers) == (1 if phase == "resume" else 0)
    assert len(calls) > 2
    diagnostic_path, = tmp_path.rglob("recursion_error.json")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert diagnostic["phase"] == phase
    assert diagnostic["recursion_limit"] == GRAPH_RECURSION_LIMIT
    assert diagnostic["research_round"] == 0
    assert diagnostic["tool_call_count"] == 0
    assert diagnostic["clarifications_used"] == MAX_CLARIFICATIONS
    assert diagnostic["report_revision_count"] == 0
    assert diagnostic["next_nodes"]
    live = [json.loads(line) for line in (diagnostic_path.parent / "node_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    starts = [item for item in live if item["event"] == "node_started"]
    assert [item["sequence"] for item in starts] == list(range(1, len(starts) + 1))
    assert all(isinstance(item["step"], int) for item in starts)
    completed = [item["node"] for item in live if item["event"] == "node_completed"]
    assert completed == diagnostic["last_executed_nodes"]
    assert len(diagnostic["node_sequence"]) == len(starts)
    if phase == "resume":
        assert any(item["node"] == "request_clarification" for item in starts)
    trace = json.loads((diagnostic_path.parent / "trace.json").read_text(encoding="utf-8"))
    assert any(item["event"] == "clarification_limit_reached" for item in trace)
    assert not any(item["event"] == "run_completed" for item in trace)
    assert not (diagnostic_path.parent / "final_report.md").exists()


def test_without_debug_still_explains_failure_without_creating_files(monkeypatch, tmp_path, capsys):
    setup_loop(monkeypatch, tmp_path, "resume")
    monkeypatch.setattr("builtins.input", lambda _: "补充条件")
    monkeypatch.setattr(run_module.sys, "argv", ["agent.run", "研究"])
    with pytest.raises(SystemExit) as stopped:
        run_module.main()
    assert stopped.value.code == 1
    assert "流程超过步数上限被中止" in capsys.readouterr().out
    assert not (tmp_path / "docs").exists()


def test_debug_writes_started_node_before_unhandled_model_error(monkeypatch, tmp_path):
    setup_loop(monkeypatch, tmp_path, "initial")

    class BrokenModel:
        def invoke(self, messages):
            raise RuntimeError("offline model failure")

    monkeypatch.setattr(nodes, "create_research_model", BrokenModel)
    with pytest.raises(RuntimeError, match="offline model failure"):
        run_module.run_research("研究", debug=True)
    path, = tmp_path.rglob("node_trace.jsonl")
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["node"] == "agent"
    assert events[-1]["event"] == "node_error"
    assert any(item["node"] == "agent" and item["event"] == "node_started" for item in events)
    assert not list(tmp_path.rglob("recursion_error.json"))


def test_diagnostic_failure_does_not_hide_recursion_error(monkeypatch, tmp_path):
    class Graph:
        def invoke(self, *args, **kwargs):
            raise GraphRecursionError("step limit")

        def get_state(self, config):
            raise OSError("checkpoint unavailable")

    with pytest.raises(GraphRecursionError) as caught:
        run_module._invoke_with_diagnostics(
            Graph(), {}, {"recursion_limit": GRAPH_RECURSION_LIMIT}, phase="initial",
            directory=tmp_path, live_trace=None,
        )
    assert "诊断保存失败" in caught.value.__notes__[0]
