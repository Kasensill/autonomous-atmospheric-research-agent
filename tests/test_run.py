import atmospheric_research_agent.agent.run as run_module


class FakeGraph:
    def __init__(self, result):
        self.result = result
        self.received_state = None
        self.received_config = None

    def invoke(self, state, config):
        self.received_state = state
        self.received_config = config
        return self.result


def test_run_research_starts_graph_with_initial_state(monkeypatch) -> None:
    fake_graph = FakeGraph({"question": "测试", "final_report": "报告"})
    monkeypatch.setattr(run_module, "build_research_graph", lambda: fake_graph)

    result = run_module.run_research("测试")

    assert result["final_report"] == "报告"
    assert fake_graph.received_state["question"] == "测试"
    assert fake_graph.received_config["recursion_limit"] == 30
    assert result["run_id"] == fake_graph.received_state["run_id"]
    assert result["run_duration_ms"] is not None
    assert result["trace"][-1]["event"] == "run_completed"


def test_format_final_result_prioritizes_clarification() -> None:
    output = run_module.format_final_result(
        {
            "clarification_question": "请提供地点。",
            "final_report": "不应显示这份报告。",
        }
    )

    assert "# 需要补充信息" in output
    assert "请提供地点。" in output
    assert "不应显示" not in output


def test_format_final_result_includes_debug_summary() -> None:
    output = run_module.format_final_result(
        {
            "final_report": "# 报告",
            "research_round": 2,
            "tool_call_count": 3,
            "evidence": [{"evidence_id": "1"}],
            "trace": [{"node": "agent", "event": "model_response", "detail": "请求工具。"}],
        },
        debug=True,
    )

    assert "## 运行摘要" in output
    assert "- 研究轮次：2" in output
    assert "agent / model_response / success：请求工具。" in output


def test_format_final_result_exposes_structured_report_review_in_debug_mode() -> None:
    output = run_module.format_final_result(
        {
            "final_report": "# 报告",
            "report_revision_count": 1,
            "report_review": {
                "verdict": "pass",
                "reason": "关键主张均有对应证据。",
                "supported_points": ["下沉运动抑制云雨"],
                "unsupported_or_overstated_claims": [],
                "citation_issues": [],
                "required_changes": [],
            },
        },
        debug=True,
    )

    assert "## 报告核验" in output
    assert "- 审查结论：pass" in output
    assert "- 已修订次数：1" in output
    assert "### 已支持的关键点" in output
    assert "下沉运动抑制云雨" in output
