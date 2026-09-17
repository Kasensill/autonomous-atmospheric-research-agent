import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import atmospheric_research_agent.agent.nodes as nodes_module
from atmospheric_research_agent.agent.nodes import (
    _messages_for_model,
    EvidenceAssessmentOutput,
    assess_evidence_with_model,
    execute_tool_calls,
    generate_report_with_model,
    parse_no_tool_decision,
    record_results_node,
    _reference_section,
    request_clarification_node,
    research_guard_node,
    route_after_agent,
    route_after_evidence_assessment,
    route_after_research_guard,
)
from atmospheric_research_agent.agent.state import create_initial_state


def test_parse_no_tool_assess_decision() -> None:
    decision = parse_no_tool_decision(
        '{"decision":"assess","clarification_question":null,'
        '"plan_update":{"goal":"解释副热带高压","focus_points":["下沉运动"]}}'
    )

    assert decision.decision == "assess"
    assert decision.clarification_question is None
    assert decision.plan_update["goal"] == "解释副热带高压"


def test_deepseek_agent_client_disables_thinking_mode_for_tool_loop(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(nodes_module, "ChatOpenAI", FakeChatOpenAI)

    nodes_module._create_base_model()

    assert captured_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_parse_no_tool_decision_rejects_plain_text() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_no_tool_decision("我认为需要继续研究。")


@tool
def multiply(a: int, b: int) -> dict[str, int]:
    """计算两个整数的乘积。"""

    return {"value": a * b}


def test_execute_tool_calls_preserves_call_id_and_result() -> None:
    messages = execute_tool_calls(
        [{"name": "multiply", "args": {"a": 6, "b": 7}, "id": "call_123"}],
        [multiply],
    )

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "call_123"
    assert messages[0].name == "multiply"
    assert json.loads(messages[0].content) == {"value": 42}


def test_execute_tool_calls_returns_error_for_unknown_tool() -> None:
    messages = execute_tool_calls(
        [{"name": "not_allowed", "args": {}, "id": "call_404"}],
        [multiply],
    )

    assert messages[0].status == "error"
    assert json.loads(messages[0].content)["status"] == "error"


def test_first_agent_update_keeps_original_human_message() -> None:
    initial_state = create_initial_state("什么是锋面？")
    response = AIMessage(content="{}")

    # _messages_for_model 负责发送初始 HumanMessage；agent_node 会将其写回 State。
    prompt_messages = _messages_for_model(initial_state)
    assert isinstance(prompt_messages[-1], HumanMessage)
    assert prompt_messages[-1].content == "什么是锋面？"


def test_record_results_turns_local_knowledge_into_evidence() -> None:
    state = create_initial_state("副热带高压有什么影响？")
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_local_knowledge", "args": {}, "id": "call_001"}],
        ),
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call_001",
            content=json.dumps(
                {
                    "status": "success",
                    "results": [
                        {
                            "content": "下沉运动有助于晴热。",
                            "source": {
                                "display_name": "知识库：环流 > 副热带高压 > 影响",
                                "location": "03_环流/副热带高压.md",
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ),
    ]

    update = record_results_node(state)

    assert update["tool_call_count"] == 1
    assert update["research_round"] == 1
    assert len(update["evidence"][0]["evidence_id"]) == 24
    assert update["evidence"][0]["source_location"] == "03_环流/副热带高压.md"


def test_record_results_does_not_turn_tool_error_into_evidence() -> None:
    state = create_initial_state("测试工具失败")
    state["messages"] = [
        ToolMessage(
            name="search_local_knowledge",
            tool_call_id="call_error",
            status="error",
            content='{"status":"error","message":"向量库不可用"}',
        )
    ]

    update = record_results_node(state)

    assert update["evidence"] == []
    assert update["tool_call_count"] == 1
    assert update["trace"][0]["event"] == "tool_failed"


def test_record_results_turns_current_weather_into_evidence() -> None:
    state = create_initial_state("北京现在天气如何？")
    state["messages"] = [
        ToolMessage(
            name="get_current_weather",
            tool_call_id="call_weather",
            content=json.dumps(
                {
                    "status": "success",
                    "observed_at": "2026-09-16T10:00",
                    "evidence_text": "北京在 10:00 的气温为 25°C。",
                    "source": {
                        "display_name": "Open-Meteo 当前条件：北京",
                        "location": "https://example.test/weather",
                        "organization": "Open-Meteo",
                    },
                },
                ensure_ascii=False,
            ),
        )
    ]

    update = record_results_node(state)

    assert update["evidence"][0]["source_type"] == "weather"
    assert update["evidence"][0]["published_or_observed_at"] == "2026-09-16T10:00"
    assert update["evidence"][0]["author_or_organization"] == "Open-Meteo"


def test_record_results_keeps_search_results_as_candidates() -> None:
    state = create_initial_state("寻找公开资料")
    state["messages"] = [
        ToolMessage(
            name="search_web",
            tool_call_id="call_search",
            content='{"status":"success","result_count":2,"results":[]}',
        )
    ]

    update = record_results_node(state)

    assert update["evidence"] == []
    assert update["trace"][0]["event"] == "search_candidates_available"


def test_record_results_turns_read_source_into_web_evidence() -> None:
    state = create_initial_state("读取公开资料")
    state["messages"] = [
        ToolMessage(
            name="read_source",
            tool_call_id="call_read",
            content=json.dumps(
                {
                    "status": "success",
                    "content": "网页主正文。",
                    "source": {
                        "display_name": "网页标题",
                        "location": "https://example.org/article",
                        "author": "作者甲",
                        "organization": "示例机构",
                        "published_at": "2026-01-01",
                    },
                },
                ensure_ascii=False,
            ),
        )
    ]

    update = record_results_node(state)

    assert update["evidence"][0]["source_type"] == "web"
    assert update["evidence"][0]["author_or_organization"] == "作者甲"
    assert update["evidence"][0]["published_or_observed_at"] == "2026-01-01"


def test_record_results_deduplicates_same_evidence_within_a_round() -> None:
    result = json.dumps(
        {
            "status": "success",
            "results": [
                {
                    "content": "相同的知识片段。",
                    "source": {"location": "same.md", "display_name": "知识库：同一资料"},
                }
            ],
        },
        ensure_ascii=False,
    )
    state = create_initial_state("测试去重")
    state["messages"] = [
        ToolMessage(name="search_local_knowledge", tool_call_id="call_1", content=result),
        ToolMessage(name="search_local_knowledge", tool_call_id="call_2", content=result),
    ]

    update = record_results_node(state)

    assert len(update["evidence"]) == 1
    assert "跳过 1 条重复证据" in update["trace"][1]["detail"]


def test_research_guard_allows_assessment_when_evidence_exists() -> None:
    state = create_initial_state("测试")
    state["evidence"] = [{"evidence_id": "evidence_1", "content": "已有资料"}]

    update = research_guard_node(state)

    assert "research_round" not in update
    assert route_after_research_guard(state) == "assess_evidence"


def test_research_guard_retries_empty_research_within_budget() -> None:
    state = create_initial_state("测试")

    update = research_guard_node(state)
    state.update(update)

    assert state["research_round"] == 1
    assert route_after_research_guard(state) == "agent"


def test_research_guard_routes_to_limited_report_when_budget_is_exhausted() -> None:
    state = create_initial_state("测试")
    state["research_round"] = 2

    state.update(research_guard_node(state))

    assert state["research_round"] == 3
    assert route_after_research_guard(state) == "generate_report"


class FakeAssessmentModel:
    def invoke(self, messages: list[object]) -> EvidenceAssessmentOutput:
        assert len(messages) == 2
        return EvidenceAssessmentOutput(
            sufficient=False,
            covered_points=["解释了下沉运动"],
            missing_points=["缺少区域与季节条件"],
            potential_conflicts=[],
            next_step="continue",
            reason="证据尚不足以覆盖问题的全部条件。",
        )


def test_assess_evidence_returns_structured_assessment() -> None:
    state = create_initial_state("副热带高压有什么影响？")
    state["evidence"] = [
        {
            "evidence_id": "evidence_1",
            "content": "下沉运动常抑制云雨。",
            "source_display_name": "知识库：副热带高压",
        }
    ]

    update = assess_evidence_with_model(state, FakeAssessmentModel())

    assert update["assessment"]["sufficient"] is False
    assert update["assessment"]["next_step"] == "continue"
    assert update["trace"][0]["event"] == "assessment_completed"


def test_route_after_evidence_assessment_respects_structured_decision() -> None:
    state = create_initial_state("测试")
    state["assessment"] = {
        "sufficient": True,
        "covered_points": ["全部"],
        "missing_points": [],
        "potential_conflicts": [],
        "next_step": "full_report",
        "reason": "证据充分。",
    }

    assert route_after_evidence_assessment(state) == "generate_report"


class FakeReportModel:
    def invoke(self, messages: list[object]) -> AIMessage:
        assert len(messages) == 2
        return AIMessage(content="# 结论\n\n下沉运动可抑制云雨。[E1]")


def test_generate_report_appends_programmatic_sources() -> None:
    state = create_initial_state("副热带高压有什么影响？")
    state["evidence"] = [
        {
            "evidence_id": "evidence_1",
            "content": "下沉运动常抑制云雨。",
            "source_display_name": "知识库：副热带高压 > 影响",
            "source_location": "03_环流/副热带高压.md",
        }
    ]
    state["assessment"] = {
        "sufficient": True,
        "covered_points": ["天气影响"],
        "missing_points": [],
        "potential_conflicts": [],
        "next_step": "full_report",
        "reason": "证据充分。",
    }

    update = generate_report_with_model(state, FakeReportModel())

    assert "下沉运动可抑制云雨。[E1]" in update["final_report"]
    assert "## 资料来源" in update["final_report"]
    assert "[E1] 知识库：副热带高压 > 影响（03_环流/副热带高压.md）" in update["final_report"]


def test_reference_section_normalizes_markdown_titles_and_urls() -> None:
    reference_section = _reference_section(
        [
            {
                "citation_id": "E1",
                "source_display_name": "百科条目（[原始链接](https://example.org/article)）",
                "source_location": "[来源](https://example.org/article)",
                "author_or_organization": "示例机构",
                "published_or_observed_at": "2026-01-01",
            }
        ]
    )

    assert "原始链接](" not in reference_section
    assert "[来源页](https://example.org/article)" in reference_section
    assert "来源：示例机构；时间：2026-01-01" in reference_section


def test_generate_report_without_evidence_does_not_call_model() -> None:
    state = create_initial_state("没有证据的问题")

    update = generate_report_with_model(state, model=None)

    assert "不能给出可靠结论" in update["final_report"]
    assert "未取得可引用的正式证据" in update["final_report"]


def test_request_clarification_records_waiting_state() -> None:
    state = create_initial_state("北京现在天气如何？")
    state["clarification_question"] = "请提供你想查询的具体地点。"

    update = request_clarification_node(state)

    assert update["trace"][0]["event"] == "waiting_for_user_information"
    assert "具体地点" in update["trace"][0]["detail"]


def test_clarify_decision_requires_a_question() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_no_tool_decision(
            '{"decision":"clarify","clarification_question":null,"plan_update":{}}'
        )


def test_route_after_agent_sends_tool_calls_to_tools() -> None:
    state = create_initial_state("测试工具调用")
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_local_knowledge", "args": {}, "id": "call_1"}],
        )
    ]

    assert route_after_agent(state) == "tools"


def test_route_after_agent_sends_clarification_to_user_path() -> None:
    state = create_initial_state("测试澄清")
    state["messages"] = [AIMessage(content="{}")]
    state["clarification_question"] = "请提供地点。"

    assert route_after_agent(state) == "request_clarification"


def test_route_after_agent_blocks_tool_calls_that_exceed_budget() -> None:
    state = create_initial_state("测试预算")
    state["tool_call_count"] = 7
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "search_local_knowledge", "args": {}, "id": "call_1"},
                {"name": "search_web", "args": {}, "id": "call_2"},
            ],
        )
    ]

    assert route_after_agent(state) == "generate_report"


def test_route_after_agent_blocks_tool_calls_after_research_round_limit() -> None:
    state = create_initial_state("测试研究轮次预算")
    state["research_round"] = nodes_module.MAX_RESEARCH_ROUNDS
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[{"name": "search_local_knowledge", "args": {}, "id": "call_1"}],
        )
    ]

    assert route_after_agent(state) == "generate_report"


def test_agent_node_degrades_empty_no_tool_response_to_guard_path(monkeypatch) -> None:
    class EmptyResponseModel:
        def invoke(self, messages):
            return AIMessage(content="")

    monkeypatch.setattr(nodes_module, "create_research_model", lambda: EmptyResponseModel())
    update = nodes_module.agent_node(create_initial_state("测试空模型响应"))

    assert update["trace"][0]["event"] == "empty_no_tool_response"
    state = create_initial_state("测试空模型响应")
    state.update(update)
    assert route_after_agent(state) == "research_guard"


def test_agent_node_degrades_invalid_no_tool_response_to_guard_path(monkeypatch) -> None:
    class InvalidResponseModel:
        def invoke(self, messages):
            return AIMessage(content="我认为证据已经够了。")

    monkeypatch.setattr(nodes_module, "create_research_model", lambda: InvalidResponseModel())
    update = nodes_module.agent_node(create_initial_state("测试非 JSON 模型响应"))

    assert update["trace"][0]["event"] == "invalid_no_tool_response"
    state = create_initial_state("测试非 JSON 模型响应")
    state.update(update)
    assert route_after_agent(state) == "research_guard"
