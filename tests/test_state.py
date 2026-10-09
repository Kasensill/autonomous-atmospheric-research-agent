import pytest

from atmospheric_research_agent.agent.nodes import (
    ReportReviewOutput,
    check_citation_integrity,
    route_after_report_verification,
    verify_report_with_model,
)
from atmospheric_research_agent.agent.state import create_initial_state


def test_create_initial_state_sets_explicit_defaults() -> None:
    state = create_initial_state("  什么条件有利于台风形成？  ")

    assert state["question"] == "什么条件有利于台风形成？"
    assert state["run_id"]
    assert state["started_at"]
    assert state["finished_at"] is None
    assert state["run_duration_ms"] is None
    assert state["messages"] == []
    assert state["recalled_memories"] == []
    assert state["evidence"] == []
    assert state["research_round"] == 0
    assert state["tool_call_count"] == 0
    assert state["clarification_question"] is None
    assert state["final_report"] is None
    assert state["draft_report"] is None
    assert state["report_revision_count"] == 0
    assert "report_review" not in state
    assert state["trace"][0]["event"] == "run_started"


class FakeVerificationModel:
    def __init__(self, result: ReportReviewOutput) -> None:
        self.result = result
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return self.result


def _state_with_draft() -> dict:
    state = create_initial_state("测试问题")
    state["draft_report"] = "# 草稿\n\n结论 [E1]。"
    state["evidence"] = [
        {
            "evidence_id": "e1",
            "content": "测试证据。",
            "source_display_name": "知识库：测试",
            "source_location": "test.md",
            "source_type": "local_knowledge",
            "tool_name": "search_local_knowledge",
        }
    ]
    return state


def test_verify_report_returns_structured_review_without_network() -> None:
    model = FakeVerificationModel(
        ReportReviewOutput(
            verdict="revise",
            supported_points=["有来源的结论"],
            unsupported_or_overstated_claims=["把相关性写成因果性"],
            citation_issues=[],
            required_changes=["将因果表述改为相关或可能影响"],
            reason="存在一处过度推断。",
        )
    )

    update = verify_report_with_model(_state_with_draft(), model)

    assert update["report_review"]["verdict"] == "revise"
    assert "因果" in update["report_review"]["required_changes"][0]
    assert "正式证据" in model.received_messages[1].content


def test_citation_integrity_rejects_nonexistent_evidence_number_even_if_model_passes() -> None:
    state = _state_with_draft()
    state["draft_report"] = "# 草稿\n\n一个没有依据的引用 [E99]。"
    model = FakeVerificationModel(
        ReportReviewOutput(
            verdict="pass",
            supported_points=["模型误判为通过"],
            unsupported_or_overstated_claims=[],
            citation_issues=[],
            required_changes=[],
            reason="模型认为没有问题。",
        )
    )

    update = verify_report_with_model(state, model)

    assert update["report_review"]["verdict"] == "revise"
    assert "[E99]" in update["report_review"]["citation_issues"][0]
    assert route_after_report_verification({**state, **update}) == "revise_report"


def test_citation_integrity_accepts_existing_evidence_number() -> None:
    assert check_citation_integrity(_state_with_draft()) == []


@pytest.mark.parametrize(
    ("review", "revision_count", "expected_route"),
    [
        ({"verdict": "pass"}, 0, "finalize_report"),
        ({"verdict": "revise"}, 0, "revise_report"),
        ({"verdict": "revise"}, 1, "finalize_limited_report"),
    ],
)
def test_route_after_report_verification_has_a_bounded_loop(
    review: dict, revision_count: int, expected_route: str
) -> None:
    state = create_initial_state("测试")
    state["report_review"] = review
    state["report_revision_count"] = revision_count

    assert route_after_report_verification(state) == expected_route
