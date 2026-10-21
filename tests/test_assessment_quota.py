import pytest

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.run import format_final_result
from atmospheric_research_agent.agent.settings import MAX_CLARIFICATIONS
from atmospheric_research_agent.agent.state import create_initial_state


def state_with_history(used):
    state = create_initial_state("研究")
    state.update(
        evidence=[{"source_type": "local_knowledge", "content": "一般机制"}],
        clarification_history=[{"question": str(i), "answer": "补充"} for i in range(used)],
        research_round=3, tool_call_count=8,
    )
    return state


@pytest.mark.parametrize("used,sentence", [
    (0, "当前可以追问用户（已追问 0 次，还可追问 2 次）。"),
    (1, "当前可以追问用户（已追问 1 次，还可追问 1 次）。"),
    (2, "当前不可再追问用户。"),
    (3, "当前不可再追问用户。"),
])
def test_quota_sentence_is_explicit_and_separate_from_research_budgets(used, sentence):
    messages = nodes._assessment_messages(state_with_history(used))
    assert len(messages) == 2
    assert messages[0].content.endswith("\n\n" + sentence)
    assert "当前研究轮次" not in messages[0].content
    assert "当前工具调用数" not in messages[0].content
    assert "当前研究轮次：3/3" in messages[1].content
    assert "当前工具调用数：8/8" in messages[1].content
    assert "已完成澄清" not in "\n".join(m.content for m in messages)


@pytest.mark.parametrize("used,next_step,reason,expected_route", [
    (0, "clarify", "澄清额度已满", "request_clarification"),
    (2, "clarify", "仍有澄清额度", "generate_draft"),
    (0, "limited_report", "澄清额度已满", "generate_draft"),
])
def test_program_quota_and_routing_do_not_trust_free_text_reason(used, next_step, reason, expected_route):
    class Model:
        def invoke(self, messages):
            return nodes.EvidenceAssessmentOutput(
                sufficient=False, covered_points=[], missing_points=["地点"], potential_conflicts=[],
                next_step=next_step, reason=reason,
                clarification_question="哪个城市？" if next_step == "clarify" else None,
            )

    state = state_with_history(used)
    update = nodes.assess_evidence_with_model(state, Model())
    event = next(e for e in update["trace"] if e["event"] == "assessment_completed")
    assert event["clarification_quota"] == {
        "used": used, "limit": MAX_CLARIFICATIONS, "remaining": max(0, MAX_CLARIFICATIONS - used),
    }
    assert nodes.route_after_evidence_assessment({**state, **update}) == expected_route
    output = format_final_result({**state, **update}, debug=True)
    assert '"clarification_quota"' in output and f'"used": {used}' in output
    assert reason in event["detail"]  # 保留模型解释，但不把它当作程序事实。
