"""离线验证两次修订预算、最新审查反馈与真实图的最终交付路径。"""

import json

import pytest
from langchain_core.messages import AIMessage

import atmospheric_research_agent.agent.graph as graph_module
from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.settings import GRAPH_RECURSION_LIMIT
from atmospheric_research_agent.agent.state import create_initial_state


@pytest.mark.parametrize("last_verdict", ["pass", "revise"])
def test_three_reviews_allow_two_revisions_then_deliver_or_withhold(monkeypatch, last_verdict):
    verdicts = ["revise", "revise", last_verdict]
    reviewed_drafts = []
    revision_inputs = []
    stored_reports = []

    class DraftModel:
        def invoke(self, messages):
            return AIMessage(content="原草稿：下沉运动可抑制云雨。[E1]")

    class ReviewModel:
        def invoke(self, messages):
            index = len(reviewed_drafts)
            assert index < len(verdicts), "不应发生第四次核验"
            reviewed_drafts.append(messages[1].content)
            verdict = verdicts[index]
            return nodes.ReportReviewOutput(
                verdict=verdict,
                supported_points=["下沉运动可抑制云雨"],
                unsupported_or_overstated_claims=[],
                citation_issues=[],
                required_changes=[f"第 {index + 1} 轮要求修改"] if verdict == "revise" else [],
                reason=f"第 {index + 1} 轮审查",
            )

    class RevisionModel:
        def invoke(self, messages):
            prompt = messages[1].content
            review_json = prompt.split("审查清单（JSON）：\n", 1)[1].split("\n\n正式证据", 1)[0]
            review = json.loads(review_json)
            revision_inputs.append(prompt)
            revision = len(revision_inputs)
            assert review["required_changes"] == [f"第 {revision} 轮要求修改"]
            # 第二次修改必须以第一次修订稿为基础，并使用第二轮审查，而非首轮意见。
            previous = "原草稿：" if revision == 1 else f"第 {revision - 1} 次修订稿："
            assert f"待修订草稿：\n{previous}" in prompt
            return AIMessage(content=f"第 {revision} 次修订稿：下沉运动可抑制云雨。[E1]")

    monkeypatch.setattr(graph_module, "recall_memory_node", lambda _: {"recalled_memories": []})
    monkeypatch.setattr(graph_module, "agent_node", lambda _: {"messages": [AIMessage(content="{}")]})
    monkeypatch.setattr(graph_module, "assess_evidence_node", lambda _: {
        "assessment": {"sufficient": True, "next_step": "full_report"},
    })
    monkeypatch.setattr(graph_module, "generate_draft_node", lambda state: nodes.generate_draft_with_model(state, DraftModel()))
    monkeypatch.setattr(graph_module, "verify_report_node", lambda state: nodes.verify_report_with_model(state, ReviewModel()))
    monkeypatch.setattr(graph_module, "revise_report_node", lambda state: nodes.revise_report_with_model(state, RevisionModel()))

    def store(state):
        stored_reports.append(state["final_report"])
        return {}

    monkeypatch.setattr(graph_module, "store_memory_node", store)
    state = create_initial_state("下沉运动如何影响云雨？")
    state["evidence"] = [{
        "evidence_id": "mechanism",
        "content": "下沉运动可抑制云雨。",
        "source_type": "local_knowledge",
        "source_display_name": "内部知识库：垂直运动",
        "source_location": "vertical_motion.md",
    }]
    result = graph_module.build_research_graph().invoke(state, {"recursion_limit": GRAPH_RECURSION_LIMIT})

    assert len(reviewed_drafts) == len(verdicts)
    assert len(revision_inputs) == len(verdicts) - 1
    assert result["report_revision_count"] == len(revision_inputs)
    assert result["report_review"]["reason"] == "第 3 轮审查"
    events = [entry["event"] for entry in result["trace"]]
    if last_verdict == "pass":
        assert "第 2 次修订稿" in result["final_report"]
        assert "## 资料来源" in result["final_report"]
        assert stored_reports == [result["final_report"]]
        assert "report_finalized" in events
        assert "report_withheld_after_failed_review" not in events
    else:
        assert "不输出该草稿" in result["final_report"]
        assert "第 2 次修订稿" not in result["final_report"]
        assert not stored_reports
        assert "report_withheld_after_failed_review" in events
        assert "report_finalized" not in events
