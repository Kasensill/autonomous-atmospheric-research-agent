from atmospheric_research_agent.agent.memory import (
    make_memory_record,
    recall_project_memory,
    save_project_memory,
)


def _record(question: str, summary: str) -> dict:
    return {
        "memory_id": f"memory-{question}",
        "question": question,
        "summary": summary,
        "source_locations": ["knowledge.md"],
        "created_at": "2026-09-20T00:00:00+00:00",
        "status": "verified_report",
    }


def test_project_memory_recalls_related_record_but_not_unrelated_record(tmp_path) -> None:
    store_path = tmp_path / "project_memory.json"
    save_project_memory(
        _record("副热带高压为什么带来高温少雨？", "下沉运动抑制云雨并促进晴空增温。"),
        store_path=store_path,
    )
    save_project_memory(
        _record("台风如何形成？", "需要暖海温和低风切变。"),
        store_path=store_path,
    )

    memories = recall_project_memory("副热带高压的下沉运动有什么影响？", store_path=store_path)

    assert len(memories) == 1
    assert memories[0]["question"] == "副热带高压为什么带来高温少雨？"


def test_memory_summary_excludes_program_reference_section() -> None:
    record = make_memory_record(
        question="测试问题",
        final_report="# 结论\n\n有来源的结论 [E1]。\n\n## 资料来源\n\n- [E1] 测试来源",
        evidence=[
            {
                "evidence_id": "e1",
                "source_display_name": "知识库：测试来源",
                "source_location": "test.md",
            }
        ],
    )

    assert "有来源的结论" in record["summary"]
    assert "资料来源" not in record["summary"]
    assert record["source_locations"] == ["test.md"]
