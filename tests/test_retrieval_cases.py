import json
from pathlib import Path

import pytest

from atmospheric_research_agent.evaluation.retrieval_cases import load_retrieval_cases
from atmospheric_research_agent.ingestion.build_index import split_markdown_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals" / "retrieval_cases.json"
KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"


def test_case_dataset_loads_and_has_expected_v1_6_coverage() -> None:
    cases = load_retrieval_cases(CASES_PATH)

    assert len(cases) == 12
    assert {case.category for case in cases} >= {
        "semantic_paraphrase",
        "boundary_condition",
        "multimechanism",
    }
    assert not [case.case_id for case in cases if case.known_data_gap]


def test_all_case_source_labels_point_to_real_non_bibliography_sections() -> None:
    # 校验真正会进入索引的层级路径；存在Markdown标题不等于有可检索正文。
    section_cache = {}
    for case in load_retrieval_cases(CASES_PATH):
        for source in (*case.expected_sources, *case.irrelevant_sources):
            document = KNOWLEDGE_BASE / source.location
            assert document.is_file(), f"{case.case_id}: 缺少资料 {source.location}"
            if source.location not in section_cache:
                section_cache[source.location] = {
                    chunk.metadata["section"]
                    for chunk in split_markdown_file(document, KNOWLEDGE_BASE)
                }
            assert source.section in section_cache[source.location], (
                f"{case.case_id}: {source.location} 中没有可检索章节 {source.section}"
            )
            assert all(part.strip() not in {"资料说明", "资料来源", "参考资料", "参考文献"}
                       for part in source.section.split(">"))


def test_migrated_labels_have_body_support_not_only_title_or_scope() -> None:
    """迁移后的正标签必须保留可抽验的正文锚点，不用整篇/资料说明凑命中。"""
    raw = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    migrated = [case for case in raw if case.get("annotation_revision") == "2026-10-27"]
    assert {case["id"] for case in migrated} == {
        "subtropical_high_semantic_mechanism", "high_pressure_boundary", "low_level_jet_heavy_rain",
    }
    for case in migrated:
        for source in case["expected_sources"]:
            chunks = split_markdown_file(KNOWLEDGE_BASE / source["location"], KNOWLEDGE_BASE)
            bodies = [chunk.text.split(f"## {source['section']}\n", 1)[1]
                      for chunk in chunks if chunk.metadata["section"] == source["section"]]
            assert source["supporting_text"]
            assert any(source["supporting_text"] in body for body in bodies), case["id"]


def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    case_path = tmp_path / "duplicate.json"
    case_path.write_text(
        json.dumps(
            [
                {
                    "id": "duplicate",
                    "question": "问题一",
                    "category": "test",
                    "expected_sources": [{"location": "a.md", "section": "定义与背景"}],
                    "note": "说明",
                },
                {
                    "id": "duplicate",
                    "question": "问题二",
                    "category": "test",
                    "expected_sources": [{"location": "b.md", "section": "形成机制"}],
                    "note": "说明",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复"):
        load_retrieval_cases(case_path)
