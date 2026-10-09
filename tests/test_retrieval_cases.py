import json
from pathlib import Path

import pytest

from atmospheric_research_agent.evaluation.retrieval_cases import load_retrieval_cases


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
    for case in load_retrieval_cases(CASES_PATH):
        for source in case.expected_sources:
            document = KNOWLEDGE_BASE / source.location
            assert document.is_file(), f"{case.case_id}: 缺少资料 {source.location}"
            headings = {
                line.removeprefix("## ").strip()
                for line in document.read_text(encoding="utf-8").splitlines()
                if line.startswith("## ")
            }
            assert source.section in headings, (
                f"{case.case_id}: {source.location} 中没有章节 {source.section}"
            )
            assert source.section not in {"资料来源", "参考资料", "参考文献"}


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
