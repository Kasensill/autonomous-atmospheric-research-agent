from pathlib import Path

import pytest

from atmospheric_research_agent.evaluation.report_verification_cases import (
    load_report_verification_cases,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "evals" / "report_verification_cases.json"


def test_report_verification_dataset_has_positive_and_negative_cases() -> None:
    cases = load_report_verification_cases(CASES_PATH)

    assert len(cases) == 5
    assert [case.expected_verdict for case in cases].count("pass") == 1
    assert [case.expected_verdict for case in cases].count("revise") == 4
    assert {case.category for case in cases} >= {
        "supported_claim",
        "overstatement",
        "citation_missing",
        "citation_invalid",
        "citation_semantic_mismatch",
    }


def test_report_verification_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    duplicated = CASES_PATH.read_text(encoding="utf-8").replace(
        '"absolute_overstatement"', '"supported_qualified_claim"'
    )
    case_path = tmp_path / "duplicated.json"
    case_path.write_text(duplicated, encoding="utf-8")

    with pytest.raises(ValueError, match="ID 重复"):
        load_report_verification_cases(case_path)
