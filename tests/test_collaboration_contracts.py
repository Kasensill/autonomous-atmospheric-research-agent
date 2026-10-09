import pytest
from pydantic import ValidationError

from atmospheric_research_agent.agent.collaboration import (
    DelegationTask,
    VerificationResult,
)


def test_event_data_delegation_requires_location_and_time_range() -> None:
    with pytest.raises(ValidationError, match="location 和 time_range"):
        DelegationTask(
            target="event_data",
            goal="核对事件数据",
            question="郑州暴雨为何形成？",
            required_outputs=["降水观测"],
        )


def test_research_delegation_can_omit_event_specific_scope() -> None:
    task = DelegationTask(
        target="research",
        goal="解释极端降水的一般机制",
        question="暴雨形成需要什么条件？",
        required_outputs=["机制证据"],
    )

    assert task.target == "research"
    assert task.task_id


def test_verifier_revise_must_name_a_follow_up_target() -> None:
    with pytest.raises(ValidationError, match="补查任务"):
        VerificationResult(
            verdict="revise",
            reason="缺少事件数据。",
        )


def test_verifier_pass_needs_no_follow_up_task() -> None:
    result = VerificationResult(
        verdict="pass",
        supported_conclusions=["结论有两类证据共同支持。"],
        reason="机制和事件事实均有可追溯证据。",
    )

    assert result.follow_up_tasks == []
