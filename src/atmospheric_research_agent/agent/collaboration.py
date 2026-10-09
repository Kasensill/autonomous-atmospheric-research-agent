"""V3 Multi-Agent 的结构化交接契约。

这里暂不创建 Coordinator 或第二个 Agent Loop。先把不同角色之间能交换什么
数据固定下来，避免未来子图之间传递整段 messages、混淆证据归属。
"""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class DelegationTask(BaseModel):
    """Coordinator 发给某个专业 Agent 的任务单。"""

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    target: Literal["research", "event_data"]
    goal: str = Field(min_length=1)
    question: str = Field(min_length=1)
    required_outputs: list[str] = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)
    location: str | None = None
    time_range: str | None = None

    @model_validator(mode="after")
    def event_data_task_requires_time_and_location(self) -> "DelegationTask":
        """事件数据任务不能让 Data Agent 猜测用户本应提供的时空范围。"""

        if self.target == "event_data" and (not self.location or not self.time_range):
            raise ValueError("event_data 任务必须提供 location 和 time_range。")
        return self


class PacketEvidence(BaseModel):
    """子 Agent 交付的证据引用；正文仍必须能追溯到具体来源。"""

    evidence_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_display_name: str = Field(min_length=1)
    source_location: str = Field(min_length=1)
    source_type: Literal["local_knowledge", "web", "weather", "observation", "reanalysis"]


class EvidenceClaim(BaseModel):
    """一个结论及其对应证据编号，防止只交付无来源的自然语言总结。"""

    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: Literal["high", "medium", "low"]


class ResearchPacket(BaseModel):
    """Research Agent 的机制研究交付物。"""

    task_id: str
    status: Literal["complete", "partial", "needs_clarification"]
    claims: list[EvidenceClaim] = Field(default_factory=list)
    evidence: list[PacketEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_data: list[str] = Field(default_factory=list)


class DataPacket(BaseModel):
    """Event/Data Agent 的地点、时间和观测/再分析事实交付物。"""

    task_id: str
    status: Literal["complete", "partial", "needs_clarification"]
    observations: list[EvidenceClaim] = Field(default_factory=list)
    evidence: list[PacketEvidence] = Field(default_factory=list)
    data_limits: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)


class VerificationFollowUp(BaseModel):
    """Verifier 指定补查目标，Coordinator 据此精确派回对应 Agent。"""

    target: Literal["research", "event_data"]
    goal: str = Field(min_length=1)
    required_outputs: list[str] = Field(min_length=1)


class VerificationResult(BaseModel):
    """Verifier 对多个证据包的结构化结论，而非最终报告正文。"""

    verdict: Literal["pass", "revise", "blocked"]
    supported_conclusions: list[str] = Field(default_factory=list)
    unsupported_or_overstated_claims: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    follow_up_tasks: list[VerificationFollowUp] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def revise_requires_a_targeted_follow_up(self) -> "VerificationResult":
        if self.verdict == "revise" and not self.follow_up_tasks:
            raise ValueError("verdict=revise 时必须给出至少一条明确的补查任务。")
        return self
