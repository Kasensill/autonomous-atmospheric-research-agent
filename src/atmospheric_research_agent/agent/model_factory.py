"""模型客户端、工具绑定与结构化输出。"""
from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

from .settings import DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL, MODEL_REQUEST_TIMEOUT_SECONDS
from .model_trace import model_trace_options
from .tools import RESEARCH_TOOLS
from .tools.clarification import ASK_USER_TOOL


class EvidenceAssessmentOutput(BaseModel):
    """证据评估模型必须返回的结构化结论。"""

    sufficient: bool
    covered_points: list[str]
    missing_points: list[str]
    potential_conflicts: list[str]
    next_step: Literal["continue", "clarify", "full_report", "limited_report"]
    reason: str
    clarification_question: str | None = None
    limitation_reasons: list[Literal["tool_capability_gap", "evidence_conflict", "insufficient_evidence"]] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_clarification_question(self) -> "EvidenceAssessmentOutput":
        if self.next_step == "clarify" and not (self.clarification_question or "").strip():
            raise ValueError("clarify 必须给出具体 clarification_question。")
        return self


class ReportReviewOutput(BaseModel):
    """Reflection / Verification 节点必须返回的可审计检查结论。"""

    verdict: Literal["pass", "revise"]
    supported_points: list[str]
    unsupported_or_overstated_claims: list[str]
    citation_issues: list[str]
    required_changes: list[str]
    reason: str


def _create_base_model(*, thinking: Literal["disabled", "enabled"] = "disabled") -> ChatOpenAI:
    """创建研究模型客户端，工具绑定由调用方处理。"""

    load_dotenv()
    provider = os.getenv("RESEARCH_MODEL_PROVIDER", "deepseek").strip().lower()
    if provider == "qwen":
        key_name = "DASHSCOPE_API_KEY"
        model = os.getenv("QWEN_MODEL", "qwen3.8-flash")
        base_url = os.getenv("QWEN_BASE_URL", "https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1")
        extra_body = {"enable_thinking": thinking == "enabled"}
        options = {"streaming": True} if thinking == "enabled" else {}
    elif provider == "deepseek":
        key_name = "DEEPSEEK_API_KEY"
        model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        extra_body = {"thinking": {"type": thinking}}
        options = {}
    else:
        raise ValueError("RESEARCH_MODEL_PROVIDER 仅允许 qwen 或 deepseek。")
    api_key = os.getenv(key_name)
    if not api_key:
        raise RuntimeError(f"未配置 {key_name}；请在项目根目录的 .env 中设置它。")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
        max_retries=0,
        http_socket_options=(),
        **model_trace_options(MODEL_REQUEST_TIMEOUT_SECONDS),
        extra_body=extra_body,
        **options,
    )


def create_research_model() -> Any:
    """绑定四个研究工具及不可执行的人工澄清声明。"""

    return _create_base_model().bind_tools([*RESEARCH_TOOLS, ASK_USER_TOOL])


def create_assessment_model() -> Any:
    """创建只返回结构化评估、不会请求研究工具的模型。"""

    # DeepSeek 的 thinking 模式不接受 function_calling 所需的强制 tool_choice。
    # json_mode 仍要求 JSON 对象，并由 LangChain 解析、校验为 Pydantic 模型。
    return _create_base_model().with_structured_output(
        EvidenceAssessmentOutput,
        method="json_mode",
    )


def _verification_thinking() -> Literal["disabled", "enabled"]:
    """核验专用实验开关；不影响 Agent、评估或写作模型。"""

    load_dotenv()
    value = os.getenv("VERIFY_THINKING", "disabled")
    if value not in {"disabled", "enabled"}:
        raise ValueError("VERIFY_THINKING 仅允许 disabled 或 enabled；未设置时默认为 disabled。")
    return value


def create_verification_model(*, thinking: Literal["disabled", "enabled"] | None = None) -> Any:
    """创建只审查报告草稿、不会调用工具或直接重写报告的模型。"""

    mode = _verification_thinking() if thinking is None else thinking
    return _create_base_model(thinking=mode).with_structured_output(
        ReportReviewOutput,
        method="json_mode",
    )
