"""兼容入口：研究节点按职责分别实现在相邻模块中。"""
from __future__ import annotations

import json
import os
import re
from hashlib import sha256
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator
from langgraph.types import interrupt

from .settings import (
    DEFAULT_DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    MODEL_REQUEST_TIMEOUT_SECONDS,
    MAX_REPORT_REVISION_ROUNDS,
    MAX_RESEARCH_ROUNDS,
    MAX_TOOL_CALLS,
    BRIEF_REPORT_MAX_CHARACTERS,
    research_budget_exhausted,
    tool_batch_fits_budget,
    MAX_CLARIFICATIONS,
    clarifications_exhausted,
)
from .model_trace import model_trace_options
from .memory import make_memory_record, recall_project_memory, save_project_memory, is_time_sensitive_question
from .state import (
    EvidenceAssessment,
    EvidenceRecord,
    ReportReview,
    ResearchPlan,
    ResearchState,
    TraceEntry,
)
from .tools import RESEARCH_TOOLS
from .tools.clarification import ASK_USER_NAME, ASK_USER_TOOL
from .evidence import usable_evidence, is_stale_weather, research_evidence, material_only_unverified, has_user_materials
from .materials import coverage_text, escape_material_markers
from .settings import MATERIAL_BEGIN, MATERIAL_END, HISTORY_MAX_CANDIDATES
from .history import HistoryRequest, can_lookup, matches_request, select_evidence
from .clock import BEIJING_TIMEZONE, parse_reference_time, reference_now, time_context



from .model_factory import (
    EvidenceAssessmentOutput,
    ReportReviewOutput,
    _create_base_model,
    create_research_model,
    create_assessment_model,
    _verification_thinking,
    create_verification_model,
)
from .node_shared import (
    _numbered_evidence,
    _clarification_rules,
    _clarification_limit_update,
    _without_material_body,
    _material_messages,
    _task_question,
    _comparison_scope_rule,
)
from .decision import (
    AGENT_SYSTEM_PROMPT,
    NoToolDecision,
    parse_no_tool_decision,
    _messages_for_model,
    agent_node,
    route_after_agent,
)
from .evidence_nodes import (
    research_guard_node,
    route_after_research_guard,
    _assessment_messages,
    _assessment_clarification_quota,
    assess_evidence_with_model,
    assess_evidence_node,
    route_after_evidence_assessment,
)
from .memory_nodes import (
    recall_memory_node,
    store_memory_node,
)
from .clarification_nodes import (
    request_clarification_node,
)
from .report_nodes import (
    _report_mode,
    _reference_section,
    _plain_reference_text,
    _reference_url,
    _reference_details,
    _length_instruction,
    _single_sentence_requested,
    _single_sentence_issue,
    _brief_length_issue,
    _report_messages,
    _verification_messages,
    check_citation_integrity,
    verify_report_with_model,
    verify_report_node,
    route_after_report_verification,
    generate_draft_with_model,
    generate_draft_node,
    _revision_messages,
    revise_report_with_model,
    revise_report_node,
    finalize_report_node,
    _limited_report,
    _unusable_weather_note,
    finalize_limited_report_node,
)
from .tool_nodes import (
    _control_response,
    _latest_tool_call_message,
    _tool_registry,
    execute_tool_calls,
    tools_node,
    _recent_tool_messages,
    _evidence_from_local_knowledge,
    _evidence_from_tool_message,
    record_results_node,
)
