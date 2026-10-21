"""在线 Research Agent 的集中运行配置。

这里放的是可预期会调整、但不应由模型写入 ResearchState 的程序规则。
离线入库的切分与向量库配置仍保留在 ``ingestion.build_index``，因为它们属于
另一条独立的数据准备流程。
"""

from collections.abc import Mapping
from typing import Any

# 图与研究循环预算：防止模型反复调用工具或让图无限循环。
MAX_RESEARCH_ROUNDS = 3
MAX_TOOL_CALLS = 8
GRAPH_RECURSION_LIMIT = 30
# 报告受控修订预算；避免 Reflection 让图在写作与自检间无限循环。
MAX_REPORT_REVISION_ROUNDS = 2
MAX_CLARIFICATIONS = 2


def clarifications_exhausted(state: Mapping[str, Any]) -> bool:
    """Agent 与证据评估共用已完成问答历史，不额外维护计数器。"""
    return len(state.get("clarification_history", [])) >= MAX_CLARIFICATIONS


def research_rounds_exhausted(state: Mapping[str, Any]) -> bool:
    """研究轮次上限；单独提供以保持预算 Trace 的具体原因。"""
    return state.get("research_round", 0) >= MAX_RESEARCH_ROUNDS


def research_budget_exhausted(state: Mapping[str, Any]) -> bool:
    """任一研究预算达到上限即耗尽。"""
    return research_rounds_exhausted(state) or state.get("tool_call_count", 0) >= MAX_TOOL_CALLS


def tool_batch_fits_budget(state: Mapping[str, Any], requested_count: int) -> bool:
    """完整批次可容纳且研究轮次未满时，才允许执行工具。"""
    return (not research_rounds_exhausted(state)
            and state.get("tool_call_count", 0) + requested_count <= MAX_TOOL_CALLS)

# V2.1 项目记忆：只召回少量相关、已核验报告摘要；不是本次报告的正式证据。
MEMORY_RECALL_LIMIT = 3
MAX_MEMORY_EXCERPT_CHARACTERS = 1_200
MEMORY_MIN_OVERLAP_TERMS = 3
MEMORY_MIN_QUERY_COVERAGE = 0.15
# 简要报告正文的机械长度上限（含 Markdown，不含程序附加的资料来源）。
BRIEF_REPORT_MAX_CHARACTERS = 900

# 工具运行参数：这些不会暴露给模型，但可以在一个位置统一调整。
LOCAL_KNOWLEDGE_TOP_K = 4
# V1.5 的两路召回候选数量；最终给 Agent 的条数仍保持 LOCAL_KNOWLEDGE_TOP_K。
VECTOR_CANDIDATE_TOP_K = 12
BM25_CANDIDATE_TOP_K = 12
# RRF 只融合名次，不直接比较向量距离与 BM25 原始分数。
RRF_RANK_CONSTANT = 60
FUSED_CANDIDATE_TOP_K = 12
RERANK_MODEL = "qwen3.7-text-rerank"
RERANK_FINAL_TOP_K = 4
# 2026-10-20 小型显式正负样本校准的暂定门槛；只适用于当前 Qwen 模型分数。
# 留出集仍有误收，见 docs/validation；不是正确概率或通用最优值。
RERANK_MIN_SCORE = 0.07
# 当前天气证据的业务新鲜度窗口；可调整，不适用于通用机制资料。
WEATHER_EVIDENCE_MAX_AGE_SECONDS = 2 * 60 * 60
WEB_SEARCH_TIMEOUT_SECONDS = 15
WEB_SEARCH_MAX_RESULTS = 5
WEATHER_REQUEST_TIMEOUT_SECONDS = 15
MAX_SOURCE_CHARACTERS = 12_000

# 未在 .env 中覆盖时使用的模型连接默认值；密钥始终只从 .env 读取。
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
