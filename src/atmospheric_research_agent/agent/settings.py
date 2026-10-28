"""在线 Research Agent 的集中运行配置。

这里放的是可预期会调整、但不应由模型写入 ResearchState 的程序规则。
离线入库的切分与向量库配置仍保留在 ``ingestion.build_index``，因为它们属于
另一条独立的数据准备流程。
"""

from collections.abc import Mapping
from typing import Any

# 图与研究循环预算：防止模型反复调用工具或让图无限循环。
MAX_RESEARCH_ROUNDS = 4
MAX_TOOL_CALLS = 8
# 单次调用的最长正常推进路径：无追问入口 recall_memory 1 步；恢复入口
# request_clarification 1 步。各加 4 轮 × 6 步（逐批取证并评估），再加
# 报告阶段 4 + 2 × 2 = 8 步（含两次修订、定稿、store_memory），均为 33 步。
# 33 × 1.2 = 39.6，取 40；这是兜底保护，不替代研究/工具/修订预算。
# 修改研究轮次、修订上限或图结构时必须重新核算。
GRAPH_RECURSION_LIMIT = 40
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
# 首版命令行材料限额；只限制送入本任务的材料，不改变研究预算，均可按验收结果调整。
MATERIAL_MAX_FILES = 3
MATERIAL_MAX_CHARS_TOTAL = 20_000
# 文本替换字符或异常控制字符比例超过此值时拒绝，避免把乱码送入模型；可调。
MATERIAL_MAX_GARBLED_RATIO = 0.02
# 材料块边界；加载时对正文中的同名标记等长转义，不能充当指令或闭合边界。
MATERIAL_BEGIN = "<USER_MATERIAL_DATA>"
MATERIAL_END = "</USER_MATERIAL_DATA>"
# auto 角色判定词表，可调；仅用于选择材料用途，不判断内容真假。
MATERIAL_REFERENCE_WORDS = ("这份", "该份", "此份", "附件", "上传的", "材料", "文件", "文档", "报告", "论文")
MATERIAL_TEXT_TASK_WORDS = ("总结", "概括", "摘要", "提炼", "梳理", "分析论证", "检查论证", "比较", "对比", "翻译", "解读")
MATERIAL_VERIFICATION_WORDS = ("核实", "核对", "是否真实", "是否属实", "验证", "真假", "是否准确")
# PDF 限额独立于最终证据正文上限；超时由可终止的独立解析进程执行。
PDF_MAX_BYTES = 10 * 1024 * 1024
PDF_MAX_PAGES = 30
PDF_MAX_EXTRACTED_CHARACTERS = 100_000
PDF_PARSE_TIMEOUT_SECONDS = 15
PDF_MIN_PAGE_TEXT_CHARACTERS = 20
PDF_MAX_SPARSE_PAGE_RATIO = 0.8
PDF_MAX_GARBLED_RATIO = 0.02
PDF_MIN_CID_OCCURRENCES = 3

# 本地 HTTP 上传入站限制；不改变 CLI、PDF 解析或研究循环预算，均为可调工程限额。
MATERIAL_UPLOAD_MAX_BYTES = PDF_MAX_BYTES  # 每文件最多与 PDF 相同的 10MB，TXT/MD 同样受限。
MATERIAL_UPLOAD_MAX_REQUEST_BYTES = MATERIAL_MAX_FILES * MATERIAL_UPLOAD_MAX_BYTES + 64 * 1024  # 含表单开销。
MATERIAL_UPLOAD_MAX_NAME_CHARACTERS = 120  # 仅接受短文件名，不接受路径或设备名。
MATERIAL_UPLOAD_DIRECTORY = "tmp/material_uploads"  # 已忽略的本地原文件目录；不作为解析缓存。

# 未在 .env 中覆盖时使用的模型连接默认值；密钥始终只从 .env 读取。
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
