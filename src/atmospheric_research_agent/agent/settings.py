"""在线 Research Agent 的集中运行配置。

这里放的是可预期会调整、但不应由模型写入 ResearchState 的程序规则。
离线入库的切分与向量库配置仍保留在 ``ingestion.build_index``，因为它们属于
另一条独立的数据准备流程。
"""

# 图与研究循环预算：防止模型反复调用工具或让图无限循环。
MAX_RESEARCH_ROUNDS = 3
MAX_TOOL_CALLS = 8
GRAPH_RECURSION_LIMIT = 30

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
WEB_SEARCH_TIMEOUT_SECONDS = 15
WEB_SEARCH_MAX_RESULTS = 5
WEATHER_REQUEST_TIMEOUT_SECONDS = 15
MAX_SOURCE_CHARACTERS = 12_000

# 未在 .env 中覆盖时使用的模型连接默认值；密钥始终只从 .env 读取。
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
