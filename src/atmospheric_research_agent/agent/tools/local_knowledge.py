"""本地大气科学知识库检索工具。"""

import os
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from ...ingestion.build_index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LEXICAL_INDEX_DIR,
    DEFAULT_VECTOR_STORE,
    embed_texts,
)
from ..retrieval.bm25 import bm25_index_path, search_bm25
from ..retrieval.fusion import reciprocal_rank_fusion
from ..retrieval.rerank import rerank_fused_candidates
from ..settings import (
    BM25_CANDIDATE_TOP_K,
    FUSED_CANDIDATE_TOP_K,
    LOCAL_KNOWLEDGE_TOP_K,
    RRF_RANK_CONSTANT,
    VECTOR_CANDIDATE_TOP_K,
)

# 兼容当前函数参数名；实际默认值在 agent.settings 集中维护。
DEFAULT_TOP_K = LOCAL_KNOWLEDGE_TOP_K


def _get_collection(vector_store_dir: Path, collection_name: str) -> Any:
    """打开已建好的 Chroma collection，不在在线查询时创建空库。"""

    if not vector_store_dir.exists():
        raise FileNotFoundError(
            f"向量库不存在：{vector_store_dir}。请先运行 ingestion.build_index 完成入库。"
        )

    try:
        import chromadb
    except ImportError as error:  # pragma: no cover - 依赖安装问题
        raise RuntimeError("未安装 chromadb，请先安装项目依赖。") from error

    client = chromadb.PersistentClient(path=str(vector_store_dir))
    try:
        return client.get_collection(name=collection_name)
    except Exception as error:
        raise RuntimeError(
            f"未找到知识库 collection '{collection_name}'。请先运行入库程序。"
        ) from error


def _source_from_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    """把 Chroma metadata 规范化为工具和对比视图共用的来源结构。"""

    source = metadata or {}
    return {
        "display_name": source.get("display_source", "知识库：未标注来源"),
        "location": source.get("source_path", ""),
        "type": source.get("source_type", "local_knowledge"),
        "category": source.get("category", ""),
        "title": source.get("title", ""),
        "section": source.get("section", ""),
    }


def _vector_candidates(
    query: str,
    *,
    top_k: int,
    vector_store_dir: Path,
    collection_name: str,
    embedding_model: str | None,
) -> list[dict[str, Any]]:
    """执行语义召回，并保留 Chunk ID 与向量名次供后续融合使用。"""

    model_name = embedding_model or os.getenv(
        "DASHSCOPE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
    )
    query_embedding = embed_texts([query], model_name)[0]
    collection = _get_collection(vector_store_dir, collection_name)
    response = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = response.get("documents", [[]])[0] or []
    metadatas = response.get("metadatas", [[]])[0] or []
    distances = response.get("distances", [[]])[0] or []
    chunk_ids = response.get("ids", [[]])[0] or []
    candidates: list[dict[str, Any]] = []
    for index, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances, strict=True), start=1
    ):
        candidate = {
            "content": document,
            "source": _source_from_metadata(metadata),
            "vector_rank": index,
            # 越小越接近；它只代表语义检索距离，并非来源可信度。
            "retrieval_distance": distance,
        }
        if index <= len(chunk_ids):
            candidate["chunk_id"] = chunk_ids[index - 1]
        candidates.append(candidate)
    return candidates


def _bm25_candidates(
    query: str,
    *,
    top_k: int,
    vector_store_dir: Path,
    lexical_index_dir: Path,
    collection_name: str,
) -> list[dict[str, Any]]:
    """执行词法召回，再从 Chroma 按相同 Chunk ID 取回正文与来源。"""

    lexical_results = search_bm25(
        query,
        index_path=bm25_index_path(lexical_index_dir, collection_name),
        collection_name=collection_name,
        top_k=top_k,
    )
    if not lexical_results:
        return []

    collection = _get_collection(vector_store_dir, collection_name)
    lexical_ids = [result.chunk_id for result in lexical_results]
    lexical_data = collection.get(ids=lexical_ids, include=["documents", "metadatas"])
    lexical_lookup: dict[str, tuple[str, dict[str, Any] | None]] = {
        chunk_id: (document, metadata)
        for chunk_id, document, metadata in zip(
            lexical_data.get("ids", []),
            lexical_data.get("documents", []),
            lexical_data.get("metadatas", []),
            strict=True,
        )
    }
    return [
        {
            "chunk_id": result.chunk_id,
            "content": lexical_lookup[result.chunk_id][0],
            "source": _source_from_metadata(lexical_lookup[result.chunk_id][1]),
            "bm25_rank": result.rank,
            # 仅在本次 BM25 查询内有相对意义，不能与向量距离直接比较。
            "bm25_score": result.score,
        }
        for result in lexical_results
        if result.chunk_id in lexical_lookup
    ]


def _hybrid_local_retrieval(
    query: str,
    *,
    final_top_k: int,
    vector_store_dir: Path,
    lexical_index_dir: Path,
    collection_name: str,
    embedding_model: str | None,
) -> dict[str, Any]:
    """生产检索路径：双路召回 → RRF → Qwen Rerank，并提供受控回退。"""

    vector_candidates: list[dict[str, Any]] = []
    bm25_candidates: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        vector_candidates = _vector_candidates(
            query,
            top_k=VECTOR_CANDIDATE_TOP_K,
            vector_store_dir=vector_store_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
    except (FileNotFoundError, RuntimeError) as error:
        warnings.append(f"向量召回不可用：{error}")
    try:
        bm25_candidates = _bm25_candidates(
            query,
            top_k=BM25_CANDIDATE_TOP_K,
            vector_store_dir=vector_store_dir,
            lexical_index_dir=lexical_index_dir,
            collection_name=collection_name,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        warnings.append(f"BM25 召回不可用：{error}")

    if vector_candidates and bm25_candidates:
        fused_candidates = reciprocal_rank_fusion(
            vector_candidates,
            bm25_candidates,
            rank_constant=RRF_RANK_CONSTANT,
            top_k=FUSED_CANDIDATE_TOP_K,
        )
        try:
            reranked_candidates = rerank_fused_candidates(query, fused_candidates, top_k=final_top_k)
            return {
                "results": reranked_candidates,
                "retrieval_mode": "hybrid_rrf_rerank",
                "warnings": warnings,
            }
        except (RuntimeError, ValueError) as error:
            warnings.append(f"Rerank 不可用，已使用 RRF 排序：{error}")
            return {
                "results": fused_candidates[:final_top_k],
                "retrieval_mode": "hybrid_rrf_fallback",
                "warnings": warnings,
            }

    if vector_candidates:
        warnings.append("BM25 未提供候选，已回退到纯向量召回。")
        return {
            "results": vector_candidates[:final_top_k],
            "retrieval_mode": "vector_fallback",
            "warnings": warnings,
        }
    if bm25_candidates:
        warnings.append("向量召回未提供候选，已回退到纯 BM25 召回。")
        return {
            "results": bm25_candidates[:final_top_k],
            "retrieval_mode": "bm25_fallback",
            "warnings": warnings,
        }
    raise RuntimeError("本地知识库检索失败；" + "；".join(warnings or ["两路召回均无结果。"]))


def query_local_knowledge(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
) -> dict[str, Any]:
    """执行混合检索与精排，并返回适合传给 Agent 的结构化资料。"""

    clean_query = query.strip()
    if not clean_query:
        return {
            "status": "error",
            "tool": "search_local_knowledge",
            "message": "检索问题不能为空。",
            "results": [],
        }
    if top_k < 1:
        raise ValueError("top_k 必须至少为 1。")

    try:
        retrieval_result = _hybrid_local_retrieval(
            clean_query,
            final_top_k=top_k,
            vector_store_dir=vector_store_dir,
            lexical_index_dir=DEFAULT_LEXICAL_INDEX_DIR,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
    except (FileNotFoundError, RuntimeError) as error:
        return {
            "status": "error",
            "tool": "search_local_knowledge",
            "message": str(error),
            "results": [],
        }

    return {
        "status": "success",
        "tool": "search_local_knowledge",
        "query": clean_query,
        "result_count": len(retrieval_result["results"]),
        "retrieval_mode": retrieval_result["retrieval_mode"],
        "warnings": retrieval_result["warnings"],
        "results": retrieval_result["results"],
    }


def compare_local_retrievers(
    query: str,
    *,
    vector_top_k: int = VECTOR_CANDIDATE_TOP_K,
    bm25_top_k: int = BM25_CANDIDATE_TOP_K,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE,
    lexical_index_dir: Path = DEFAULT_LEXICAL_INDEX_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str | None = None,
    include_rerank: bool = False,
) -> dict[str, Any]:
    """并排返回语义、BM25、RRF 结果；可选对 RRF 候选调用 Qwen 精排。"""

    clean_query = query.strip()
    if not clean_query:
        return {"status": "error", "message": "检索问题不能为空。"}

    vector_candidates = _vector_candidates(
        clean_query,
        top_k=vector_top_k,
        vector_store_dir=vector_store_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )
    bm25_candidates = _bm25_candidates(
        clean_query,
        top_k=bm25_top_k,
        vector_store_dir=vector_store_dir,
        lexical_index_dir=lexical_index_dir,
        collection_name=collection_name,
    )
    fused_candidates = reciprocal_rank_fusion(
        vector_candidates,
        bm25_candidates,
        rank_constant=RRF_RANK_CONSTANT,
        top_k=FUSED_CANDIDATE_TOP_K,
    )
    result: dict[str, Any] = {
        "status": "success",
        "query": clean_query,
        "vector_candidates": vector_candidates,
        "bm25_candidates": bm25_candidates,
        "fused_candidates": fused_candidates,
    }
    if include_rerank:
        try:
            result["reranked_candidates"] = rerank_fused_candidates(clean_query, fused_candidates)
            result["rerank_status"] = "success"
        except (RuntimeError, ValueError) as error:
            # Rerank 是精度增强层；失败时保留可用的 RRF 候选用于排障和后续回退。
            result["reranked_candidates"] = []
            result["rerank_status"] = "error"
            result["rerank_message"] = str(error)
    return result


@tool
def search_local_knowledge(query: str) -> dict[str, Any]:
    """查询内部大气科学知识库，获取与问题相关、可追溯的资料片段。

    适用于基础概念、机制、天气系统、气候现象等已有内部资料的问题。
    """

    return query_local_knowledge(query)
