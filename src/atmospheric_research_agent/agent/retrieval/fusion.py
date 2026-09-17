"""纯函数形式的 Reciprocal Rank Fusion（RRF）。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    vector_candidates: list[dict[str, Any]],
    bm25_candidates: list[dict[str, Any]],
    *,
    rank_constant: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """按 RRF 融合两条候选列表，保留每条候选来自哪一路的诊断信息。

    RRF(candidate) = Σ 1 / (rank_constant + 该候选在某一路的名次)
    它只比较名次；向量距离和 BM25 原始分数仍保留在结果中，但不会直接相加。
    """

    if rank_constant < 1:
        raise ValueError("RRF rank_constant 必须至少为 1。")
    if top_k < 1:
        raise ValueError("top_k 必须至少为 1。")

    scores: defaultdict[str, float] = defaultdict(float)
    candidates_by_id: dict[str, dict[str, Any]] = {}
    diagnostics: dict[str, dict[str, Any]] = defaultdict(dict)

    def add_candidates(
        candidates: list[dict[str, Any]], *, rank_key: str, score_key: str
    ) -> None:
        for candidate in candidates:
            chunk_id = candidate.get("chunk_id")
            rank = candidate.get(rank_key)
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("RRF 候选缺少稳定 chunk_id，无法与另一条检索路径对齐。")
            if not isinstance(rank, int) or rank < 1:
                raise ValueError(f"RRF 候选缺少有效 {rank_key}。")

            scores[chunk_id] += 1 / (rank_constant + rank)
            candidates_by_id.setdefault(chunk_id, candidate)
            diagnostics[chunk_id][rank_key] = rank
            if score_key in candidate:
                diagnostics[chunk_id][score_key] = candidate[score_key]

    add_candidates(vector_candidates, rank_key="vector_rank", score_key="retrieval_distance")
    add_candidates(bm25_candidates, rank_key="bm25_rank", score_key="bm25_score")

    ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    fused_candidates: list[dict[str, Any]] = []
    for fused_rank, chunk_id in enumerate(ordered_ids, start=1):
        candidate = dict(candidates_by_id[chunk_id])
        candidate["fused_rank"] = fused_rank
        candidate["retrieval"] = {
            **diagnostics[chunk_id],
            "rrf_score": scores[chunk_id],
        }
        fused_candidates.append(candidate)
    return fused_candidates
