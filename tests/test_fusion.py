import pytest

from atmospheric_research_agent.agent.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_promotes_candidate_supported_by_both_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        [
            {"chunk_id": "vector_only", "vector_rank": 1, "retrieval_distance": 0.1},
            {"chunk_id": "shared", "vector_rank": 2, "retrieval_distance": 0.2},
        ],
        [
            {"chunk_id": "shared", "bm25_rank": 1, "bm25_score": 8.0},
            {"chunk_id": "bm25_only", "bm25_rank": 2, "bm25_score": 7.0},
        ],
        rank_constant=60,
        top_k=3,
    )

    assert [candidate["chunk_id"] for candidate in fused] == [
        "shared",
        "vector_only",
        "bm25_only",
    ]
    assert fused[0]["retrieval"] == {
        "vector_rank": 2,
        "retrieval_distance": 0.2,
        "bm25_rank": 1,
        "bm25_score": 8.0,
        "rrf_score": pytest.approx(1 / 62 + 1 / 61),
    }


def test_rrf_rejects_candidates_without_stable_ids() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        reciprocal_rank_fusion(
            [{"vector_rank": 1}],
            [],
            rank_constant=60,
            top_k=1,
        )
