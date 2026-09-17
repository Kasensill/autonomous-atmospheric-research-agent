from pathlib import Path

from atmospheric_research_agent.agent.tools import local_knowledge
from atmospheric_research_agent.agent.retrieval.bm25 import BM25SearchResult


class FakeCollection:
    def query(self, **kwargs: object) -> dict[str, list[list[object]]]:
        assert kwargs["n_results"] == 4
        return {
            "documents": [["副热带高压控制区常见下沉运动。"]],
            "metadatas": [
                [
                    {
                        "display_source": "知识库：环流 > 副热带高压 > 影响",
                        "source_path": "03_环流/副热带高压.md",
                        "source_type": "local_knowledge",
                        "category": "03_环流",
                        "title": "副热带高压",
                        "section": "影响",
                    }
                ]
            ],
            "distances": [[0.31]],
        }


def test_query_local_knowledge_returns_traceable_results(monkeypatch) -> None:
    monkeypatch.setattr(
        local_knowledge,
        "_hybrid_local_retrieval",
        lambda *args, **kwargs: {
            "results": [
                {
                    "content": "副热带高压控制区常见下沉运动。",
                    "source": {
                        "display_name": "知识库：环流 > 副热带高压 > 影响",
                        "location": "03_环流/副热带高压.md",
                    },
                    "retrieval_distance": 0.31,
                }
            ],
            "retrieval_mode": "hybrid_rrf_rerank",
            "warnings": [],
        },
    )

    result = local_knowledge.query_local_knowledge(
        "副热带高压为什么高温少雨？",
        vector_store_dir=Path("unused"),
    )

    assert result["status"] == "success"
    assert result["result_count"] == 1
    assert result["results"][0]["source"]["location"] == "03_环流/副热带高压.md"
    assert result["results"][0]["retrieval_distance"] == 0.31
    assert result["retrieval_mode"] == "hybrid_rrf_rerank"


def test_query_local_knowledge_rejects_blank_query() -> None:
    result = local_knowledge.query_local_knowledge("   ")

    assert result["status"] == "error"
    assert result["results"] == []


def test_compare_local_retrievers_keeps_two_rankings_separate(monkeypatch) -> None:
    class ComparisonCollection:
        def get(self, **kwargs: object) -> dict[str, list[object]]:
            assert kwargs["ids"] == ["bm25_chunk"]
            return {
                "ids": ["bm25_chunk"],
                "documents": ["850 hPa 低空急流可加强水汽输送。"],
                "metadatas": [
                    {
                        "display_source": "知识库：天气系统 > 低空急流",
                        "source_path": "weather/jet.md",
                    }
                ],
            }

    monkeypatch.setattr(
        local_knowledge,
        "_vector_candidates",
        lambda *args, **kwargs: [
            {
                "chunk_id": "vector_chunk",
                "content": "向量候选。",
                "source": {"display_name": "知识库：向量候选"},
                "vector_rank": 1,
                "retrieval_distance": 0.2,
            }
        ],
    )
    monkeypatch.setattr(
        local_knowledge,
        "search_bm25",
        lambda *args, **kwargs: [BM25SearchResult("bm25_chunk", rank=1, score=3.5)],
    )
    monkeypatch.setattr(local_knowledge, "_get_collection", lambda *args: ComparisonCollection())

    result = local_knowledge.compare_local_retrievers("850 hPa 低空急流")

    assert result["status"] == "success"
    assert result["vector_candidates"][0]["chunk_id"] == "vector_chunk"
    assert result["bm25_candidates"][0]["chunk_id"] == "bm25_chunk"
    assert result["bm25_candidates"][0]["bm25_score"] == 3.5
    assert result["fused_candidates"][0]["fused_rank"] == 1


def test_compare_local_retrievers_keeps_rrf_when_rerank_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        local_knowledge,
        "_vector_candidates",
        lambda *args, **kwargs: [
            {
                "chunk_id": "shared_chunk",
                "content": "候选正文。",
                "source": {"display_name": "知识库：候选"},
                "vector_rank": 1,
                "retrieval_distance": 0.2,
            }
        ],
    )
    monkeypatch.setattr(
        local_knowledge,
        "search_bm25",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(local_knowledge, "_get_collection", lambda *args: object())
    monkeypatch.setattr(
        local_knowledge,
        "rerank_fused_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rerank 不可用")),
    )

    result = local_knowledge.compare_local_retrievers("测试", include_rerank=True)

    assert result["rerank_status"] == "error"
    assert "Rerank 不可用" in result["rerank_message"]
    assert result["fused_candidates"][0]["chunk_id"] == "shared_chunk"


def test_hybrid_retrieval_uses_reranked_results_when_all_layers_work(monkeypatch) -> None:
    vector_candidate = {
        "chunk_id": "shared_chunk",
        "content": "低空急流可输送水汽。",
        "source": {"display_name": "知识库：急流"},
        "vector_rank": 1,
        "retrieval_distance": 0.2,
    }
    bm25_candidate = {
        "chunk_id": "shared_chunk",
        "content": "低空急流可输送水汽。",
        "source": {"display_name": "知识库：急流"},
        "bm25_rank": 1,
        "bm25_score": 5.0,
    }
    monkeypatch.setattr(local_knowledge, "_vector_candidates", lambda *args, **kwargs: [vector_candidate])
    monkeypatch.setattr(local_knowledge, "_bm25_candidates", lambda *args, **kwargs: [bm25_candidate])
    monkeypatch.setattr(
        local_knowledge,
        "rerank_fused_candidates",
        lambda query, candidates, top_k: [
            {**candidates[0], "retrieval": {"rerank_rank": 1, "rerank_score": 0.9}}
        ],
    )

    result = local_knowledge._hybrid_local_retrieval(
        "低空急流",
        final_top_k=4,
        vector_store_dir=Path("unused"),
        lexical_index_dir=Path("unused"),
        collection_name="test",
        embedding_model=None,
    )

    assert result["retrieval_mode"] == "hybrid_rrf_rerank"
    assert result["results"][0]["retrieval"]["rerank_score"] == 0.9


def test_hybrid_retrieval_falls_back_to_vector_when_bm25_fails(monkeypatch) -> None:
    vector_candidate = {
        "chunk_id": "vector_chunk",
        "content": "向量候选。",
        "source": {"display_name": "知识库：向量"},
        "vector_rank": 1,
        "retrieval_distance": 0.2,
    }
    monkeypatch.setattr(local_knowledge, "_vector_candidates", lambda *args, **kwargs: [vector_candidate])
    monkeypatch.setattr(
        local_knowledge,
        "_bm25_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("BM25 不存在")),
    )

    result = local_knowledge._hybrid_local_retrieval(
        "测试",
        final_top_k=4,
        vector_store_dir=Path("unused"),
        lexical_index_dir=Path("unused"),
        collection_name="test",
        embedding_model=None,
    )

    assert result["retrieval_mode"] == "vector_fallback"
    assert "BM25 不存在" in result["warnings"][0]


def test_hybrid_retrieval_falls_back_to_rrf_when_rerank_fails(monkeypatch) -> None:
    vector_candidate = {
        "chunk_id": "shared_chunk",
        "content": "候选正文。",
        "source": {"display_name": "知识库：候选"},
        "vector_rank": 1,
        "retrieval_distance": 0.2,
    }
    bm25_candidate = {
        "chunk_id": "shared_chunk",
        "content": "候选正文。",
        "source": {"display_name": "知识库：候选"},
        "bm25_rank": 1,
        "bm25_score": 5.0,
    }
    monkeypatch.setattr(local_knowledge, "_vector_candidates", lambda *args, **kwargs: [vector_candidate])
    monkeypatch.setattr(local_knowledge, "_bm25_candidates", lambda *args, **kwargs: [bm25_candidate])
    monkeypatch.setattr(
        local_knowledge,
        "rerank_fused_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("Rerank 超时")),
    )

    result = local_knowledge._hybrid_local_retrieval(
        "测试",
        final_top_k=4,
        vector_store_dir=Path("unused"),
        lexical_index_dir=Path("unused"),
        collection_name="test",
        embedding_model=None,
    )

    assert result["retrieval_mode"] == "hybrid_rrf_fallback"
    assert "Rerank 超时" in result["warnings"][0]
