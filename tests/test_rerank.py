import atmospheric_research_agent.agent.retrieval.rerank as rerank_module


def test_rerank_fused_candidates_preserves_routing_diagnostics(monkeypatch) -> None:
    candidates = [
        {
            "chunk_id": "first_input",
            "content": "第一个 RRF 候选。",
            "retrieval": {"fused_rank": 1, "rrf_score": 0.03},
        },
        {
            "chunk_id": "second_input",
            "content": "第二个 RRF 候选。",
            "retrieval": {"fused_rank": 2, "rrf_score": 0.02},
        },
    ]
    monkeypatch.setattr(
        rerank_module,
        "call_qwen_rerank",
        lambda query, documents, top_k, model: [
            {"index": 1, "relevance_score": 0.93},
            {"index": 0, "relevance_score": 0.61},
        ],
    )

    reranked = rerank_module.rerank_fused_candidates("测试问题", candidates, top_k=2)

    assert [item["chunk_id"] for item in reranked] == ["second_input", "first_input"]
    assert reranked[0]["retrieval"] == {
        "fused_rank": 2,
        "rrf_score": 0.02,
        "rerank_rank": 1,
        "rerank_score": 0.93,
    }


def test_call_qwen_rerank_normalizes_sdk_response(monkeypatch) -> None:
    class FakeTextReRank:
        @staticmethod
        def call(**kwargs):
            assert kwargs["model"] == "qwen3.7-text-rerank"
            assert kwargs["top_n"] == 2
            return {
                "status_code": 200,
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.4},
                    ]
                },
            }

    class FakeDashScope:
        TextReRank = FakeTextReRank

    monkeypatch.setattr(rerank_module, "_configure_dashscope", lambda: FakeDashScope())

    results = rerank_module.call_qwen_rerank(
        "测试问题",
        ["候选一", "候选二"],
        top_k=2,
    )

    assert results == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.4},
    ]
