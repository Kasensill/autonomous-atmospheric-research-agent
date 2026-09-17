from pathlib import Path
from types import SimpleNamespace

from atmospheric_research_agent.agent.retrieval.bm25 import (
    bm25_index_path,
    search_bm25,
    tokenize_for_bm25,
    write_bm25_index,
)
from atmospheric_research_agent.ingestion.build_index import refresh_bm25_index


def test_tokenize_for_bm25_preserves_scientific_terms(tmp_path: Path) -> None:
    terms_path = tmp_path / "terms.txt"
    terms_path.write_text("低空急流\n", encoding="utf-8")

    tokens = tokenize_for_bm25("850 hPa 低空急流会影响 ENSO 吗？", terms_path)

    assert "850hpa" in tokens
    assert "低空急流" in tokens
    assert "enso" in tokens


def test_bm25_index_returns_exact_term_match_first(tmp_path: Path) -> None:
    index_path = tmp_path / "atmospheric_knowledge_bm25.json"
    chunks = [
        SimpleNamespace(chunk_id="jet", text="850 hPa 低空急流可加强水汽输送。"),
        SimpleNamespace(chunk_id="typhoon", text="热带气旋需要暖海温和低风切变。"),
        SimpleNamespace(chunk_id="radar", text="天气雷达可以观测降水回波。"),
    ]
    write_bm25_index(chunks, index_path=index_path, collection_name="atmospheric_knowledge")

    results = search_bm25(
        "850 hPa 低空急流",
        index_path=index_path,
        collection_name="atmospheric_knowledge",
        top_k=2,
    )

    assert results[0].chunk_id == "jet"
    assert results[0].rank == 1
    assert results[0].score > 0


def test_refresh_bm25_index_does_not_need_embedding_api(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    category = knowledge_base / "01_基础"
    category.mkdir(parents=True)
    (category / "01_术语.md").write_text(
        "# 术语\n\n## 概述\n850 hPa 低空急流是重要的天气分析术语。\n",
        encoding="utf-8",
    )
    index_dir = tmp_path / "retrieval_store"

    summary = refresh_bm25_index(
        knowledge_base_dir=knowledge_base,
        lexical_index_dir=index_dir,
        collection_name="test_collection",
    )

    assert summary.chunk_count == 1
    assert summary.bm25_index_path == bm25_index_path(index_dir, "test_collection")
    assert summary.bm25_index_path.exists()
