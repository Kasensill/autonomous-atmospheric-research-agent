from pathlib import Path

from atmospheric_research_agent.ingestion.build_index import load_knowledge_chunks


def test_load_knowledge_chunks_preserves_markdown_hierarchy(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    category = knowledge_base / "01_基础"
    category.mkdir(parents=True)
    (category / "01_测试主题.md").write_text(
        "# 测试主题\n\n## 定义与背景\n第一段。\n\n## 形成机制\n第二段。\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(knowledge_base)

    assert len(chunks) == 2
    assert chunks[0].metadata["category"] == "01_基础"
    assert chunks[0].metadata["title"] == "测试主题"
    assert chunks[0].metadata["section"] == "定义与背景"
    assert chunks[0].metadata["source_path"] == "01_基础/01_测试主题.md"
    assert chunks[0].metadata["display_source"] == "知识库：01_基础 > 测试主题 > 定义与背景"
    assert "第一段。" in chunks[0].text


def test_chunk_ids_are_stable_for_the_same_document(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    (knowledge_base / "topic.md").write_text("# 主题\n\n## 概述\n内容。\n", encoding="utf-8")

    first_load = load_knowledge_chunks(knowledge_base)
    second_load = load_knowledge_chunks(knowledge_base)

    assert [chunk.chunk_id for chunk in first_load] == [chunk.chunk_id for chunk in second_load]


def test_source_sections_remain_in_markdown_but_are_not_retrievable_chunks(tmp_path: Path) -> None:
    knowledge_base = tmp_path / "knowledge_base"
    knowledge_base.mkdir()
    document_path = knowledge_base / "topic.md"
    document_path.write_text(
        "# 主题\n\n## 定义与背景\n可检索正文。\n\n## 资料来源\n- https://example.org\n",
        encoding="utf-8",
    )

    chunks = load_knowledge_chunks(knowledge_base)

    assert len(chunks) == 1
    assert chunks[0].metadata["section"] == "定义与背景"
    assert "## 资料来源" in document_path.read_text(encoding="utf-8")
