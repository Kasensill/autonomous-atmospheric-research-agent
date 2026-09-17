"""将大气科学 Markdown 知识库写入本地 ChromaDB。

本模块只负责离线入库，不参与在线 Agent 的工具选择或报告生成。
每个 Chunk 保留知识库路径、分类、主题和小节，使检索结果可直接作为
``[知识库：分类 > 主题 > 小节]`` 形式的本地来源。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from ..agent.retrieval.bm25 import bm25_index_path, write_bm25_index


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "data" / "knowledge_base"
DEFAULT_VECTOR_STORE = PROJECT_ROOT / "data" / "vector_store"
DEFAULT_LEXICAL_INDEX_DIR = PROJECT_ROOT / "data" / "retrieval_store"
DEFAULT_COLLECTION_NAME = "atmospheric_knowledge"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
MAX_CHUNK_CHARACTERS = 1_200
CHUNK_OVERLAP_CHARACTERS = 120
EMBEDDING_BATCH_SIZE = 10
# 这些内容用于人类追溯原始 Markdown，不是能直接支持研究结论的知识片段。
NON_RETRIEVABLE_SECTION_TITLES = frozenset({"资料来源", "参考资料", "参考文献"})


@dataclass(frozen=True)
class KnowledgeChunk:
    """一个可检索、可追溯的本地知识片段。"""

    chunk_id: str
    text: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class IndexSummary:
    """一次入库运行的简要结果。"""

    document_count: int
    chunk_count: int
    collection_name: str
    dry_run: bool
    bm25_index_path: Path | None = None


def find_markdown_files(knowledge_base_dir: Path) -> list[Path]:
    """按稳定顺序找到知识库中的所有 Markdown 文件。"""
    return sorted(path for path in knowledge_base_dir.rglob("*.md") if path.is_file())


def _split_long_text(text: str) -> list[str]:
    """将异常长的小节按字符窗口切开，并保留少量重叠上下文。"""
    if len(text) <= MAX_CHUNK_CHARACTERS:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARACTERS, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start + MAX_CHUNK_CHARACTERS // 2:
                end = newline
        parts.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP_CHARACTERS, start + 1)
    return [part for part in parts if part]


def _make_chunk_id(relative_path: Path, section_index: int, part_index: int) -> str:
    raw_id = f"{relative_path.as_posix()}::{section_index}::{part_index}"
    return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]


def _is_retrievable_section(section_name: str) -> bool:
    """判断小节是否应进入检索候选池，而非仅作为原文的书目说明。"""

    normalized_title = "".join(section_name.split())
    return normalized_title not in NON_RETRIEVABLE_SECTION_TITLES


def split_markdown_file(file_path: Path, knowledge_base_dir: Path) -> list[KnowledgeChunk]:
    """按一级和二级标题切分一个知识文档，并保留可引用的层级元数据。"""
    relative_path = file_path.relative_to(knowledge_base_dir)
    category = relative_path.parts[0] if len(relative_path.parts) > 1 else "未分类"
    lines = file_path.read_text(encoding="utf-8").splitlines()

    title = file_path.stem
    section = "概述"
    sections: list[tuple[str, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        if content and _is_retrievable_section(section):
            sections.append((section, content))
        buffer.clear()

    for line in lines:
        if line.startswith("# "):
            flush()
            title = line[2:].strip() or title
            section = "概述"
        elif line.startswith("## "):
            flush()
            section = line[3:].strip() or "概述"
        else:
            buffer.append(line)
    flush()

    chunks: list[KnowledgeChunk] = []
    for section_index, (section_name, content) in enumerate(sections):
        for part_index, part in enumerate(_split_long_text(content)):
            display_source = f"知识库：{category} > {title} > {section_name}"
            text = f"# {title}\n## {section_name}\n{part}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=_make_chunk_id(relative_path, section_index, part_index),
                    text=text,
                    metadata={
                        "source_type": "local_knowledge",
                        "source_path": relative_path.as_posix(),
                        "category": category,
                        "title": title,
                        "section": section_name,
                        "display_source": display_source,
                    },
                )
            )
    return chunks


def load_knowledge_chunks(knowledge_base_dir: Path = DEFAULT_KNOWLEDGE_BASE) -> list[KnowledgeChunk]:
    """读取整个知识库并返回所有可检索 Chunk。"""
    if not knowledge_base_dir.exists():
        raise FileNotFoundError(f"知识库目录不存在：{knowledge_base_dir}")

    chunks: list[KnowledgeChunk] = []
    for file_path in find_markdown_files(knowledge_base_dir):
        chunks.extend(split_markdown_file(file_path, knowledge_base_dir))
    if not chunks:
        raise ValueError(f"知识库中没有可入库的 Markdown 内容：{knowledge_base_dir}")
    return chunks


def embed_texts(texts: Iterable[str], model_name: str) -> list[list[float]]:
    """使用 DashScope/Qwen 将文本批量向量化。"""
    try:
        import dashscope
    except ImportError as error:  # pragma: no cover - 依赖安装问题
        raise RuntimeError("未安装 dashscope，请先安装项目依赖。") from error

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，请在项目根目录 .env 中配置。")
    dashscope.api_key = api_key

    text_list = list(texts)
    embeddings: list[list[float]] = []
    for start in range(0, len(text_list), EMBEDDING_BATCH_SIZE):
        batch = text_list[start : start + EMBEDDING_BATCH_SIZE]
        response = dashscope.TextEmbedding.call(model=model_name, input=batch)
        if response.status_code != 200:
            raise RuntimeError(
                f"Qwen Embedding 请求失败（第 {start // EMBEDDING_BATCH_SIZE + 1} 批）：{response.message}"
            )
        embeddings.extend(item["embedding"] for item in response.output["embeddings"])
    return embeddings


def write_to_chroma(
    chunks: list[KnowledgeChunk],
    embeddings: list[list[float]],
    vector_store_dir: Path,
    collection_name: str,
    reset_collection: bool,
) -> None:
    """将 Chunk 和向量写入项目自己的 ChromaDB。"""
    try:
        import chromadb
    except ImportError as error:  # pragma: no cover - 依赖安装问题
        raise RuntimeError("未安装 chromadb，请先安装项目依赖。") from error

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk 数量与 Embedding 数量不一致。")

    vector_store_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(vector_store_dir))
    if reset_collection:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(name=collection_name)
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=embeddings,
    )


def build_index(
    knowledge_base_dir: Path = DEFAULT_KNOWLEDGE_BASE,
    vector_store_dir: Path = DEFAULT_VECTOR_STORE,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    dry_run: bool = False,
    reset_collection: bool = False,
    lexical_index_dir: Path = DEFAULT_LEXICAL_INDEX_DIR,
) -> IndexSummary:
    """执行完整入库流程；dry_run 只验证资料和切分，不请求 API。"""
    files = find_markdown_files(knowledge_base_dir)
    chunks = load_knowledge_chunks(knowledge_base_dir)
    if not dry_run:
        embeddings = embed_texts((chunk.text for chunk in chunks), embedding_model)
        write_to_chroma(chunks, embeddings, vector_store_dir, collection_name, reset_collection)
        write_bm25_index(
            chunks,
            index_path=bm25_index_path(lexical_index_dir, collection_name),
            collection_name=collection_name,
        )
    return IndexSummary(
        document_count=len(files),
        chunk_count=len(chunks),
        collection_name=collection_name,
        dry_run=dry_run,
        bm25_index_path=None if dry_run else bm25_index_path(lexical_index_dir, collection_name),
    )


def refresh_bm25_index(
    knowledge_base_dir: Path = DEFAULT_KNOWLEDGE_BASE,
    lexical_index_dir: Path = DEFAULT_LEXICAL_INDEX_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> IndexSummary:
    """只重建 BM25 索引，不请求 Embedding API，也不改动已有 Chroma 数据。"""

    files = find_markdown_files(knowledge_base_dir)
    chunks = load_knowledge_chunks(knowledge_base_dir)
    index_path = bm25_index_path(lexical_index_dir, collection_name)
    write_bm25_index(chunks, index_path=index_path, collection_name=collection_name)
    return IndexSummary(
        document_count=len(files),
        chunk_count=len(chunks),
        collection_name=collection_name,
        dry_run=False,
        bm25_index_path=index_path,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="将本地大气科学知识库写入 ChromaDB。")
    parser.add_argument("--dry-run", action="store_true", help="只检查资料读取与切分，不调用 API。")
    parser.add_argument("--reset", action="store_true", help="写入前删除同名 Chroma collection。")
    parser.add_argument(
        "--refresh-bm25",
        action="store_true",
        help="只重建 BM25 词法索引；不调用 Embedding API，也不改动 Chroma。",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME, help="Chroma collection 名称。")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("DASHSCOPE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="DashScope Embedding 模型名称。",
    )
    arguments = parser.parse_args()
    if arguments.refresh_bm25:
        if arguments.dry_run or arguments.reset:
            parser.error("--refresh-bm25 不能与 --dry-run 或 --reset 同时使用。")
        summary = refresh_bm25_index(collection_name=arguments.collection)
        mode = "BM25 索引刷新完成"
    else:
        summary = build_index(
            collection_name=arguments.collection,
            embedding_model=arguments.embedding_model,
            dry_run=arguments.dry_run,
            reset_collection=arguments.reset,
        )
        mode = "检查完成" if summary.dry_run else "入库完成"
    print(f"{mode}：{summary.document_count} 篇文档，{summary.chunk_count} 个 Chunk。")
    if summary.bm25_index_path:
        print(f"BM25 索引：{summary.bm25_index_path}")


if __name__ == "__main__":
    main()
