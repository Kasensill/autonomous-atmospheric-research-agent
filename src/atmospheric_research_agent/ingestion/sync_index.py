"""备份后增量同步语料索引；先准备向量与 BM25，再写库并清理失效 ID。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from .build_index import (
    PROJECT_ROOT, DEFAULT_KNOWLEDGE_BASE, DEFAULT_VECTOR_STORE, DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL, DEFAULT_LEXICAL_INDEX_DIR, EMBEDDING_BATCH_SIZE,
    load_knowledge_chunks, embed_texts,
)
from ..agent.retrieval.bm25 import bm25_index_path, write_bm25_index


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sync_index() -> dict:
    import chromadb

    load_dotenv(PROJECT_ROOT / ".env")
    chunks = load_knowledge_chunks(DEFAULT_KNOWLEDGE_BASE)
    desired = {chunk.chunk_id: chunk for chunk in chunks}
    if len(desired) != len(chunks):
        raise ValueError("Chunk ID 重复，未写入索引。")
    directory = PROJECT_ROOT / "docs" / "validation" / datetime.now().astimezone().date().isoformat() / "corpus-expansion" / ("sync_" + uuid4().hex[:12])
    probe = (directory / "before.json").relative_to(PROJECT_ROOT).as_posix()
    if subprocess.run(["git", "check-ignore", "--quiet", "--no-index", "--", probe], cwd=PROJECT_ROOT).returncode:
        raise RuntimeError("备份目录未被忽略，停止索引更新。")
    directory.mkdir(parents=True, exist_ok=False)
    collection = chromadb.PersistentClient(path=str(DEFAULT_VECTOR_STORE)).get_collection(DEFAULT_COLLECTION_NAME)
    original = collection.get(include=["documents", "metadatas", "embeddings"])
    original["embeddings"] = original["embeddings"].tolist()
    save_json(directory / "before.json", original)
    previous = {key: i for i, key in enumerate(original["ids"])}
    lexical = bm25_index_path(DEFAULT_LEXICAL_INDEX_DIR, DEFAULT_COLLECTION_NAME)
    if not lexical.exists():
        raise RuntimeError("旧 BM25 不存在；本同步入口要求已有完整基线，未写库。")
    shutil.copy2(lexical, directory / "before_bm25.json")
    embeddings = {}
    changed = []
    for chunk in chunks:
        old = previous.get(chunk.chunk_id)
        if old is not None and original["documents"][old] == chunk.text:
            embeddings[chunk.chunk_id] = original["embeddings"][old]
        else:
            changed.append(chunk)
    model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    dimension = len(original["embeddings"][0]) if original["ids"] else None
    for start in range(0, len(changed), EMBEDDING_BATCH_SIZE):
        batch = changed[start:start + EMBEDDING_BATCH_SIZE]
        vectors = embed_texts([item.text for item in batch], model)
        if len(vectors) != len(batch):
            raise ValueError("向量条数与请求不匹配；旧索引未改动。")
        for chunk, vector in zip(batch, vectors, strict=True):
            if (dimension is not None and len(vector) != dimension) or not all(math.isfinite(x) for x in vector):
                raise ValueError("向量维度或数值异常；旧索引未改动。")
            embeddings[chunk.chunk_id] = vector
        save_json(directory / f"embeddings_{start:05d}.json", dict(zip([c.chunk_id for c in batch], vectors, strict=True)))
        print(f"Embedding 准备：{min(start + len(batch), len(changed))}/{len(changed)}", flush=True)
    staged_bm25 = directory / "next_bm25.json"
    write_bm25_index(chunks, index_path=staged_bm25, collection_name=DEFAULT_COLLECTION_NAME)
    obsolete = sorted(set(previous) - set(desired))
    try:
        for start in range(0, len(chunks), 50):
            batch = chunks[start:start + 50]
            collection.upsert(ids=[c.chunk_id for c in batch], documents=[c.text for c in batch],
                              metadatas=[c.metadata for c in batch], embeddings=[embeddings[c.chunk_id] for c in batch])
        if obsolete:
            collection.delete(ids=obsolete)
        # 同目录临时文件原子替换，避免查询读到半个 JSON；不重置集合。
        temporary = lexical.with_suffix(".next.json")
        shutil.copy2(staged_bm25, temporary)
        os.replace(temporary, lexical)
        current = collection.get(include=["documents", "metadatas"])
        lexical_data = json.loads(lexical.read_text(encoding="utf-8"))
        if set(current["ids"]) != set(desired) or {r["chunk_id"] for r in lexical_data["records"]} != set(desired):
            raise RuntimeError("两套索引 ID 集合不一致。")
        for key, document, metadata in zip(current["ids"], current["documents"], current["metadatas"], strict=True):
            if document != desired[key].text or metadata != desired[key].metadata:
                raise RuntimeError("写入后正文或 metadata 不一致。")
    except Exception:
        # 不调用 API；使用备份向量恢复原集合，删除本次新增 ID。
        for start in range(0, len(original["ids"]), 50):
            stop = start + 50
            collection.upsert(ids=original["ids"][start:stop], documents=original["documents"][start:stop],
                              metadatas=original["metadatas"][start:stop], embeddings=original["embeddings"][start:stop])
        added = sorted(set(desired) - set(previous))
        if added:
            collection.delete(ids=added)
        shutil.copy2(directory / "before_bm25.json", lexical)
        save_json(directory / "failure.json", {"status": "failed", "rollback": "completed"})
        raise
    summary = {
        "status": "completed", "collection": DEFAULT_COLLECTION_NAME,
        "documents": len(list(DEFAULT_KNOWLEDGE_BASE.rglob("*.md"))),
        "before_chunks": len(previous), "after_chunks": len(chunks),
        "embedded": len(changed), "reused": len(chunks) - len(changed), "obsolete_ids_removed": len(obsolete),
        "embedding_model": model, "embedding_dimension": dimension,
        "vector_and_bm25_ids_match": True, "documents_and_metadata_match": True,
        "corpus_sha256": hashlib.sha256("".join(c.chunk_id + c.text for c in chunks).encode()).hexdigest(),
        "backup_directory": directory.relative_to(PROJECT_ROOT).as_posix(),
    }
    save_json(directory / "summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(sync_index(), ensure_ascii=False, indent=2))
