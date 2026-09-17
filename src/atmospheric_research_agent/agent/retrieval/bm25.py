"""中文 BM25 索引：与 Chroma 共享稳定的 Chunk ID。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Protocol

import jieba
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TERMS_PATH = PROJECT_ROOT / "data" / "retrieval_resources" / "atmospheric_terms.txt"
JIEBA_CACHE_DIR = PROJECT_ROOT / "data" / "retrieval_store" / "jieba_cache"
BM25_INDEX_SCHEMA_VERSION = 1

# 数值 + 单位、英文缩写与连字符术语要整体保留，而不是交给中文分词器拆散。
TECHNICAL_TOKEN_PATTERN = re.compile(
    r"(?:\d+(?:\.\d+)?\s*(?:hpa|km|mm|m/s|km/h|°c|℃|%)|[A-Za-z][A-Za-z0-9._-]*)",
    re.IGNORECASE,
)
MEANINGFUL_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


class ChunkLike(Protocol):
    """离线入库 Chunk 所需的最小字段，避免依赖具体 dataclass。"""

    chunk_id: str
    text: str


@dataclass(frozen=True)
class BM25SearchResult:
    """BM25 返回的一个候选及其原始词法相关性分数。"""

    chunk_id: str
    rank: int
    score: float


def _load_terms(terms_path: Path) -> tuple[str, ...]:
    """读取可人工维护的大气科学领域词表；空行与注释不参与分词。"""

    if not terms_path.exists():
        return ()
    return tuple(
        line.strip()
        for line in terms_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


@lru_cache(maxsize=8)
def _create_tokenizer(terms_path_text: str) -> Any:
    """创建独立 jieba 分词器并注入领域术语，不污染全局分词状态。"""

    tokenizer = jieba.Tokenizer()
    # 不使用系统临时目录的共享 jieba.cache，避免多个 Python 进程在 Windows 上
    # 同时初始化时互相占用缓存文件。该目录是项目运行数据，已被 Git 忽略。
    JIEBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.cache_file = str(JIEBA_CACHE_DIR / "atmospheric_terms.cache")
    for term in _load_terms(Path(terms_path_text)):
        tokenizer.add_word(term)
    return tokenizer


def _normalize_technical_token(token: str) -> str:
    """把 ``850 hPa``、``850hPa`` 统一为可精确匹配的 ``850hpa``。"""

    return re.sub(r"\s+", "", token).lower()


def _tokenize_chinese_segment(text: str, tokenizer: Any) -> list[str]:
    """对不含已保护术语的中文片段分词并去除纯标点。"""

    return [
        token.lower()
        for token in tokenizer.lcut(text)
        if token.strip() and MEANINGFUL_TOKEN_PATTERN.search(token)
    ]


def tokenize_for_bm25(text: str, terms_path: Path = DEFAULT_TERMS_PATH) -> list[str]:
    """中文分词，同时保留气象高度层、缩写和英文专业词等精确匹配项。"""

    clean_text = text.replace("\u3000", " ").strip()
    if not clean_text:
        return []

    tokenizer = _create_tokenizer(str(terms_path.resolve()))
    tokens: list[str] = []
    cursor = 0
    for match in TECHNICAL_TOKEN_PATTERN.finditer(clean_text):
        tokens.extend(_tokenize_chinese_segment(clean_text[cursor : match.start()], tokenizer))
        tokens.append(_normalize_technical_token(match.group(0)))
        cursor = match.end()
    tokens.extend(_tokenize_chinese_segment(clean_text[cursor:], tokenizer))
    return tokens


def bm25_index_path(index_dir: Path, collection_name: str) -> Path:
    """按 collection 名称隔离 BM25 文件，避免不同知识库混用。"""

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", collection_name)
    return index_dir / f"{safe_name}_bm25.json"


def write_bm25_index(
    chunks: Iterable[ChunkLike],
    *,
    index_path: Path,
    collection_name: str,
    terms_path: Path = DEFAULT_TERMS_PATH,
) -> None:
    """把 Chunk ID 与分词结果保存为可审阅的 JSON 索引，不重复保存正文。"""

    records = [
        {"chunk_id": chunk.chunk_id, "tokens": tokenize_for_bm25(chunk.text, terms_path)}
        for chunk in chunks
    ]
    if not records:
        raise ValueError("不能为零个 Chunk 建立 BM25 索引。")
    if any(not record["tokens"] for record in records):
        raise ValueError("存在分词后为空的 Chunk，无法建立可靠的 BM25 索引。")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        json.dumps(
            {
                "schema_version": BM25_INDEX_SCHEMA_VERSION,
                "collection_name": collection_name,
                "chunk_count": len(records),
                "terms_path": str(terms_path),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_bm25_index(index_path: Path, *, collection_name: str) -> tuple[list[str], BM25Okapi]:
    """读取 JSON 并在进程内重建 BM25；索引损坏时给出可理解错误。"""

    if not index_path.exists():
        raise FileNotFoundError(f"BM25 索引不存在：{index_path}。请先刷新本地索引。")
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != BM25_INDEX_SCHEMA_VERSION:
            raise ValueError("BM25 索引版本不兼容。")
        if payload.get("collection_name") != collection_name:
            raise ValueError("BM25 索引与当前 Chroma collection 不匹配。")
        records = payload["records"]
        chunk_ids = [record["chunk_id"] for record in records]
        tokenized_corpus = [record["tokens"] for record in records]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"BM25 索引无法读取：{error}") from error

    if not chunk_ids or len(chunk_ids) != len(tokenized_corpus):
        raise ValueError("BM25 索引没有有效的 Chunk 记录。")
    return chunk_ids, BM25Okapi(tokenized_corpus)


def search_bm25(
    query: str,
    *,
    index_path: Path,
    collection_name: str,
    top_k: int,
    terms_path: Path = DEFAULT_TERMS_PATH,
) -> list[BM25SearchResult]:
    """按词法相关性检索候选；后续由融合层决定如何与向量结果合并。"""

    if top_k < 1:
        raise ValueError("top_k 必须至少为 1。")
    query_tokens = tokenize_for_bm25(query, terms_path)
    if not query_tokens:
        return []

    chunk_ids, index = load_bm25_index(index_path, collection_name=collection_name)
    scored = [
        (chunk_id, float(score))
        for chunk_id, score in zip(chunk_ids, index.get_scores(query_tokens), strict=True)
        if score > 0
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [
        BM25SearchResult(chunk_id=chunk_id, rank=rank, score=score)
        for rank, (chunk_id, score) in enumerate(scored[:top_k], start=1)
    ]
