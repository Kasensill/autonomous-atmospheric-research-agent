"""SQLite 任务档案；只保存可展示状态，不保存图、模型消息或凭据。"""
from __future__ import annotations

import json
from datetime import UTC, datetime
import unicodedata
from string import punctuation
from pathlib import Path
import sqlite3
from threading import RLock

from ..agent.settings import SERVICE_DATABASE_TIMEOUT_SECONDS


def _utc_timestamp(*values: str | None) -> str | None:
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        try:
            instant = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if instant.tzinfo is None:
                instant = instant.replace(tzinfo=UTC)
            return instant.astimezone(UTC).isoformat(timespec="microseconds")
        except (ValueError, OverflowError):
            continue
    return None


def _history_key(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKC", value if isinstance(value, str) else "").casefold()
    return "".join(ch for ch in normalized if not ch.isspace()
                   and ch not in punctuation and not unicodedata.category(ch).startswith("P"))


class SQLiteRunStore:
    """每个数据库只允许一个管理器持有；事务提交后才发布任务状态。"""

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path = ':memory:'):
        self._lock = RLock()
        self._owner = None
        self._connection = None
        try:
            if str(path) != ':memory:':
                path = Path(path).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                self._acquire_owner(path)
            self._connection = sqlite3.connect(str(path), timeout=SERVICE_DATABASE_TIMEOUT_SECONDS,
                                               check_same_thread=False)
            self._connection.create_function("history_utc", 3, _utc_timestamp, deterministic=True)
            self._connection.create_function("history_key", 1, _history_key, deterministic=True)
            version = self._connection.execute('PRAGMA user_version').fetchone()[0]
            if version not in {0, self.SCHEMA_VERSION}:
                raise ValueError('不支持此数据库版本；请保留原文件并使用兼容版本。')
            with self._connection:
                self._connection.execute('''CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, request_key TEXT NOT NULL UNIQUE,
                    conversation_id TEXT NOT NULL, status TEXT NOT NULL,
                    payload TEXT NOT NULL)''')
                self._connection.execute('CREATE INDEX IF NOT EXISTS runs_conversation ON runs(conversation_id)')
                self._connection.execute(f'PRAGMA user_version = {self.SCHEMA_VERSION}')
        except BaseException:
            self.close()
            raise

    def _acquire_owner(self, path: Path) -> None:
        # 操作系统锁随进程退出释放；锁文件存在不代表服务还在运行，不删除锁文件。
        self._owner = path.with_suffix(path.suffix + '.lock').open('a+b')
        self._owner.seek(0, 2)
        if self._owner.tell() == 0:
            self._owner.write(b'0')
            self._owner.flush()
        self._owner.seek(0)
        try:
            import msvcrt
            msvcrt.locking(self._owner.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            import fcntl
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def save(self, record: dict) -> None:
        payload = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._lock, self._connection:
            self._connection.execute('''INSERT INTO runs
                (run_id, request_key, conversation_id, status, payload) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET request_key=excluded.request_key,
                conversation_id=excluded.conversation_id, status=excluded.status, payload=excluded.payload''',
                (record['run_id'], record['key'], record['conversation_id'], record['status'], payload))

    def _one(self, column: str, value: str) -> dict | None:
        with self._lock:
            row = self._connection.execute(f'SELECT payload FROM runs WHERE {column}=?', (value,)).fetchone()
            return json.loads(row[0]) if row else None

    def get(self, run_id: str) -> dict | None:
        return self._one('run_id', run_id)

    def by_key(self, key: str) -> dict | None:
        return self._one('request_key', key)

    def list(self, limit: int, offset: int = 0, conversation_id: str | None = None) -> list[dict]:
        query, args = 'SELECT payload FROM runs', []
        if conversation_id is not None:
            query += ' WHERE conversation_id=?'
            args.append(conversation_id)
        query += ' ORDER BY rowid DESC LIMIT ? OFFSET ?'
        with self._lock:
            return [json.loads(row[0]) for row in self._connection.execute(query, (*args, limit, offset))]

    def history_candidates(self, current_id: str, limit: int, *, target_id: str | None,
                           topic: str | None, kind: str | None, reports_only: bool,
                           prefix_length: int) -> tuple[list[dict], dict]:
        scope = "run_id <> ?"
        args = [current_id]
        if target_id:
            scope += " AND run_id = ?"
            args.append(target_id)
        delivery = "json_extract(payload, '$.delivery_type')"
        body = "json_extract(payload, '$.state.final_report')"
        base = (f"status = 'completed' AND COALESCE({delivery}, '') <> 'failure_notice' "
                f"AND typeof({body}) = 'text' AND length(trim({body}, char(9)||char(10)||char(13)||' ')) > 0")
        if reports_only or kind == "report":
            base += f" AND {delivery} IN ('full_report', 'limited_report')"
        elif kind == "answer":
            base += f" AND {delivery} = 'history_answer'"
        key = _history_key(topic)
        sql = f"""SELECT run_id, {delivery} AS delivery_type,
                   json_extract(payload, '$.question') AS question,
                   json_extract(payload, '$.parent_run_id') AS parent_run_id,
                   history_utc(json_extract(payload, '$.state.finished_at'),
                               json_extract(payload, '$.updated_at'),
                               json_extract(payload, '$.state.started_at')) AS updated_at,
                   CASE WHEN ? IS NULL THEN 1 WHEN ? = '' THEN 0 ELSE
                     instr(history_key(json_extract(payload, '$.question')), ?) > 0 OR
                     instr(history_key(substr({body}, 1, ?)), ?) > 0 END AS topic_match
                   FROM runs WHERE {scope} AND {base}
                   ORDER BY updated_at DESC, run_id ASC LIMIT ?"""
        with self._lock:
            counts = self._connection.execute(
                f"""SELECT count(*), count(CASE WHEN status='completed' THEN 1 END),
                     count(CASE WHEN {base} THEN 1 END) FROM runs WHERE {scope}""", args).fetchone()
            cursor = self._connection.execute(sql, (topic, key, key, prefix_length, key, *args, limit))
            columns = [col[0] for col in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor]
        matches = [row for row in rows if row.pop("topic_match")]
        return matches, {"candidate_count_in_scope": counts[0], "completed_count": counts[1],
                         "candidate_count_after_base_filter": counts[2],
                         "candidate_count_after_topic": len(matches), "truncated": counts[2] > limit}

    def unfinished(self) -> list[dict]:
        with self._lock:
            return [json.loads(row[0]) for row in self._connection.execute(
                "SELECT payload FROM runs WHERE status IN ('running','waiting_user')")]

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._owner is not None:
            self._owner.close()
            self._owner = None
