"""SQLite 任务档案；只保存可展示状态，不保存图、模型消息或凭据。"""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from threading import RLock

from ..agent.settings import SERVICE_DATABASE_TIMEOUT_SECONDS


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
