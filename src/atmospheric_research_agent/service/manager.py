"""SQLite任务档案与进程内单任务执行；接口不访问执行中的图。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
import logging
import sqlite3
from uuid import UUID
from threading import RLock
from pathlib import Path
import traceback
from typing import Any, Callable
from uuid import uuid4

from ..agent.settings import MAX_CLARIFICATIONS, MAX_RESEARCH_ROUNDS, MAX_TOOL_CALLS
from .engine import ResearchSession
from ..agent.model_trace import model_trace_scope, failure_metadata
from ..agent.tools.source_reader import source_read_scope
from ..agent.history import HistoryContext, resolve_archived_history, lookup_summary, _EXPLICIT_RUN_ID
from ..agent.settings import HISTORY_ARCHIVE_SCAN_LIMIT, HISTORY_TOPIC_REPORT_PREFIX
from .storage import SQLiteRunStore
from .uploads import MaterialUpload, UploadStore, upload_fingerprint, validate_uploads

LOGGER = logging.getLogger(__name__)
MAX_RECORDS = 20
PUBLIC_FIELDS = frozenset({
    'run_id', 'question', 'started_at', 'finished_at', 'run_duration_ms',
    'research_round', 'tool_call_count', 'clarification_history', 'evidence',
    'report_review', 'report_revision_count', 'report_mode', 'final_report',
    'program_limitations',
    'materials', 'material_role', 'conversation_context',
})


def now() -> str:
    return datetime.now(UTC).isoformat()


class ServiceError(Exception):
    def __init__(self, status_code: int, code: str, message: str,
                 run_id: str | None = None, error_id: str | None = None):
        super().__init__(message)
        self.status_code, self.code, self.message = status_code, code, message
        self.run_id, self.error_id = run_id, error_id

    def payload(self) -> dict:
        return {k:getattr(self,k) for k in ('code','message','run_id','error_id')}


@dataclass
class RunRecord:
    run_id: str
    question: str
    key: str
    status: str = 'running'
    delivery_type: str | None = None
    version: int = 1
    updated_at: str = field(default_factory=now)
    state: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    clarification: dict | None = None
    error: dict | None = None
    elapsed_ms: float = 0
    engine: Any = None
    material_paths: list = field(default_factory=list)
    material_names: list[str] = field(default_factory=list)
    material_role: str | None = None
    material_fingerprint: str | None = None
    conversation_id: str = ""
    parent_run_id: str | None = None
    conversation_context: list = field(default_factory=list)
    material_snapshot: dict | None = None
    resume_count: int = 0
    history_source_run_id: str | None = None
    history_lookup: dict | None = None


class RunManager:
    """一把锁保护准入及发布；锁内不运行模型，不等待 Future。"""

    def __init__(self, *, session_factory: Callable = ResearchSession, max_records: int = MAX_RECORDS,
                 upload_store: UploadStore | None = None, db_path: str | Path = ":memory:"):
        if max_records < 1:
            raise ValueError('max_records 必须大于零。')
        self._factory, self._max_records = session_factory, max_records
        self._upload_store = upload_store or UploadStore()
        self._lock = RLock()
        self._records: dict[str, RunRecord] = {}
        self._active: str | None = None
        self._closed = False
        self._storage_failed = False
        self._store_closed = False
        self._closing = False
        self._checkpoint_conn = None
        self._checkpointer = None
        try:
            self._store = SQLiteRunStore(db_path)
            if str(db_path) != ':memory:' and session_factory is ResearchSession:
                from langgraph.checkpoint.sqlite import SqliteSaver
                from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
                path = Path(db_path).resolve()
                self._checkpoint_conn = sqlite3.connect(path.with_suffix(path.suffix + '.checkpoints'),
                                                        check_same_thread=False)
                self._checkpointer = SqliteSaver(self._checkpoint_conn, serde=JsonPlusSerializer(allowed_msgpack_modules=None))
                self._checkpointer.setup()
            for payload in self._store.unfinished():
                record = RunRecord(**payload)
                self._interrupt(record)
        except Exception as error:
            if self._checkpoint_conn is not None:
                self._checkpoint_conn.close()
            if hasattr(self, '_store'):
                self._store.close()
            LOGGER.error('storage_startup_error type=%s', type(error).__name__)
            raise RuntimeError('任务数据库不可用或已被其他服务占用；请检查目录、数据库版本及服务实例。') from None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='research')

    @property
    def available(self) -> bool:
        with self._lock:
            return not self._closed

    def _storage_error(self, error: Exception) -> ServiceError:
        self._closed = self._storage_failed = True
        error_id = str(uuid4())
        LOGGER.error('storage_error error_id=%s type=%s', error_id, type(error).__name__)
        return ServiceError(503, 'storage_unavailable', '任务档案存储失败，服务已停止接收新任务；请检查服务日志。',
                            error_id=error_id)

    def _read(self, method: str, *args):
        if self._store_closed or self._storage_failed:
            raise ServiceError(503, 'storage_unavailable', '任务档案不可用，请检查服务状态。')
        try:
            return getattr(self._store, method)(*args)
        except Exception as error:
            raise self._storage_error(error) from None

    def _persist(self, record: RunRecord) -> None:
        if self._storage_failed:
            return
        # 原始材料路径、图、检查点、消息和模型客户端不进入任务档案。
        payload = {f.name: deepcopy(getattr(record, f.name)) for f in fields(record)
                   if f.name not in {'engine', 'material_paths'}}
        try:
            self._store.save(payload)
        except Exception as error:
            raise self._storage_error(error) from None

    def _record(self, run_id: str) -> RunRecord:
        if run_id in self._records:
            return self._records[run_id]
        payload = self._read('get', run_id)
        if payload is None:
            raise ServiceError(404,'run_not_found','任务不存在。',run_id)
        return RunRecord(**payload)

    def _touch(self, record: RunRecord) -> None:
        record.version += 1
        record.updated_at = now()
        self._persist(record)

    def _interrupt(self, record: RunRecord) -> None:
        record.status = 'interrupted'
        record.delivery_type = None
        record.clarification = None
        record.error = {'code': 'execution_interrupted',
                        'message': '研究执行中断：服务已停止或重启，可用的执行检查点将保留。',
                        'run_id': record.run_id, 'error_id': None}
        record.state['finished_at'] = now()
        record.state.pop('final_report', None)
        record.trace.append({'node': 'service', 'event': 'execution_interrupted', 'status': 'error',
                             'timestamp': now(), 'detail': record.error['message']})
        self._touch(record)

    def _view(self, record: RunRecord) -> dict:
        return deepcopy({
            'run_id':record.run_id,'question':record.question,'status':record.status,
            'conversation_id':record.conversation_id or record.run_id, 'parent_run_id':record.parent_run_id,
            'delivery_type':record.delivery_type,'version':record.version,'updated_at':record.updated_at,
            'resume_count': record.resume_count,
            'can_resume': record.status == 'interrupted' and self._has_checkpoint(record.run_id),
            'clarification':record.clarification,'error':record.error,'elapsed_ms':record.elapsed_ms,
            'research_round':record.state.get('research_round',0),
            'tool_call_count':record.state.get('tool_call_count',0),
            'clarification_count':len(record.state.get('clarification_history',[])),
            'limits':{'research_rounds':MAX_RESEARCH_ROUNDS,'tool_calls':MAX_TOOL_CALLS,
                      'clarifications':MAX_CLARIFICATIONS},
            'latest_event':record.trace[-1] if record.trace else None,
            'material_files': record.material_names,
            'material_role': record.state.get('material_role', record.material_role),
        })

    def get(self, run_id: str) -> dict:
        with self._lock:
            return self._view(self._record(run_id))

    def list_runs(self, limit: int = MAX_RECORDS, offset: int = 0,
                  conversation_id: str | None = None) -> list[dict]:
        from ..agent.settings import SERVICE_HISTORY_MAX_PAGE_SIZE
        if not 1 <= limit <= SERVICE_HISTORY_MAX_PAGE_SIZE or offset < 0:
            raise ServiceError(422, 'invalid_request', '历史列表分页参数无效。')
        with self._lock:
            return [self._view(RunRecord(**r)) for r in self._read('list', limit, offset, conversation_id)]

    def _ensure_open(self) -> None:
        if self._closed:
            raise ServiceError(503,'service_unavailable','服务正在关闭或执行设施不可用。')

    def _prune(self) -> None:
        while len(self._records) >= self._max_records:
            victim = next((r for r in self._records.values()
                           if r.status in {'completed','failed','abandoned','interrupted'}),None)
            if victim is None:
                raise ServiceError(409,'task_busy','请先完成或放弃当前等待中的任务。',self._active)
            del self._records[victim.run_id]

    def create(self, question: str, key: str, *, materials: list[MaterialUpload] | None = None,
               material_role: str = 'auto', parent_run_id: str | None = None) -> dict:
        question = question.strip()
        if not question or len(question)>10000 or not key or len(key)>128:
            raise ServiceError(422,'invalid_request','问题或请求标识无效。')
        if materials is not None:
            try:
                validate_uploads(materials, material_role)
            except ValueError as error:
                raise ServiceError(422, 'invalid_material', str(error)) from None
        fingerprint = upload_fingerprint(materials, material_role) if materials else None
        with self._lock:
            self._ensure_open()
            existing = self._read('by_key', key)
            if existing is not None:
                record = self._record(existing['run_id'])
                if (record.question != question or record.material_fingerprint != fingerprint
                        or record.parent_run_id != parent_run_id
                        or (parent_run_id and record.material_role != material_role)):
                    raise ServiceError(409,'idempotency_conflict','同一请求标识不能用于不同问题、材料或材料用途。',record.run_id)
                return self._view(record)
            if self._active is not None:
                raise ServiceError(409,'task_busy','请先完成或放弃当前等待中的任务。',self._active)
            parent = self._record(parent_run_id) if parent_run_id else None
            if parent and parent.status != 'completed':
                raise ServiceError(409, 'parent_not_completed', '只有已完成任务可以继续提问；澄清请回答原任务。', parent_run_id)
            if parent and materials:
                raise ServiceError(422, 'invalid_request', '继续提问复用原材料；更换材料请新建研究。')
            record=RunRecord(str(uuid4()),question,key)
            record.conversation_id = (parent.conversation_id or parent.run_id) if parent else record.run_id
            if parent:
                from ..agent.settings import CONVERSATION_CONTEXT_MAX_CHARACTERS, CONVERSATION_CONTEXT_MAX_TURNS
                record.parent_run_id = parent.run_id
                record.material_role = material_role
                record.material_names = list(parent.material_names)
                record.material_snapshot = deepcopy({
                    'materials': parent.state.get('materials', []),
                    'evidence': [e for e in parent.state.get('evidence', []) if e.get('source_type') == 'user_material'],
                }) if parent.material_names else None
                history = [*parent.conversation_context, {
                    'run_id': parent.run_id, 'question': parent.question,
                    'clarifications': parent.state.get('clarification_history', []),
                    'report': parent.state.get('final_report', ''), 'delivery_type': parent.delivery_type,
                }][-CONVERSATION_CONTEXT_MAX_TURNS:]
                # 按最近轮次优先保留；JSON截断只发生在文本数据中，不形成新证据。
                import json
                remaining = CONVERSATION_CONTEXT_MAX_CHARACTERS
                selected = []
                for item in reversed(history):
                    text = item['text'] if 'text' in item else json.dumps(item, ensure_ascii=False)
                    if remaining <= 0:
                        break
                    selected.insert(0, {'run_id': item['run_id'], 'text': text[:remaining],
                                        'truncated': item.get('truncated', False) or len(text) > remaining})
                    remaining -= min(len(text), remaining)
                record.conversation_context = selected
            self._prune()
            if materials:
                try:
                    record.material_paths = self._upload_store.save(record.run_id, materials)
                except OSError as error:
                    error_id = str(uuid4())
                    LOGGER.error('material_save_error error_id=%s type=%s', error_id, type(error).__name__)
                    raise ServiceError(500, 'material_save_failed', '材料保存失败，请按错误编号检查服务日志。',
                                       error_id=error_id) from None
                record.material_names = [item.name for item in materials]
                record.material_role = material_role
                record.material_fingerprint = fingerprint
            record.state={'run_id':record.run_id,'question':question}
            self._persist(record)
            self._records[record.run_id]=record
            self._active=record.run_id
            self._submit(record,None)
            return self._view(record)

    def _submit(self, record: RunRecord, answer: str | None) -> None:
        try:
            self._executor.submit(self._execute,record,answer)
        except Exception as error:
            self._fail(record,error)
            self._release(record)
            self._closed=True
            raise ServiceError(503,'service_unavailable','无法安排研究执行。',record.run_id,
                               record.error['error_id']) from None

    def _has_checkpoint(self, run_id: str) -> bool:
        if self._checkpointer is None or self._store_closed:
            return False
        try:
            return self._checkpointer.get_tuple({'configurable': {'thread_id': run_id}}) is not None
        except Exception as error:
            LOGGER.warning('checkpoint_read_error run_id=%s type=%s', run_id, type(error).__name__)
            return False

    def _history_context(self, record: RunRecord) -> HistoryContext:
        def lookup(request: dict, state: dict) -> dict:
            target_id = None
            if request["target"] == "run_id":
                try:
                    target_id = str(UUID(request["value"]))
                except ValueError:
                    target_id = None
                text = " ".join([state["question"], *(x["answer"] for x in state.get("clarification_history", []))])
                if not target_id or target_id not in {record.parent_run_id, *_EXPLICIT_RUN_ID.findall(text.lower())}:
                    return {"status": "not_found", "history_lookup": lookup_summary(request, not_found_reason="no_match")}
            with self._lock:
                try:
                    rows, counts = self._store.history_candidates(
                        record.run_id, HISTORY_ARCHIVE_SCAN_LIMIT, target_id=target_id,
                        topic=request["value"] if request["target"] == "topic" else None,
                        kind=request["kind"] if not target_id else None, reports_only=request["purpose"] == "support_research",
                        prefix_length=HISTORY_TOPIC_REPORT_PREFIX)
                    return resolve_archived_history(request, state, rows, counts, self._store.get)
                except (sqlite3.Error, ValueError) as error:
                    raise self._storage_error(error) from None
        return HistoryContext(lookup)

    def resume(self, run_id: str) -> dict:
        with self._lock:
            self._ensure_open()
            record = self._record(run_id)
            if record.status != 'interrupted':
                raise ServiceError(409, 'resume_conflict', '任务当前无法继续执行。', run_id)
            if self._active is not None:
                raise ServiceError(409, 'task_busy', '请先结束当前任务。', self._active)
            if not self._has_checkpoint(run_id):
                raise ServiceError(409, 'checkpoint_missing', '这项任务没有可用的执行检查点。', run_id)
            session = ResearchSession(record.question, run_id=run_id, checkpointer=self._checkpointer)
            session.history_context = self._history_context(record)
            try:
                session.restore()
            except ValueError:
                session.close()
                raise ServiceError(409, 'checkpoint_unusable', '执行检查点无法继续。', run_id) from None
            session.elapsed_ms = record.elapsed_ms
            if run_id not in self._records:
                self._prune()
            record.resume_count += 1
            record.engine = session
            record.error = None
            record.clarification = None
            record.status = 'running'
            # 从图检查点重新对齐节点轨迹，保留服务侧中断记录。
            record.trace = [*session.state.get('trace', []),
                            *(item for item in record.trace if item.get('node') == 'service')]
            self._publish(record, session.state)
            self._records[run_id] = record
            self._active = run_id
            self._submit(record, None)
            return self._view(record)

    def answer(self, run_id: str, question_id: str, answer: str) -> dict:
        if not isinstance(answer,str) or not answer.strip() or len(answer)>10000:
            raise ServiceError(422,'invalid_request','回答必须为非空文本且不超过长度限制。',run_id)
        with self._lock:
            self._ensure_open()
            record=self._record(run_id)
            if record.status!='waiting_user' or not record.clarification or record.clarification['question_id']!=question_id:
                raise ServiceError(409,'clarification_conflict','该问题已失效、已回答或任务不在等待中。',run_id)
            record.status='running'
            record.clarification=None
            self._touch(record)
            self._submit(record,answer.strip())
            return self._view(record)

    def _publish(self, record: RunRecord, state: dict) -> None:
        with self._lock:
            if record.status == 'interrupted':
                if self._checkpointer is not None:
                    raise RuntimeError('服务已停止')
                return
            record.state=deepcopy({k:v for k,v in state.items() if k in PUBLIC_FIELDS})
            graph_trace = state.get('trace', [])
            for entry in reversed(graph_trace):
                if entry.get("event") == "history_lookup" and "history_lookup" in entry:
                    record.history_lookup = deepcopy(entry["history_lookup"])
                    record.history_source_run_id = record.history_lookup["selected_run_id"]
                    break
            previous_count = sum(item.get('node') != 'service' for item in record.trace)
            record.trace.extend(deepcopy(graph_trace[previous_count:]))
            self._touch(record)

    def _publish_model_event(self, record: RunRecord, event: dict) -> None:
        with self._lock:
            if record.status == 'running' and not self._store_closed:
                record.trace.append(event)
                self._touch(record)

    def _fail(self, record: RunRecord, error: Exception) -> None:
        if record.status == 'interrupted':
            return
        error_id=str(uuid4())
        # 不记录异常消息、局部变量、正文或绝对路径；位置只取模块文件名和行号。
        frames=[{'file':f.filename.replace('\\','/').rsplit('/',1)[-1], 'line':f.lineno,
                 'function':f.name} for f in traceback.extract_tb(error.__traceback__)]
        metadata = failure_metadata(error)
        LOGGER.error('research_error run_id=%s error_id=%s diagnosis=%s frames=%s',
                     record.run_id, error_id, metadata, frames)
        record.status='failed'
        record.delivery_type=None
        record.clarification=None
        record.error={'code':'execution_failed','message':'本轮研究执行失败，请按错误编号检查服务端日志。',
                      'run_id':record.run_id,'error_id':error_id}
        if metadata["failure_kind"] != "model_error":
            status = metadata.get("http_status")
            reason = f"HTTP {status}" if status else metadata["failure_kind"]
            record.error["message"] = f"模型请求失败（{reason}），请按错误编号检查 Trace。"
        self._touch(record)

    def _release(self, record: RunRecord) -> None:
        if record.engine is not None:
            try:
                record.engine.close()
            except Exception as error:
                self._fail(record,error)
                self._closed=True
            record.engine=None
        if self._active==record.run_id:
            self._active=None

    def _execute(self, record: RunRecord, answer: str | None) -> None:
        observed_errors=[]
        try:
            with self._lock:
                if record.status == 'interrupted':
                    return
            if record.engine is None:
                options = ({'material_paths': record.material_paths, 'material_role': record.material_role}
                           if record.material_paths else {})
                if record.parent_run_id:
                    options.update(material_snapshot=record.material_snapshot, material_role=record.material_role,
                                   conversation_context=record.conversation_context)
                if self._checkpointer is not None:
                    options["checkpointer"] = self._checkpointer
                record.engine=self._factory(record.question,run_id=record.run_id,on_error=observed_errors.append, **options)
                if isinstance(record.engine, ResearchSession):
                    record.engine.history_context = self._history_context(record)
            with self._lock:
                if record.status == 'interrupted':
                    self._release(record)
                    return
            record.engine.on_error=observed_errors.append
            # 恢复到已有人工中断或已完成的检查点时，直接发布状态。
            if answer is not None or record.engine.status not in {'waiting', 'completed'}:
                publish_event = lambda event: self._publish_model_event(record, event)
                with model_trace_scope(publish_event), source_read_scope(
                    lambda: self._closed or record.status == 'interrupted', publish_event
                ):
                    record.engine.advance(answer,on_progress=lambda state:self._publish(record,state))
            with self._lock:
                if record.status == 'interrupted':
                    self._release(record)
                    return
                self._publish(record,record.engine.state)
                record.elapsed_ms=record.engine.elapsed_ms
                if record.engine.status=='waiting':
                    if self._closed:
                        raise RuntimeError('服务关闭')
                    record.status='waiting_user'
                    record.clarification={'question_id':str(uuid4()),'question':record.engine.question}
                elif record.engine.status=='completed':
                    record.status='completed'
                    record.delivery_type=('history_answer' if any(t.get('node') == 'answer_history' for t in record.trace)
                        else 'failure_notice' if any(t.get('node')=='finalize_limited_report'
                        for t in record.trace) else 'limited_report' if record.state.get('report_mode')=='limited'
                        else 'full_report')
                    self._release(record)
                else:
                    self._fail(record,observed_errors[-1] if observed_errors else RuntimeError('execution_failed'))
                    self._release(record)
                self._touch(record)
        except Exception as error:
            with self._lock:
                self._fail(record,error)
                self._release(record)

    def abandon(self, run_id: str) -> dict:
        with self._lock:
            record=self._record(run_id)
            if record.status!='waiting_user':
                raise ServiceError(409,'cannot_abandon','只能放弃等待回答的任务，运行中不支持强制取消。',run_id)
            record.status='abandoned'
            record.clarification=None
            self._release(record)
            self._touch(record)
            return self._view(record)

    def result(self, run_id: str) -> dict:
        with self._lock:
            record=self._record(run_id)
            if record.status in {'failed', 'interrupted'}:
                raise ServiceError(409,**record.error)
            if record.status=='abandoned':
                raise ServiceError(409,'task_abandoned','该任务已放弃，没有研究报告。',run_id)
            if record.status!='completed':
                raise ServiceError(409,'result_not_ready','研究尚未完成。',run_id)
            return deepcopy({'run_id':run_id,'status':record.status,'delivery_type':record.delivery_type,
                             'state':record.state})

    def trace(self, run_id: str, after: int=0, limit: int=100) -> dict:
        if after<0 or not 1<=limit<=500:
            raise ServiceError(422,'invalid_request','Trace游标或数量无效。',run_id)
        with self._lock:
            record=self._record(run_id)
            events=[{**deepcopy(e),'sequence':i+1} for i,e in enumerate(record.trace[after:after+limit],start=after)]
            return {'run_id':run_id,'events':events,'next_after':after+len(events),
                    'has_more':after+len(events)<len(record.trace)}

    def close(self) -> None:
        with self._lock:
            if self._store_closed or self._closing:
                return
            self._closing = True
            self._closed = True
        try:
            with self._lock:
                for record in self._records.values():
                    if record.status in {'running', 'waiting_user'}:
                        waiting = record.status == 'waiting_user'
                        self._interrupt(record)
                        if waiting:
                            self._release(record)
        finally:
            self._executor.shutdown(wait=self._checkpointer is not None, cancel_futures=False)
            with self._lock:
                if self._checkpoint_conn is not None:
                    self._checkpoint_conn.close()
                self._store.close()
                self._store_closed = True
