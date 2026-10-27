"""进程内单任务管理；接口只读取快照，不并发访问执行中的图。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from threading import RLock
import traceback
from typing import Any, Callable
from uuid import uuid4

from ..agent.settings import MAX_CLARIFICATIONS, MAX_RESEARCH_ROUNDS, MAX_TOOL_CALLS
from .engine import ResearchSession

LOGGER = logging.getLogger(__name__)
MAX_RECORDS = 20
PUBLIC_FIELDS = frozenset({
    'run_id', 'question', 'started_at', 'finished_at', 'run_duration_ms',
    'research_round', 'tool_call_count', 'clarification_history', 'evidence',
    'report_review', 'report_revision_count', 'report_mode', 'final_report',
    'program_limitations',
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


class RunManager:
    """一把锁保护准入及发布；锁内不运行模型，不等待 Future。"""

    def __init__(self, *, session_factory: Callable = ResearchSession, max_records: int = MAX_RECORDS):
        if max_records < 1:
            raise ValueError('max_records 必须大于零。')
        self._factory, self._max_records = session_factory, max_records
        self._lock = RLock()
        self._records: dict[str, RunRecord] = {}
        self._keys: dict[str, str] = {}
        self._active: str | None = None
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='research')

    @property
    def available(self) -> bool:
        with self._lock:
            return not self._closed

    def _record(self, run_id: str) -> RunRecord:
        if run_id not in self._records:
            raise ServiceError(404,'run_not_found','任务不存在或已被清理。',run_id)
        return self._records[run_id]

    def _touch(self, record: RunRecord) -> None:
        record.version += 1
        record.updated_at = now()

    def _view(self, record: RunRecord) -> dict:
        return deepcopy({
            'run_id':record.run_id,'question':record.question,'status':record.status,
            'delivery_type':record.delivery_type,'version':record.version,'updated_at':record.updated_at,
            'clarification':record.clarification,'error':record.error,'elapsed_ms':record.elapsed_ms,
            'research_round':record.state.get('research_round',0),
            'tool_call_count':record.state.get('tool_call_count',0),
            'clarification_count':len(record.state.get('clarification_history',[])),
            'limits':{'research_rounds':MAX_RESEARCH_ROUNDS,'tool_calls':MAX_TOOL_CALLS,
                      'clarifications':MAX_CLARIFICATIONS},
            'latest_event':record.trace[-1] if record.trace else None,
        })

    def get(self, run_id: str) -> dict:
        with self._lock:
            return self._view(self._record(run_id))

    def list_runs(self) -> list[dict]:
        with self._lock:
            return [self._view(r) for r in reversed(list(self._records.values()))]

    def _ensure_open(self) -> None:
        if self._closed:
            raise ServiceError(503,'service_unavailable','服务正在关闭或执行设施不可用。')

    def _prune(self) -> None:
        while len(self._records) >= self._max_records:
            victim = next((r for r in self._records.values()
                           if r.status in {'completed','failed','abandoned'}),None)
            if victim is None:
                raise ServiceError(409,'task_busy','请先完成或放弃当前等待中的任务。',self._active)
            del self._records[victim.run_id]
            self._keys.pop(victim.key,None)

    def create(self, question: str, key: str) -> dict:
        question = question.strip()
        if not question or len(question)>10000 or not key or len(key)>128:
            raise ServiceError(422,'invalid_request','问题或请求标识无效。')
        with self._lock:
            self._ensure_open()
            if key in self._keys:
                record=self._record(self._keys[key])
                if record.question != question:
                    raise ServiceError(409,'idempotency_conflict','同一请求标识不能用于不同问题。',record.run_id)
                return self._view(record)
            if self._active is not None:
                raise ServiceError(409,'task_busy','请先完成或放弃当前等待中的任务。',self._active)
            self._prune()
            record=RunRecord(str(uuid4()),question,key)
            record.state={'run_id':record.run_id,'question':question}
            self._records[record.run_id]=record
            self._keys[key]=record.run_id
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
            record.state=deepcopy({k:v for k,v in state.items() if k in PUBLIC_FIELDS})
            record.trace=deepcopy(state.get('trace',[]))
            self._touch(record)

    def _fail(self, record: RunRecord, error: Exception) -> None:
        error_id=str(uuid4())
        # 不记录异常消息、局部变量、正文或绝对路径；位置只取模块文件名和行号。
        frames=[{'file':f.filename.replace('\\','/').rsplit('/',1)[-1], 'line':f.lineno,
                 'function':f.name} for f in traceback.extract_tb(error.__traceback__)]
        LOGGER.error('research_error run_id=%s error_id=%s type=%s frames=%s',
                     record.run_id,error_id,type(error).__name__,frames)
        record.status='failed'
        record.delivery_type=None
        record.clarification=None
        record.error={'code':'execution_failed','message':'本轮研究执行失败，请按错误编号检查服务端日志。',
                      'run_id':record.run_id,'error_id':error_id}
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
            if record.engine is None:
                record.engine=self._factory(record.question,run_id=record.run_id,on_error=observed_errors.append)
            record.engine.on_error=observed_errors.append
            record.engine.advance(answer,on_progress=lambda state:self._publish(record,state))
            with self._lock:
                self._publish(record,record.engine.state)
                record.elapsed_ms=record.engine.elapsed_ms
                if record.engine.status=='waiting':
                    if self._closed:
                        raise RuntimeError('服务关闭')
                    record.status='waiting_user'
                    record.clarification={'question_id':str(uuid4()),'question':record.engine.question}
                elif record.engine.status=='completed':
                    record.status='completed'
                    record.delivery_type=('failure_notice' if any(t.get('node')=='finalize_limited_report'
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
            if record.status=='failed':
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
            self._closed=True
            for record in self._records.values():
                if record.status=='waiting_user':
                    record.status='abandoned';record.clarification=None
                    self._release(record);self._touch(record)
        # 不假装能杀死运行中的线程；进程退出仍可能等待底层调用。
        self._executor.shutdown(wait=False,cancel_futures=False)
