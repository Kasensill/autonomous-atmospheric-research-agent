"""服务契约离线测试；不访问模型、索引或长期记忆。"""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Barrier
from time import monotonic, sleep

import pytest

from atmospheric_research_agent.service.manager import RunManager, ServiceError


def eventually(manager, run_id, status):
    deadline = monotonic() + 5
    while monotonic() < deadline:
        value = manager.get(run_id)
        if value['status'] == status:
            return value
        sleep(.005)
    pytest.fail(f'任务未进入 {status}: {manager.get(run_id)}')


class FakeSession:
    def __init__(self, question, *, run_id, on_error=None):
        self.state = {'run_id':run_id,'question':question,'research_round':1,'tool_call_count':2,
                      'trace':[], 'clarification_history':[]}
        self.status, self.question, self.error, self.elapsed_ms = 'ready', '', '', 0
        self.closed = False
        self.calls = 0

    def advance(self, answer=None, on_progress=None):
        self.calls += 1
        if self.state['question'] == 'boom':
            raise RuntimeError('Bearer confidential-do-not-log')
        self.state['trace'].append({'node':'agent','event':'done','detail':'ok'})
        if on_progress:
            on_progress(self.state)
        if answer is None and self.state['question'] == 'clarify':
            self.status, self.question = 'waiting', '地点？'
        else:
            self.status = 'completed'
            self.state.update(final_report='离线报告',report_mode='limited')

    def close(self):
        self.closed = True


def test_parallel_creation_is_idempotent_and_content_conflict():
    gate = Event()
    made = []
    class Blocked(FakeSession):
        def advance(self, *args, **kwargs):
            assert gate.wait(5)
            super().advance(*args, **kwargs)
    def factory(*args, **kwargs):
        item = Blocked(*args, **kwargs); made.append(item); return item
    manager = RunManager(session_factory=factory)
    try:
        barrier = Barrier(2)
        def create():
            barrier.wait(); return manager.create('same', 'key')
        with ThreadPoolExecutor(2) as pool:
            a, b = list(pool.map(lambda _:create(), range(2)))
        assert a['run_id'] == b['run_id']
        with pytest.raises(ServiceError) as error:
            manager.create('different', 'key')
        assert error.value.code == 'idempotency_conflict'
        with pytest.raises(ServiceError) as error:
            manager.create('another', 'new')
        assert error.value.code == 'task_busy'
        gate.set(); eventually(manager, a['run_id'], 'completed')
        assert len(made) == 1 and made[0].closed
    finally:
        gate.set(); manager.close()


def test_parallel_answers_old_question_budget_and_trace():
    sessions = []
    def factory(*args, **kwargs):
        s=FakeSession(*args, **kwargs); sessions.append(s); return s
    manager = RunManager(session_factory=factory)
    try:
        item = manager.create('clarify','key')
        waiting = eventually(manager,item['run_id'],'waiting_user')
        qid = waiting['clarification']['question_id']
        assert manager.get(item['run_id'])['clarification']['question_id'] == qid
        with pytest.raises(ServiceError):
            manager.answer(item['run_id'], 'old', '北京')
        barrier = Barrier(2)
        def answer():
            barrier.wait()
            try:
                manager.answer(item['run_id'],qid,'北京'); return 202
            except ServiceError as error:
                return error.status_code
        with ThreadPoolExecutor(2) as pool:
            assert sorted(pool.map(lambda _:answer(),range(2))) == [202,409]
        done=eventually(manager,item['run_id'],'completed')
        assert done['delivery_type']=='limited_report'
        assert done['research_round']==1 and done['tool_call_count']==2
        assert sessions[0].calls == 2
        first=manager.trace(item['run_id'],0,1)
        second=manager.trace(item['run_id'],first['next_after'],100)
        assert first['events'][0]['sequence']==1
        assert second['events'][0]['sequence']==2
        result=manager.result(item['run_id']);result['state']['question']='tampered'
        assert manager.get(item['run_id'])['question']=='clarify'
    finally:
        manager.close()


def test_failure_releases_slot_no_secret_logging_and_retention(caplog):
    manager=RunManager(session_factory=FakeSession,max_records=1)
    try:
        item=manager.create('boom','one')
        failed=eventually(manager,item['run_id'],'failed')
        assert failed['error']['error_id']
        assert 'confidential' not in caplog.text
        assert failed['error']['error_id'] in caplog.text
        with pytest.raises(ServiceError) as error:
            manager.result(item['run_id'])
        assert error.value.code=='execution_failed'
        new=manager.create('ok','two');eventually(manager,new['run_id'],'completed')
        assert len(manager.list_runs())==1
        with pytest.raises(ServiceError) as error:
            manager.get(item['run_id'])
        assert error.value.status_code==404
    finally:
        manager.close()


def test_abandon_waiting_releases_graph_and_slot():
    sessions=[]
    def factory(*args, **kwargs):
        item=FakeSession(*args, **kwargs);sessions.append(item);return item
    manager=RunManager(session_factory=factory)
    try:
        item=manager.create('clarify','one'); waiting=eventually(manager,item['run_id'],'waiting_user')
        with pytest.raises(ServiceError) as error:
            manager.result(item['run_id'])
        assert error.value.code=='result_not_ready'
        result=manager.abandon(item['run_id'])
        assert result['status']=='abandoned' and result['delivery_type'] is None
        assert sessions[0].closed
        with pytest.raises(ServiceError):
            manager.answer(item['run_id'], waiting['clarification']['question_id'],'北京')
        new=manager.create('ok','two');eventually(manager,new['run_id'],'completed')
    finally:
        manager.close()


@pytest.mark.parametrize('mode,node,expected',[
    ('full','finalize_report','full_report'),
    ('limited','finalize_report','limited_report'),
    (None,'finalize_limited_report','failure_notice'),
])
def test_delivery_types_are_completed_not_execution_errors(mode,node,expected):
    class Ending(FakeSession):
        def advance(self,*args,**kwargs):
            self.status='completed'
            self.state.update(final_report='输出',report_mode=mode,
                              trace=[{'node':node,'event':'completed'}])
    manager=RunManager(session_factory=Ending)
    try:
        item=manager.create('ok','key')
        done=eventually(manager,item['run_id'],'completed')
        assert done['delivery_type']==expected and done['error'] is None
    finally:
        manager.close()


def test_running_cannot_be_abandoned_and_close_rejects_new():
    gate=Event()
    class Blocked(FakeSession):
        def advance(self,*args,**kwargs):
            assert gate.wait(5)
            super().advance(*args,**kwargs)
    manager=RunManager(session_factory=Blocked)
    try:
        item=manager.create('ok','key')
        with pytest.raises(ServiceError) as error:
            manager.abandon(item['run_id'])
        assert error.value.code=='cannot_abandon'
        manager.close()
        with pytest.raises(ServiceError) as error:
            manager.create('next','next')
        assert error.value.status_code==503
        gate.set()
        eventually(manager,item['run_id'],'completed')
    finally:
        gate.set();manager.close()


def test_second_clarification_has_new_id_old_answer_rejected():
    class Twice(FakeSession):
        def advance(self,*args,**kwargs):
            super().advance(*args,**kwargs)
            if self.calls<=2:
                self.status='waiting';self.question=f'问题{self.calls}'
    manager=RunManager(session_factory=Twice)
    try:
        item=manager.create('clarify','key')
        first=eventually(manager,item['run_id'],'waiting_user')['clarification']['question_id']
        manager.answer(item['run_id'],first,'one')
        second=eventually(manager,item['run_id'],'waiting_user')['clarification']['question_id']
        assert first!=second
        with pytest.raises(ServiceError):
            manager.answer(item['run_id'],first,'duplicate')
        manager.answer(item['run_id'],second,'two')
        eventually(manager,item['run_id'],'completed')
    finally:
        manager.close()
