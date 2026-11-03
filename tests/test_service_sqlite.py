"""SQLite 档案、重启中断及故障边界；全部使用替身模型和临时数据库。"""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Event
from time import monotonic, sleep

import pytest

from asgi_helpers import MemoryClient
from test_service import FakeSession, eventually
from test_service_api import fake_graph, wait
from atmospheric_research_agent.service import engine
from atmospheric_research_agent.service.api import create_app
from atmospheric_research_agent.service.manager import RunManager, ServiceError
from atmospheric_research_agent.service.uploads import MaterialUpload, UploadStore
from atmospheric_research_agent.ui.client import RemoteResearchTask, ResearchClient


class ArchivedSession(FakeSession):
    def __init__(self, *args, conversation_context=None, material_snapshot=None, material_role=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.state['conversation_context'] = conversation_context or []
        if material_snapshot:
            self.state.update(material_snapshot)


def test_completed_result_trace_and_idempotency_survive_reopen(tmp_path):
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=ArchivedSession)
    root = first.create('initial', 'same-key')
    eventually(first, root['run_id'], 'completed')
    result, trace, view = first.result(root['run_id']), first.trace(root['run_id']), first.get(root['run_id'])
    first.close()
    def forbidden(*a, **k):
        pytest.fail('只读历史和重放幂等请求不能执行模型')
    second = RunManager(db_path=db, session_factory=forbidden)
    try:
        assert second.result(root['run_id']) == result
        assert second.trace(root['run_id']) == trace
        assert second.get(root['run_id']) == view
        assert second.create('initial', 'same-key')['run_id'] == root['run_id']
        with pytest.raises(ServiceError) as error:
            second.create('changed', 'same-key')
        assert error.value.code == 'idempotency_conflict'
    finally:
        second.close()


def test_archived_parent_beyond_cache_reopens_for_followup(tmp_path):
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=ArchivedSession, max_records=1)
    root = first.create('initial', 'root')
    eventually(first, root['run_id'], 'completed')
    for i in range(3):
        view = first.create(f'other-{i}', f'other-{i}')
        eventually(first, view['run_id'], 'completed')
    assert len(first._records) == 1
    assert len(first.list_runs()) == 4
    first.close()
    second = RunManager(db_path=db, session_factory=ArchivedSession, max_records=1)
    try:
        child = second.create('继续', 'child', parent_run_id=root['run_id'])
        eventually(second, child['run_id'], 'completed')
        assert child['conversation_id'] == root['conversation_id']
        assert child['run_id'] != root['run_id']
        state = second.result(child['run_id'])['state']
        assert '离线报告' in state['conversation_context'][0]['text']
        assert state['research_round'] == 1 and state['tool_call_count'] == 2
        assert not state.get('evidence')
        assert second.result(root['run_id'])['state']['question'] == 'initial'
    finally:
        second.close()


def test_materials_resume_after_database_reopen_without_file_io(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'build_research_graph', fake_graph)
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, upload_store=UploadStore(tmp_path / 'uploads'))
    root = first.create('总结材料', 'root', materials=[MaterialUpload('原创.txt', '原创材料：甲是教学设定。'.encode())],
                        material_role='analysis')
    eventually(first, root['run_id'], 'completed')
    original = first.result(root['run_id'])['state']
    first.close()
    def forbidden(*a, **k):
        pytest.fail('继续提问不应重新读取或解析材料')
    monkeypatch.setattr('atmospheric_research_agent.agent.materials.load_materials', forbidden)
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    second = RunManager(db_path=db)
    try:
        child = second.create('核实材料说法', 'child', parent_run_id=root['run_id'], material_role='auto')
        eventually(second, child['run_id'], 'completed')
        state = second.result(child['run_id'])['state']
        assert state['materials'] == original['materials']
        assert state['material_role'] == 'clue'
        assert state['evidence'] and all(e['source_type'] == 'user_material' and e['role'] == 'clue'
                                         for e in state['evidence'])
        data = second._store.get(root['run_id'])
        assert 'engine' not in data and 'material_paths' not in data
        assert 'messages' not in data['state'] and 'recalled_memories' not in data['state']
        assert str(tmp_path) not in json.dumps(data, ensure_ascii=False)
    finally:
        second.close()


def test_waiting_close_becomes_interrupted_and_old_answers_are_rejected(tmp_path):
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=FakeSession)
    root = first.create('clarify', 'root')
    waiting = eventually(first, root['run_id'], 'waiting_user')
    first.close()
    second = RunManager(db_path=db, session_factory=FakeSession)
    try:
        view = second.get(root['run_id'])
        assert view['status'] == 'interrupted' and view['clarification'] is None
        assert view['delivery_type'] is None and view['tool_call_count'] == 2
        with pytest.raises(ServiceError) as error:
            second.answer(root['run_id'], waiting['clarification']['question_id'], 'old')
        assert error.value.code == 'clarification_conflict'
        with pytest.raises(ServiceError) as error:
            second.result(root['run_id'])
        assert error.value.code == 'execution_interrupted'
        with pytest.raises(ServiceError):
            second.create('follow', 'follow', parent_run_id=root['run_id'])
        assert second.create('clarify', 'root')['status'] == 'interrupted'
        assert sum(t['event'] == 'execution_interrupted' for t in second.trace(root['run_id'])['events']) == 1
    finally:
        second.close()
    third = RunManager(db_path=db)
    try:
        assert third.get(root['run_id'])['version'] == view['version']
    finally:
        third.close()


def test_late_worker_cannot_overwrite_interrupted_result(tmp_path):
    gate, returned = Event(), Event()
    class Blocked(FakeSession):
        def advance(self, *args, **kwargs):
            assert gate.wait(5)
            super().advance(*args, **kwargs)
        def close(self):
            super().close()
            returned.set()
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=Blocked)
    root = first.create('initial', 'root')
    # 保证线程已进入advance，测试关闭后真正晚到的模型替身结果。
    deadline = monotonic() + 5
    while first._records[root['run_id']].engine is None and monotonic() < deadline:
        sleep(.005)
    first.close()
    gate.set()
    assert returned.wait(5)
    second = RunManager(db_path=db, session_factory=FakeSession)
    try:
        assert second.get(root['run_id'])['status'] == 'interrupted'
        new = second.create('new', 'new')
        eventually(second, new['run_id'], 'completed')
    finally:
        second.close()


WORKER = '''
import os, sys, time
from pathlib import Path
from atmospheric_research_agent.service.manager import RunManager
class Session:
    def __init__(self, question, *, run_id, on_error=None):
        self.state={'run_id':run_id,'question':question,'trace':[],'research_round':2,'tool_call_count':3}
        self.status='ready';self.question='地点？';self.elapsed_ms=0
    def advance(self, answer=None, on_progress=None):
        self.state['trace']=[{'event':'before_crash','node':'probe'}]
        if on_progress: on_progress(self.state)
        if sys.argv[2]=='running':
            self.status='running'
            Path(sys.argv[1]+'.ready').touch()
            time.sleep(60)
        else: self.status='waiting'
    def close(self): pass
m=RunManager(db_path=sys.argv[1],session_factory=Session)
r=m.create('probe','crash-key')
while m.get(r['run_id'])['status']=='running': time.sleep(.01)
Path(sys.argv[1]+'.ready').touch()
time.sleep(60)
'''


@pytest.mark.parametrize('status', ['running', 'waiting_user'])
def test_real_process_termination_recovers_unfinished_without_reexecution(tmp_path, status):
    db = tmp_path / 'crash.sqlite3'
    process = subprocess.Popen([sys.executable, '-X', 'utf8', '-c', WORKER, str(db), status],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
    try:
        # 用隔离目录中的就绪标记避免管道readline无界阻塞。
        deadline = monotonic() + 15
        ready = Path(str(db) + '.ready')
        while not ready.exists() and process.poll() is None and monotonic() < deadline:
            sleep(.02)
        assert ready.exists(), '隔离子进程未进入预期任务状态'
    finally:
        process.terminate()
        process.communicate(timeout=10)
    def forbidden(*a, **k):
        pytest.fail('启动恢复不得执行研究图或模型')
    manager = RunManager(db_path=db, session_factory=forbidden)
    try:
        view = manager.list_runs()[0]
        assert view['status'] == 'interrupted' and view['clarification'] is None
        assert view['research_round'] == 2 and view['tool_call_count'] == 3
        assert manager.trace(view['run_id'])['events'][0]['event'] == 'before_crash'
        assert manager.trace(view['run_id'])['events'][-1]['event'] == 'execution_interrupted'
        assert manager.create('probe', 'crash-key')['run_id'] == view['run_id']
    finally:
        manager.close()


def test_api_history_paging_filter_and_restarted_client(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'build_research_graph', fake_graph)
    db = tmp_path / 'api.sqlite3'
    with MemoryClient(create_app(db_path=db)) as connection:
        remote = ResearchClient(client=connection)
        root = remote.create('root', 'root')
        wait(connection, root['run_id'], 'completed')
        child = remote.follow_up(root['run_id'], 'child', 'child')
        wait(connection, child['run_id'], 'completed')
        other = remote.create('other', 'other')
        wait(connection, other['run_id'], 'completed')
    with MemoryClient(create_app(db_path=db)) as connection:
        remote = ResearchClient(client=connection)
        assert remote.health()['storage'] == 'sqlite'
        assert remote.list_runs(limit=1)[0]['run_id'] == other['run_id']
        assert remote.list_runs(limit=1, offset=1)[0]['run_id'] == child['run_id']
        assert remote.list_runs(offset=100) == []
        assert len(remote.list_runs(conversation_id=root['conversation_id'])) == 2
        task = RemoteResearchTask(remote, remote.get(root['run_id']))
        task.refresh()
        assert task.status == 'completed' and task.state['final_report']
        for query in ('limit=0', 'limit=101', 'offset=-1', 'conversation_id=bad'):
            assert connection.get('/runs?' + query).status_code == 422


def test_single_database_owner_and_unsupported_schema_preserve_data(tmp_path):
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=FakeSession)
    try:
        with pytest.raises(RuntimeError, match='其他服务占用'):
            RunManager(db_path=db)
        item = first.create('ok', 'one')
        eventually(first, item['run_id'], 'completed')
    finally:
        first.close()
    with sqlite3.connect(db) as connection:
        connection.execute('PRAGMA user_version=99')
    with pytest.raises(RuntimeError, match='数据库版本'):
        RunManager(db_path=db)
    with sqlite3.connect(db) as connection:
        assert connection.execute('SELECT count(*) FROM runs').fetchone()[0] == 1
        assert connection.execute('PRAGMA user_version').fetchone()[0] == 99


def test_database_insert_failure_blocks_execution_and_hides_raw_exception(tmp_path, monkeypatch, caplog):
    def forbidden(*a, **k):
        pytest.fail('持久保存失败时不得安排模型执行')
    manager = RunManager(db_path=tmp_path / 'archive.sqlite3', session_factory=forbidden)
    def fail(*a, **k):
        raise sqlite3.OperationalError('Bearer private-secret /local/private/path')
    monkeypatch.setattr(manager._store, 'save', fail)
    try:
        with pytest.raises(ServiceError) as error:
            manager.create('ok', 'key')
        assert error.value.code == 'storage_unavailable' and error.value.status_code == 503
        assert not manager.available and not manager._records
        assert 'private-secret' not in caplog.text and 'private/path' not in caplog.text
        assert 'private-secret' not in str(error.value.payload())
    finally:
        manager.close()


def test_actual_sqlite_transaction_failure_preserves_last_durable_snapshot(tmp_path, caplog):
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=FakeSession)
    root = first.create('initial', 'root')
    eventually(first, root['run_id'], 'completed')
    original = first.result(root['run_id'])
    with sqlite3.connect(db) as connection:
        connection.execute("""CREATE TRIGGER reject_update BEFORE UPDATE ON runs
                            BEGIN SELECT RAISE(ABORT, 'private-storage-detail'); END""")
    child = first.create('next', 'next')
    eventually(first, child['run_id'], 'failed')
    assert not first.available and 'private-storage-detail' not in caplog.text
    first.close()
    with sqlite3.connect(db) as connection:
        # UPDATE 被整个事务回滚，不能把尚未提交的状态误称为已持久保存。
        payload = json.loads(connection.execute('SELECT payload FROM runs WHERE run_id=?',
                                               (child['run_id'],)).fetchone()[0])
        assert payload['status'] == 'running' and payload['trace'] == []
        connection.execute('DROP TRIGGER reject_update')
    second = RunManager(db_path=db)
    try:
        assert second.result(root['run_id']) == original
        assert second.get(child['run_id'])['status'] == 'interrupted'
    finally:
        second.close()


def test_streamlit_restarted_history_pagination_and_interruption(tmp_path, monkeypatch):
    from asgi_helpers import require_loopback
    require_loopback()
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=FakeSession)
    for i in range(20):
        view = first.create(f'历史{i}', f'history-{i}')
        eventually(first, view['run_id'], 'completed')
    waiting = first.create('clarify', 'waiting')
    eventually(first, waiting['run_id'], 'waiting_user')
    first.close()
    second = RunManager(db_path=db, session_factory=FakeSession)
    with MemoryClient(create_app(second)) as connection:
        remote = ResearchClient(client=connection)
        monkeypatch.setattr('atmospheric_research_agent.ui.client.ResearchClient', lambda: remote)
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'streamlit_app.py', default_timeout=15).run()
        assert not app.exception
        next(b for b in app.button if b.label.startswith('执行中断 ·')).click().run()
        assert not app.exception
        assert any(m.value == '执行中断' for m in app.metric)
        assert next(b for b in app.button if b.label == '继续研究').disabled
        assert not any(b.label == '提交回答，继续原任务' for b in app.button)
        next(b for b in app.button if b.label == '下一页').click().run()
        assert not app.exception and app.session_state['history_offset'] == 20
        assert any(b.label.endswith('历史0') for b in app.button)
        next(b for b in app.button if b.label.endswith('历史0')).click().run()
        assert not app.exception and any(b.label == '继续研究' for b in app.button)


@pytest.mark.parametrize('question, ending', [('boom', 'failed'), ('clarify', 'abandoned')])
def test_terminal_errors_and_abandonment_remain_terminal_after_restart(tmp_path, question, ending):
    db = tmp_path / 'archive.sqlite3'
    first = RunManager(db_path=db, session_factory=FakeSession)
    root = first.create(question, 'root')
    eventually(first, root['run_id'], 'failed' if ending == 'failed' else 'waiting_user')
    if ending == 'abandoned':
        first.abandon(root['run_id'])
    before = first.get(root['run_id'])
    first.close()
    second = RunManager(db_path=db)
    try:
        assert second.get(root['run_id']) == before
        assert second.get(root['run_id'])['status'] == ending
    finally:
        second.close()
