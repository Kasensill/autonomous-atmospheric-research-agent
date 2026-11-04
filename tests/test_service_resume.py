"""真实进程退出后的 SQLite 检查点恢复。"""
import subprocess
import sys

import pytest

from asgi_helpers import MemoryClient
from test_service_api import fake_graph, wait
from atmospheric_research_agent.service import engine
from atmospheric_research_agent.service.api import create_app
from atmospheric_research_agent.service.manager import RunManager

CHILD = """
import os, sys, time
from langgraph.graph import END, START, StateGraph
from atmospheric_research_agent.agent.state import ResearchState
from atmospheric_research_agent.agent.nodes import request_clarification_node
from atmospheric_research_agent.service import engine
from atmospheric_research_agent.service.manager import RunManager
def build(*, checkpointer):
    graph = StateGraph(ResearchState)
    graph.add_node('request_clarification', request_clarification_node)
    def finish(state):
        os._exit(23)
    def prepare(state):
        return {'clarification_question':'请提供地点。','research_round':1,'tool_call_count':2,
                'messages':[], 'trace':[{'node':'prepare','event':'prepared','detail':'offline'}]}
    graph.add_node('prepare', prepare)
    graph.add_node('finish', finish)
    graph.add_edge(START, 'prepare')
    graph.add_conditional_edges('prepare', lambda s: 'request_clarification' if s['question']=='clarify' else 'finish')
    graph.add_edge('request_clarification', 'finish')
    graph.add_edge('finish', END)
    return graph.compile(checkpointer=checkpointer)
engine.build_research_graph = build
manager = RunManager(db_path=sys.argv[1])
view = manager.create(sys.argv[2], 'probe')
while manager.get(view['run_id'])['status'] == 'running':
    time.sleep(.02)
os._exit(23 if manager.get(view['run_id'])['status'] == 'waiting_user' else 24)
"""


@pytest.mark.parametrize('question', ['ordinary', 'clarify'])
def test_restart_resume_preserves_checkpoint_and_blocks_duplicate(tmp_path, monkeypatch, question):
    path = tmp_path / 'runs.sqlite3'
    process = subprocess.run([sys.executable, '-X', 'utf8', '-c', CHILD, str(path), question],
                             timeout=25, capture_output=True, text=True, encoding='utf8')
    assert process.returncode == 23, process.stderr
    monkeypatch.setattr(engine, 'build_research_graph', fake_graph)
    with MemoryClient(create_app(db_path=path)) as client:
        before = client.get('/runs').json()[0]
        run_id = before['run_id']
        assert before['status'] == 'interrupted' and before['can_resume']
        endpoint = f'/runs/{run_id}/resume'
        assert client.post(endpoint).status_code == 202
        assert client.post(endpoint).status_code == 409
        if question == 'clarify':
            waiting = wait(client, run_id, 'waiting_user')
            assert waiting['clarification']['question'] == '请提供地点。'
            answer = {'question_id':waiting['clarification']['question_id'], 'answer':'北京'}
            assert client.post(f'/runs/{run_id}/answers', json=answer).status_code == 202
        done = wait(client, run_id, 'completed')
        assert (done['research_round'], done['tool_call_count'], done['resume_count']) == (1, 2, 1)
        result = client.get(f'/runs/{run_id}/result').json()['state']
        if question == 'clarify':
            assert result['clarification_history'][0]['answer'] == '北京'
        trace = client.get(f'/runs/{run_id}/trace').json()['events']
        assert sum(t['event'] == 'prepared' for t in trace) == 1
        assert sum(t['event'] == 'execution_interrupted' for t in trace) == 1


def test_legacy_interrupted_run_has_no_resume_checkpoint():
    manager = RunManager()
    with MemoryClient(create_app(manager)) as client:
        from atmospheric_research_agent.service.manager import RunRecord
        from uuid import uuid4
        record = RunRecord(str(uuid4()), '历史任务', 'legacy')
        manager._interrupt(record)
        assert client.get(f'/runs/{record.run_id}').json()['can_resume'] is False
        response = client.post(f'/runs/{record.run_id}/resume')
        assert response.status_code == 409
        assert response.json()['code'] == 'checkpoint_missing'


def test_streamlit_continue_button_reopens_checkpoint_question(tmp_path, monkeypatch):
    from pathlib import Path
    from asgi_helpers import require_loopback
    from atmospheric_research_agent.ui.client import ResearchClient
    require_loopback()
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    monkeypatch.setattr(engine, 'build_research_graph', fake_graph)
    path = tmp_path / 'runs.sqlite3'
    with MemoryClient(create_app(db_path=path)) as client:
        run_id = client.post('/runs', json={'question': 'clarify'},
                             headers={'Idempotency-Key': 'ui-probe'}).json()['run_id']
        wait(client, run_id, 'waiting_user')
    with MemoryClient(create_app(db_path=path)) as client:
        remote = ResearchClient(client=client)
        monkeypatch.setattr('atmospheric_research_agent.ui.client.ResearchClient', lambda: remote)
        app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'streamlit_app.py').run()
        app.button(key=f"open_{run_id}").click().run()
        button = next(b for b in app.button if b.label == '继续研究')
        assert not button.disabled
        button.click().run()
        wait(client, run_id, 'waiting_user')
        app.run()
        assert not app.exception
        assert any(b.label == '提交回答，继续原任务' for b in app.button)
        assert not any(b.label == '继续研究' for b in app.button)
