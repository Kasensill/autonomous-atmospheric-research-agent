from time import monotonic, sleep
from uuid import uuid4

import pytest
from asgi_helpers import MemoryClient
from langgraph.graph import END, START, StateGraph

from atmospheric_research_agent.agent.nodes import request_clarification_node
from atmospheric_research_agent.agent.state import ResearchState
from atmospheric_research_agent.agent.resume import ResumeValidatedGraph
from atmospheric_research_agent.service import engine
from atmospheric_research_agent.service.api import create_app
from atmospheric_research_agent.service.manager import RunManager


def fake_graph(*,checkpointer):
    graph=StateGraph(ResearchState)
    def prepare(state):
        return {'clarification_question':'请提供地点。','research_round':1,'tool_call_count':2,
                'messages':[], 'trace':[{'node':'prepare','event':'prepared','detail':'offline'}]}
    def finish(state):
        if state['question']=='throw':
            raise RuntimeError('secret must not reach HTTP or logs')
        return {'final_report':'已恢复：'+str(state.get('clarification_history',[])),
                'report_mode':'full','report_review':{'verdict':'pass'},
                'trace':[{'node':'finish','event':'finished','detail':'offline'}]}
    graph.add_node('prepare',prepare)
    graph.add_node('request_clarification',request_clarification_node)
    graph.add_node('finish',finish)
    graph.add_edge(START,'prepare')
    graph.add_conditional_edges('prepare',lambda s:'request_clarification' if s['question']=='clarify' else 'finish')
    graph.add_edge('request_clarification','finish')
    graph.add_edge('finish',END)
    return ResumeValidatedGraph(graph.compile(checkpointer=checkpointer))


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(engine,'build_research_graph',fake_graph)
    manager=RunManager()
    with MemoryClient(create_app(manager)) as client:
        yield client


def wait(client,run_id,status):
    deadline=monotonic()+5
    while monotonic()<deadline:
        result=client.get(f'/runs/{run_id}').json()
        if result['status']==status:
            return result
        sleep(.005)
    pytest.fail(str(result))


def test_http_real_interrupt_resume_result_and_private_fields(client):
    assert client.get('/health').json()['status']=='ok'
    response=client.post('/runs',json={'question':'clarify'},headers={'Idempotency-Key':'one'})
    assert response.status_code==202
    run_id=response.json()['run_id']
    waiting=wait(client,run_id,'waiting_user')
    assert client.post('/runs',json={'question':'clarify'},headers={'Idempotency-Key':'one'}).json()['run_id']==run_id
    assert client.post('/runs',json={'question':'another'},headers={'Idempotency-Key':'two'}).status_code==409
    assert client.get(f'/runs/{run_id}/result').status_code==409
    assert client.post(f'/runs/{run_id}/answers',json={'answer':'北京','question_id':str(uuid4())}).status_code==409
    answer={'answer':'北京','question_id':waiting['clarification']['question_id']}
    assert client.post(f'/runs/{run_id}/answers',json=answer).status_code==202
    completed=wait(client,run_id,'completed')
    assert client.post(f'/runs/{run_id}/answers',json=answer).status_code==409
    assert completed['research_round']==1 and completed['tool_call_count']==2
    result=client.get(f'/runs/{run_id}/result').json()
    assert result['delivery_type']=='full_report'
    assert '北京' in result['state']['final_report']
    assert result['state']['clarification_history'][0]['answer']=='北京'
    assert 'messages' not in result['state'] and 'recalled_memories' not in result['state']
    assert len(client.get('/runs').json())==1
    trace=client.get(f'/runs/{run_id}/trace',params={'limit':1}).json()
    assert trace['next_after']==1 and trace['has_more']
    assert client.get(f'/runs/{run_id}/trace',params={'after':1}).json()['events'][0]['sequence']==2
    assert client.get(f'/runs/{run_id}/trace',params={'limit':501}).status_code==422


def test_abandon_and_unified_validation(client):
    for body in [{'question':' '},{'question':5},{'question':'ok','unexpected':'not allowed'}]:
        r=client.post('/runs',json=body,headers={'Idempotency-Key':'one'})
        assert r.status_code==422 and set(r.json())=={'code','message','run_id','error_id'}
    assert client.post('/runs',json={'question':'ok'}).status_code==422
    run_id=client.post('/runs',json={'question':'clarify'},headers={'Idempotency-Key':'one'}).json()['run_id']
    wait(client,run_id,'waiting_user')
    assert client.post(f'/runs/{run_id}/abandon').json()['status']=='abandoned'
    assert client.get(f'/runs/{run_id}/result').json()['code']=='task_abandoned'
    assert client.get('/runs/'+str(uuid4())).status_code==404
    assert client.get('/missing').json()['code']=='http_error'
    assert client.get('/health',headers={'Host':'external.example.com'}).json()['code']=='invalid_host'


def test_engine_failure_does_not_poison_service(client,caplog):
    run_id=client.post('/runs',json={'question':'throw'},headers={'Idempotency-Key':'one'}).json()['run_id']
    failed=wait(client,run_id,'failed')
    assert failed['delivery_type'] is None and failed['error']['error_id']
    assert 'secret must' not in caplog.text
    response=client.get(f'/runs/{run_id}/result')
    assert response.status_code==409 and response.json()['error_id']==failed['error']['error_id']
    new=client.post('/runs',json={'question':'ok'},headers={'Idempotency-Key':'two'}).json()['run_id']
    wait(client,new,'completed')


def test_standard_asgi_client_lifespan(monkeypatch):
    from asgi_helpers import require_loopback
    require_loopback()
    from fastapi.testclient import TestClient
    monkeypatch.setattr(engine,'build_research_graph',fake_graph)
    with TestClient(create_app(),base_url='http://localhost') as client:
        assert client.get('/health').status_code==200
        run_id=client.post('/runs',json={'question':'clarify'},headers={'Idempotency-Key':'standard'}).json()['run_id']
        waiting=wait(client,run_id,'waiting_user')
        assert client.post(f'/runs/{run_id}/answers',json={'answer':'北京','question_id':waiting['clarification']['question_id']}).status_code==202
        wait(client,run_id,'completed')


def test_engine_close_releases_real_checkpoint_storage(monkeypatch):
    monkeypatch.setattr(engine,'build_research_graph',fake_graph)
    session=engine.ResearchSession('clarify')
    session.advance()
    assert session.status=='waiting' and session.checkpointer.storage
    session.advance('济宁')
    assert session.status=='completed' and session.state['research_round']==1
    session.close()
    assert session.graph is None
    assert not session.checkpointer.storage and not session.checkpointer.writes and not session.checkpointer.blobs
