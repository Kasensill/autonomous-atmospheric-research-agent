import httpx
import pytest
from pathlib import Path

from test_service_api import client, wait
from atmospheric_research_agent.ui.client import ApiError, RemoteResearchTask, ResearchClient


def test_http_client_and_remote_task_restore(client):
    remote=ResearchClient(client=client)
    view=remote.create('clarify','key')
    wait(client,view['run_id'],'waiting_user')
    task=RemoteResearchTask(remote,remote.list_runs()[0]);task.refresh()
    assert task.status=='waiting' and task.question=='请提供地点。'
    task.answer('济宁')
    wait(client,view['run_id'],'completed');task.refresh()
    assert task.status=='completed' and '济宁' in task.state['final_report']
    assert task.ending=='完整报告' and 'sequence' in task.export_json('trace')


def test_client_does_not_retry_or_leak_connection_errors():
    calls=[]
    def fail(request):
        calls.append(request)
        raise httpx.ConnectError('Bearer secret',request=request)
    transport=httpx.MockTransport(fail)
    with httpx.Client(transport=transport,base_url='http://localhost') as connection:
        client=ResearchClient(client=connection)
        with pytest.raises(ApiError) as error:
            client.create('question','stable-key')
        assert len(calls)==1 and 'secret' not in str(error.value)
    with pytest.raises(ApiError):
        ResearchClient('http://external.example.com')


def test_streamlit_uses_api_preserves_question_and_restores_task(client,monkeypatch):
    from asgi_helpers import require_loopback
    require_loopback()
    AppTest=pytest.importorskip('streamlit.testing.v1').AppTest
    remote=ResearchClient(client=client)
    monkeypatch.setattr('atmospheric_research_agent.ui.client.ResearchClient',lambda:remote)
    app_path=Path(__file__).resolve().parents[1]/'streamlit_app.py'
    app=AppTest.from_file(app_path,default_timeout=10).run()
    assert not app.exception
    app.text_area(key='question_draft').input('clarify')
    next(b for b in app.button if b.label=='开始研究').click().run()
    run_id=remote.list_runs()[0]['run_id']
    wait(client,run_id,'waiting_user');app.run()
    assert not app.exception
    assert any(t.value=='clarify' and t.disabled for t in app.text_area)
    key=next(t.key for t in app.text_area if t.label=='你的补充')
    app.text_area(key=key).input('济宁')
    next(b for b in app.button if b.label=='提交回答，继续原任务').click().run()
    wait(client,run_id,'completed');app.run()
    assert not app.exception
    assert any('本次研究记录' in s.value for s in app.subheader)
    assert any('济宁' in str(m.value) for m in app.markdown)
    restored=AppTest.from_file(app_path,default_timeout=10).run()
    next(b for b in restored.button if b.label.startswith('completed ·')).click().run()
    assert not restored.exception
    assert any(t.value=='clarify' and t.disabled for t in restored.text_area)
