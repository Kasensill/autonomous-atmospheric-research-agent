"""上传→任务→HITL恢复→报告：真实程序链路，模型替身，不调用外部 API。"""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from asgi_helpers import MemoryClient
from test_service_api import wait
from test_service import FakeSession
from atmospheric_research_agent.agent import nodes, materials
import atmospheric_research_agent.agent.graph as graph_module
from atmospheric_research_agent.service import api
from atmospheric_research_agent.service.manager import RunManager
from atmospheric_research_agent.service.uploads import MaterialUpload, UploadStore, validate_uploads
from atmospheric_research_agent.ui.client import ResearchClient, RemoteResearchTask


BODY = '原创样例：只讨论虚构温度变化，不声称是真实观测。'.encode('utf-8')


@pytest.fixture
def uploaded_service(tmp_path, monkeypatch):
    class Agent:
        def invoke(self, messages):
            if not any('我的补充：' in str(m.content) for m in messages):
                return AIMessage(content='', tool_calls=[{'id': 'ask', 'name': 'ask_user',
                    'args': {'question': '你希望聚焦材料的哪部分？'}}])
            return AIMessage(content='{"decision":"assess","plan_update":{}}')
    class Assessment:
        def invoke(self, messages):
            return nodes.EvidenceAssessmentOutput(sufficient=True, covered_points=['文本'], missing_points=[],
                potential_conflicts=[], next_step='full_report', reason='已知材料足够分析原文。')
    class Writer:
        def invoke(self, messages):
            return AIMessage(content='该材料讨论虚构温度变化，不是真实观测。[E1]')
    class Reviewer:
        def invoke(self, messages):
            return nodes.ReportReviewOutput(verdict='pass', supported_points=['原文'],
                unsupported_or_overstated_claims=[], citation_issues=[], required_changes=[], reason='离线核验。')
    def forbidden(*args, **kwargs):
        pytest.fail('材料任务不应执行研究工具或写长期记忆')
    monkeypatch.setattr(graph_module, 'recall_memory_node', lambda _: {})
    monkeypatch.setattr(nodes, 'create_research_model', Agent)
    monkeypatch.setattr(nodes, 'create_assessment_model', Assessment)
    monkeypatch.setattr(graph_module, 'generate_draft_node', lambda s: nodes.generate_draft_with_model(s, Writer()))
    monkeypatch.setattr(graph_module, 'verify_report_node', lambda s: nodes.verify_report_with_model(s, Reviewer()))
    monkeypatch.setattr(graph_module, 'tools_node', forbidden)
    monkeypatch.setattr(nodes, 'save_project_memory', forbidden)
    calls = []
    original = materials.load_materials
    def counted(paths, *args, **kwargs):
        calls.append(list(paths))
        return original(paths, *args, **kwargs)
    monkeypatch.setattr(materials, 'load_materials', counted)
    store = UploadStore(tmp_path / 'uploads')
    manager = RunManager(upload_store=store)
    with MemoryClient(api.create_app(manager)) as client:
        yield client, store, calls


def post_material(client, *, key='material-one', content=BODY, role='analysis', name='原文.txt'):
    return client.post('/runs/with-materials', data={'question': '解读这份材料', 'material_role': role},
                       files=[('files', (name, content, 'application/octet-stream'))],
                       headers={'Idempotency-Key': key})


def test_upload_resume_reuses_evidence_and_returns_coverage(uploaded_service):
    client, store, calls = uploaded_service
    response = post_material(client)
    assert response.status_code == 202, response.text
    run_id = response.json()['run_id']
    waiting = wait(client, run_id, 'waiting_user')
    assert waiting['material_files'] == ['原文.txt'] and waiting['material_role'] == 'analysis'
    assert waiting['research_round'] == waiting['tool_call_count'] == 0
    assert len(calls) == 1
    saved = list(store.root.rglob('原文.txt'))
    assert len(saved) == 1 and saved[0].read_bytes() == BODY
    assert post_material(client).json()['run_id'] == run_id
    assert len(list(store.root.rglob('原文.txt'))) == 1
    answer = {'question_id': waiting['clarification']['question_id'], 'answer': '只分析原文。'}
    assert client.post(f'/runs/{run_id}/answers', json=answer).status_code == 202
    wait(client, run_id, 'completed')
    result = client.get(f'/runs/{run_id}/result').json()
    state = result['state']
    assert result['delivery_type'] == 'full_report'
    assert len(calls) == 1  # resume 不重复加载或解析
    assert len(state['evidence']) == 1 and state['evidence'][0]['material_id'] == sha256(BODY).hexdigest()[:8]
    assert state['materials'][0]['sent_characters'] == len(BODY.decode())
    assert '材料覆盖：' in state['final_report'] and '用户提供：原文.txt' in state['final_report']
    assert any(e['event'] == 'memory_skipped' and '含用户材料' in e['detail']
               for e in client.get(f'/runs/{run_id}/trace').json()['events'])
    assert str(store.root) not in json.dumps(result, ensure_ascii=False)
    assert not any('material_paths' in item for item in (waiting, result, state))
    assert client.post(f'/runs/{run_id}/answers', json=answer).status_code == 409


def test_upload_idempotency_includes_content_role_name_and_json_route(uploaded_service):
    client, store, _ = uploaded_service
    first = post_material(client).json()
    wait(client, first['run_id'], 'waiting_user')
    for options in ({'content': b'changed'}, {'role': 'clue'}, {'name': 'another.txt'}):
        response = post_material(client, **options)
        assert response.status_code == 409 and response.json()['code'] == 'idempotency_conflict'
    response = client.post('/runs', json={'question': '解读这份材料'}, headers={'Idempotency-Key': 'material-one'})
    assert response.status_code == 409
    assert len(list(store.root.rglob('*.txt'))) == 1
    assert client.post(f"/runs/{first['run_id']}/abandon").status_code == 200


@pytest.mark.parametrize('name', ['../escape.txt', r'..\escape.txt', 'C:escape.txt', 'CON.txt', 'no.docx'])
def test_invalid_upload_name_or_type_has_no_storage_side_effect(uploaded_service, name):
    client, store, calls = uploaded_service
    response = post_material(client, name=name)
    assert response.status_code == 422 and response.json()['code'] == 'invalid_material'
    assert client.get('/runs').json() == [] and calls == []
    assert not store.root.exists()


def test_upload_missing_empty_excess_files_and_bad_role(uploaded_service):
    client, store, _ = uploaded_service
    assert post_material(client, content=b'').status_code == 422
    assert post_material(client, role='invalid').status_code == 422
    response = client.post('/runs/with-materials', data={'question': '研究'}, headers={'Idempotency-Key': 'missing'})
    assert response.status_code == 422
    from atmospheric_research_agent.agent.settings import MATERIAL_MAX_FILES
    response = client.post('/runs/with-materials', data={'question': '研究'},
        files=[('files', (f'{i}.txt', b'x')) for i in range(MATERIAL_MAX_FILES + 1)],
        headers={'Idempotency-Key': 'many'})
    assert response.status_code == 422 and not store.root.exists()


def test_upload_size_limits_with_and_without_length(uploaded_service, monkeypatch):
    client, store, _ = uploaded_service
    monkeypatch.setattr(api, 'MATERIAL_UPLOAD_MAX_BYTES', 8)
    assert post_material(client, content=b'x' * 9).status_code == 413
    monkeypatch.setattr(api, 'MATERIAL_UPLOAD_MAX_REQUEST_BYTES', 64)
    response = post_material(client)
    assert response.status_code == 413 and response.json()['code'] == 'material_too_large'
    response = client.post('/runs/with-materials', content=iter([b'x' * 65]), headers={
        'Content-Type': 'multipart/form-data; boundary=test', 'Idempotency-Key': 'stream'})
    assert response.status_code == 413, response.text
    assert not store.root.exists()


def test_unreadable_material_is_visible_in_trace(uploaded_service):
    client, _, _ = uploaded_service
    response = post_material(client, content=b'\xff\xfe\xff')
    run_id = response.json()['run_id']
    wait(client, run_id, 'waiting_user')
    events = client.get(f'/runs/{run_id}/trace').json()['events']
    assert any(e['event'] == 'material_skipped' and 'UTF-8' in e['detail'] for e in events)
    assert client.post(f'/runs/{run_id}/abandon').status_code == 200


def test_duplicate_names_are_saved_separately_without_overwrite(uploaded_service):
    client, store, _ = uploaded_service
    response = client.post('/runs/with-materials', data={'question': '解读材料', 'material_role': 'analysis'},
        files=[('files', ('相同.txt', b'one')), ('files', ('相同.txt', b'two'))],
        headers={'Idempotency-Key': 'duplicates'})
    assert response.status_code == 202
    run_id = response.json()['run_id']
    wait(client, run_id, 'waiting_user')
    assert sorted(p.read_bytes() for p in store.root.rglob('相同.txt')) == [b'one', b'two']


def test_concurrent_material_creation_saves_only_once(tmp_path):
    created = []
    class Session(FakeSession):
        def __init__(self, *args, material_paths, material_role, **kwargs):
            created.append((material_paths, material_role))
            super().__init__(*args, **kwargs)
    store = UploadStore(tmp_path / 'uploads')
    manager = RunManager(session_factory=Session, upload_store=store)
    try:
        def create(_):
            return manager.create('clarify', 'same', materials=[MaterialUpload('one.txt', BODY)], material_role='analysis')
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(create, range(2)))
        assert results[0]['run_id'] == results[1]['run_id']
        from test_service import eventually
        eventually(manager, results[0]['run_id'], 'waiting_user')
        assert len(created) == 1 and len(list(store.root.rglob('*.txt'))) == 1
    finally:
        manager.close()


def test_client_upload_and_task_restore_keep_material_context(uploaded_service):
    client, _, calls = uploaded_service
    remote = ResearchClient(client=client)
    view = remote.create('解读这份材料', 'client-material', materials=[('原文.txt', BODY)], material_role='analysis')
    wait(client, view['run_id'], 'waiting_user')
    task = RemoteResearchTask(remote, remote.list_runs()[0])
    assert task.state['material_files'] == ['原文.txt']
    task.answer('分析原文。')
    wait(client, view['run_id'], 'completed')
    task.refresh()
    assert task.state['material_role'] == 'analysis' and task.state['materials']
    assert task.state['material_files'] == ['原文.txt'] and len(calls) == 1


@pytest.mark.parametrize('name', ['NUL.txt', 'COM1.md', 'bad\n.txt', 'bad\x00.txt', 'x' * 130 + '.txt'])
def test_filename_rules_reject_unsafe_local_names(name):
    with pytest.raises(ValueError):
        validate_uploads([MaterialUpload(name, b'content')], 'analysis')


def test_standard_asgi_upload_handles_spooled_file_and_resume(uploaded_service):
    from asgi_helpers import require_loopback
    require_loopback()
    from fastapi.testclient import TestClient
    _, store, _ = uploaded_service
    manager = RunManager(upload_store=store)
    with TestClient(api.create_app(manager), base_url='http://localhost') as client:
        # 超过 multipart 内存暂存范围，覆盖真实异步文件读取路径。
        response = post_material(client, key='spooled', content=b'a' * (2 * 1024 * 1024))
        assert response.status_code == 202, response.text
        run_id = response.json()['run_id']
        waiting = wait(client, run_id, 'waiting_user')
        assert client.post(f'/runs/{run_id}/answers', json={
            'question_id': waiting['clarification']['question_id'], 'answer': '分析已读原文。'}).status_code == 202
        wait(client, run_id, 'completed')
        state = client.get(f'/runs/{run_id}/result').json()['state']
        assert state['materials'][0]['parsed_characters'] == 2 * 1024 * 1024
        from atmospheric_research_agent.agent.settings import MATERIAL_MAX_CHARS_TOTAL
        assert state['materials'][0]['sent_characters'] == MATERIAL_MAX_CHARS_TOTAL


def test_streamlit_material_selection_resume_coverage_and_reset(uploaded_service, monkeypatch):
    from asgi_helpers import require_loopback
    require_loopback()
    import streamlit
    from types import SimpleNamespace
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    client, _, calls = uploaded_service
    remote = ResearchClient(client=client)
    monkeypatch.setattr('atmospheric_research_agent.ui.client.ResearchClient', lambda: remote)
    # AppTest 不提供文件选择器驱动；仅替换选择器返回值，其余表单、HTTP和任务全走真实实现。
    file = SimpleNamespace(name='原文.txt', size=len(BODY), getvalue=lambda: BODY)
    monkeypatch.setattr(streamlit, 'file_uploader', lambda *a, **kw:
                        [file] if kw['key'] == 'material_uploads_0' else [])
    path = Path(__file__).resolve().parents[1] / 'streamlit_app.py'
    app = AppTest.from_file(path, default_timeout=10).run()
    assert not app.exception
    app.text_area(key='question_draft').input('解读这份材料')
    app.selectbox(key='material_role_choice').select('analysis')
    next(b for b in app.button if b.label == '开始研究').click().run()
    assert not app.exception
    run_id = remote.list_runs()[0]['run_id']
    wait(client, run_id, 'waiting_user')
    app.run()
    assert any('追问后继续使用同一批材料' in c.value for c in app.caption)
    key = next(t.key for t in app.text_area if t.label == '你的补充')
    app.text_area(key=key).input('只分析材料原文。')
    next(b for b in app.button if b.label == '提交回答，继续原任务').click().run()
    wait(client, run_id, 'completed')
    app.run()
    assert not app.exception and len(calls) == 1
    assert any(t.label == '材料与覆盖' for t in app.tabs)
    assert any('本轮送入模型' in c.value and '0 页' not in c.value and '[]' not in c.value for c in app.caption)
    assert any(e.label == '查看精确范围 · 原文.txt' for e in app.expander)
    assert any(s.value == '本次研究记录' for s in app.subheader)
    next(b for b in app.button if b.label == '＋ 新建研究').click().run()
    assert not app.exception
    assert app.text_area(key='question_draft').value == ''
    assert app.selectbox(key='material_role_choice').value == 'auto'
    assert app.session_state['material_upload_generation'] == 1


@pytest.mark.parametrize('name', ['文档.md', 'DOCUMENT.TXT'])
def test_markdown_and_uppercase_text_uploads(uploaded_service, name):
    client, _, _ = uploaded_service
    response = post_material(client, name=name)
    assert response.status_code == 202
    waiting = wait(client, response.json()['run_id'], 'waiting_user')
    assert waiting['material_files'] == [name]
    events = client.get(f"/runs/{waiting['run_id']}/trace").json()['events']
    assert any(e['event'] == 'material_loaded' for e in events)


@pytest.mark.parametrize('text_layer', [True, False])
def test_pdf_upload_reuses_real_parser_and_shows_scan_failure(uploaded_service, text_layer):
    pypdf = pytest.importorskip('pypdf')
    from io import BytesIO
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    if text_layer:
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
            NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
        stream = DecodedStreamObject()
        stream.set_data(b'BT /F1 12 Tf 10 100 Td (Original offline material, not real observations.) Tj ET')
        page[NameObject('/Contents')] = stream
    buffer = BytesIO()
    writer.write(buffer)
    client, store, calls = uploaded_service
    response = post_material(client, name='原稿.pdf', content=buffer.getvalue())
    assert response.status_code == 202
    run_id = response.json()['run_id']
    wait(client, run_id, 'waiting_user')
    events = client.get(f'/runs/{run_id}/trace').json()['events']
    if text_layer:
        assert any(e['event'] == 'material_loaded' for e in events)
    else:
        assert any(e['event'] == 'material_skipped' and 'OCR' in e['detail'] for e in events)
    assert len(calls) == 1 and len(list(store.root.rglob('*.pdf'))) == 1


def test_failed_save_does_not_create_run_or_expose_path(uploaded_service, monkeypatch):
    client, _, _ = uploaded_service
    def fail(*args, **kwargs):
        raise OSError(r'C:\private\must-not-be-shown')
    monkeypatch.setattr(UploadStore, 'save', fail)
    response = post_material(client)
    assert response.status_code == 500 and response.json()['code'] == 'material_save_failed'
    assert response.json()['error_id'] and 'private' not in response.text
    assert client.get('/runs').json() == []


def test_upload_rejects_client_server_path_or_extra_fields(uploaded_service):
    client, store, _ = uploaded_service
    response = client.post('/runs/with-materials', data={
        'question': '研究', 'material_paths': '../server.txt'}, files=[('files', ('ok.txt', BODY))],
        headers={'Idempotency-Key': 'extra-field'})
    assert response.status_code == 422 and response.json()['code'] == 'invalid_request'
    assert not store.root.exists()
