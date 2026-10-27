"""Streamlit 的本地 HTTP 客户端；不持有图、检查点或模型凭据。"""
import json
import os
from urllib.parse import urlparse

import httpx


class ApiError(RuntimeError):
    def __init__(self, message: str, *, code: str='connection_failed', run_id: str | None=None):
        super().__init__(message)
        self.code, self.run_id = code, run_id


class ResearchClient:
    def __init__(self, base_url: str | None=None, *, client=None):
        base_url=base_url or os.getenv('RESEARCH_API_URL','http://127.0.0.1:8000')
        parsed=urlparse(base_url)
        if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost'} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ApiError('首版只支持无凭据的本地HTTP服务地址。',code='invalid_api_url')
        self.base_url=base_url.rstrip('/')
        self._client=client or httpx.Client(base_url=self.base_url,timeout=10,trust_env=False)

    def request(self, method: str, path: str, **kwargs):
        try:
            response=self._client.request(method,path,**kwargs)
        except httpx.HTTPError:
            raise ApiError('无法连接研究服务，请确认服务已启动。未自动重试；创建请求再次提交时沿用原请求标识。') from None
        try:
            data=response.json()
        except ValueError:
            raise ApiError('研究服务返回了无法识别的响应。',code='invalid_response') from None
        if response.is_error:
            raise ApiError(data.get('message','研究服务拒绝了请求。'),
                           code=data.get('code','http_error'),run_id=data.get('run_id'))
        return data

    def health(self):
        return self.request('GET','/health')

    def list_runs(self):
        return self.request('GET','/runs')

    def create(self, question: str, key: str):
        return self.request('POST','/runs',json={'question':question},headers={'Idempotency-Key':key})

    def get(self, run_id: str):
        return self.request('GET',f'/runs/{run_id}')

    def answer(self, run_id: str, question_id: str, answer: str):
        return self.request('POST',f'/runs/{run_id}/answers',json={'question_id':question_id,'answer':answer})

    def abandon(self, run_id: str):
        return self.request('POST',f'/runs/{run_id}/abandon')


class RemoteResearchTask:
    def __init__(self, client: ResearchClient, view: dict):
        self.client=client
        self.state={'run_id':view['run_id'],'question':view['question']}
        self.view=view
        self._trace_after=0
        self._apply(view)

    def _apply(self, view):
        self.view=view
        self.status={'waiting_user':'waiting'}.get(view['status'],view['status'])
        self.question=(view.get('clarification') or {}).get('question','')
        error=view.get('error') or {}
        self.error=error.get('message','')
        if error.get('error_id'):
            self.error+=' 错误编号：'+error['error_id']
        self.elapsed_ms=view['elapsed_ms']
        self.state.update({k:view[k] for k in ['research_round','tool_call_count']})

    def refresh(self):
        view=self.client.get(self.state['run_id'])
        changed=view['version']!=self.view['version']
        self._apply(view)
        if self.status=='completed' and not self.state.get('final_report'):
            self.state.update(self.client.request('GET',f"/runs/{view['run_id']}/result")['state'])
        if changed or 'trace' not in self.state:
            while True:
                page=self.client.request('GET',f"/runs/{view['run_id']}/trace",
                                         params={'after':self._trace_after,'limit':500})
                self.state.setdefault('trace',[]).extend(page['events'])
                self._trace_after=page['next_after']
                if not page['has_more']:
                    break
        return changed

    def answer(self, text: str):
        view=self.client.answer(self.state['run_id'],self.view['clarification']['question_id'],text)
        self._apply(view)

    @property
    def ending(self):
        if self.status=='failed': return '执行中止'
        if self.status=='abandoned': return '已放弃'
        if self.status=='waiting': return '等待补充信息'
        if self.status!='completed': return '研究进行中'
        return {'full_report':'完整报告','limited_report':'受限报告','failure_notice':'失败说明'}[self.view['delivery_type']]

    def export_json(self, field: str):
        if field not in {'evidence','trace'}:
            raise ValueError('仅允许导出证据与Trace。')
        return json.dumps({'run_id':self.state['run_id'],field:self.state.get(field,[])},ensure_ascii=False,indent=2)
