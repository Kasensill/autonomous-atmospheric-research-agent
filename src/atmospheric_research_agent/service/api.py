"""FastAPI 工厂；创建应用时不启动研究或访问外部 API。"""
from contextlib import asynccontextmanager
import logging
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.datastructures import Headers

from .manager import RunManager, ServiceError

Text = Annotated[str,StringConstraints(strict=True,strip_whitespace=True,min_length=1,max_length=10000)]
Key = Annotated[str,StringConstraints(pattern=r'^[A-Za-z0-9._:-]{1,128}$')]


class CreateRun(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question: Text


class Answer(BaseModel):
    model_config=ConfigDict(extra='forbid')
    question_id: UUID
    answer: Text


class RunView(BaseModel):
    run_id: str
    question: str
    status: Literal['running','waiting_user','completed','failed','abandoned']
    delivery_type: Literal['full_report','limited_report','failure_notice'] | None
    version: int
    updated_at: str
    clarification: dict | None
    error: dict | None
    elapsed_ms: float
    research_round: int
    tool_call_count: int
    clarification_count: int
    limits: dict[str,int]
    latest_event: dict | None


class LocalHostGuard:
    """本地Host过滤也使用统一JSON错误；不替代认证。"""
    def __init__(self, app):
        self.app=app

    async def __call__(self, scope, receive, send):
        if scope['type']=='http' and Headers(scope=scope).get('host','').split(':')[0] not in {'127.0.0.1','localhost'}:
            response=JSONResponse(status_code=400,content=ServiceError(400,'invalid_host','仅接受本地服务请求。').payload())
            await response(scope,receive,send)
            return
        await self.app(scope,receive,send)


def create_app(manager: RunManager | None=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.manager=manager or RunManager()
        try:
            yield
        finally:
            app.state.manager.close()

    app=FastAPI(title='大气研究 Agent 本地服务',version='0.1.0',lifespan=lifespan)
    app.add_middleware(LocalHostGuard)

    @app.exception_handler(ServiceError)
    async def service_error(request, error):
        return JSONResponse(status_code=error.status_code,content=error.payload())

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, error):
        return JSONResponse(status_code=422,content=ServiceError(422,'invalid_request',
            '请求参数无效，请检查问题、回答、任务编号或游标。').payload())

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request, error):
        return JSONResponse(status_code=error.status_code,content=ServiceError(error.status_code,
            'http_error','请求的地址或方法不受支持。').payload())

    @app.exception_handler(Exception)
    async def unexpected(request, error):
        error_id=str(uuid4())
        logging.getLogger(__name__).error('api_error error_id=%s type=%s',error_id,type(error).__name__)
        return JSONResponse(status_code=500,content=ServiceError(500,'internal_error',
            '服务处理失败，请按错误编号检查日志。',error_id=error_id).payload())

    @app.get('/health')
    async def health(request: Request):
        if not request.app.state.manager.available:
            raise ServiceError(503,'service_unavailable','服务正在关闭或执行设施不可用。')
        return {'status':'ok','storage':'in_memory','max_active_runs':1,'hard_cancel':False}

    @app.post('/runs',status_code=202,response_model=RunView)
    async def create(body: CreateRun, request: Request, idempotency_key: Annotated[Key,Header()]):
        return request.app.state.manager.create(body.question,idempotency_key)

    @app.get('/runs',response_model=list[RunView])
    async def list_runs(request: Request):
        return request.app.state.manager.list_runs()

    @app.get('/runs/{run_id}',response_model=RunView)
    async def get(run_id: UUID, request: Request):
        return request.app.state.manager.get(str(run_id))

    @app.post('/runs/{run_id}/answers',status_code=202,response_model=RunView)
    async def answer(run_id: UUID, body: Answer, request: Request):
        return request.app.state.manager.answer(str(run_id),str(body.question_id),body.answer)

    @app.post('/runs/{run_id}/abandon',response_model=RunView)
    async def abandon(run_id: UUID, request: Request):
        return request.app.state.manager.abandon(str(run_id))

    @app.get('/runs/{run_id}/result')
    async def result(run_id: UUID, request: Request):
        return request.app.state.manager.result(str(run_id))

    @app.get('/runs/{run_id}/trace')
    async def trace(run_id: UUID, request: Request,
              after: Annotated[int,Query(ge=0)]=0, limit: Annotated[int,Query(ge=1,le=500)]=100):
        return request.app.state.manager.trace(str(run_id),after,limit)

    return app
