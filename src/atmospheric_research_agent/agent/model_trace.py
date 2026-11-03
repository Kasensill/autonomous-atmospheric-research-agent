"""运行期间发布模型请求诊断，不采集请求或响应正文。"""
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
import logging
import re
import socket
import ssl
from time import perf_counter, sleep
from uuid import uuid4

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from openai import APIConnectionError, APIStatusError, APITimeoutError

_sink: ContextVar[Callable[[dict], None] | None] = ContextVar("model_trace_sink", default=None)
_clients: ContextVar[list[httpx.Client] | None] = ContextVar("model_trace_clients", default=None)
_node: ContextVar[str | None] = ContextVar("model_trace_node", default=None)
LOGGER = logging.getLogger(__name__)


class InvalidModelResponseError(ValueError):
    """The model endpoint returned a non-API response."""


class ModelHTTPClient(httpx.Client):
    def send(self, request: httpx.Request, *, stream=False, auth=httpx.USE_CLIENT_DEFAULT,
             follow_redirects=httpx.USE_CLIENT_DEFAULT) -> httpx.Response:
        started = perf_counter()
        # Inspect headers before reading a possible gateway error page.
        response = super().send(request, stream=True, auth=auth, follow_redirects=follow_redirects)
        if self._non_api_response(response) or response.status_code in (502, 503, 504):
            if (request.method == "POST" and isinstance(request.stream, httpx.ByteStream)
                    and perf_counter() - started < 5):
                backoff_seconds = 2 if response.status_code == 503 else 0
                response.close()
                if backoff_seconds:
                    sleep(backoff_seconds)
                request.extensions["model_retry_backoff_seconds"] = backoff_seconds
                response = super().send(request, stream=True, auth=auth, follow_redirects=follow_redirects)
            if self._non_api_response(response):
                response.close()
                raise InvalidModelResponseError("模型服务返回了非 JSON 响应。")
        if not stream:
            try:
                response.read()
            finally:
                response.close()
        return response

    @staticmethod
    def _non_api_response(response: httpx.Response) -> bool:
        media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
        return response.is_success and media_type == "text/html"


@contextmanager
def model_trace_scope(sink):
    clients: list[httpx.Client] = []
    token = _sink.set(sink)
    client_token = _clients.set(clients)
    try:
        yield
    finally:
        _clients.reset(client_token)
        _sink.reset(token)
        for client in clients:
            client.close()


@contextmanager
def model_trace_node(node: str):
    token = _node.set(node)
    try:
        yield
    finally:
        _node.reset(token)


def _request_id(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value) else None


def failure_metadata(error: BaseException) -> dict:
    chain = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen and len(chain) < 10:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__

    metadata = {"exception_type": type(error).__name__}
    if any(isinstance(item, InvalidModelResponseError) for item in chain):
        return {**metadata, "failure_kind": "invalid_model_response"}
    for item in chain:
        if isinstance(item, APIStatusError):
            status = item.status_code
            metadata.update(http_status=status, failure_kind=f"http_{status // 100}xx",
                            provider_request_id=_request_id(item.request_id))
            return metadata
    for types, kind in (
        ((httpx.ConnectTimeout,), "connect_timeout"),
        ((httpx.ReadTimeout,), "read_timeout"),
        ((httpx.WriteTimeout,), "write_timeout"),
        ((httpx.PoolTimeout,), "pool_timeout"),
        ((socket.gaierror,), "dns_failure"),
        ((ssl.SSLError,), "tls_failure"),
        ((APITimeoutError, httpx.TimeoutException), "timeout"),
        ((APIConnectionError, httpx.TransportError), "connection_failure"),
    ):
        if any(isinstance(item, types) for item in chain):
            metadata["failure_kind"] = kind
            return metadata
    metadata["failure_kind"] = "model_error"
    return metadata


class ModelTrace(BaseCallbackHandler):
    run_inline = True

    def __init__(self, sink, node: str | None, timeout: float):
        self.sink = sink
        self.node = node
        self.timeout = timeout
        self.call_id = str(uuid4())
        self.started = None

    def emit(self, event: str, status: str, detail: str, **data) -> None:
        entry = {"node": "service", "graph_node": self.node, "event": event, "status": status,
                 "detail": detail, "timestamp": datetime.now(UTC).isoformat(),
                 "model_call_id": self.call_id, **{k: v for k, v in data.items() if v is not None}}
        if self.started is not None and event != "model_request_started":
            entry["duration_ms"] = round((perf_counter() - self.started) * 1000)
        try:
            self.sink(entry)
        except Exception as error:
            LOGGER.warning("model_trace_publish_error type=%s", type(error).__name__)

    def request(self, request: httpx.Request) -> None:
        self.call_id = str(uuid4())
        self.started = perf_counter()
        backoff_seconds = request.extensions.get("model_retry_backoff_seconds")
        self.emit("model_request_started", "started", "模型请求已发起，等待 HTTP 响应。",
                  timeout_seconds=self.timeout, retry=True if backoff_seconds is not None else None,
                  retry_backoff_seconds=backoff_seconds)

    def response(self, response: httpx.Response) -> None:
        code = response.status_code
        request_id = _request_id(response.headers.get("x-request-id") or response.headers.get("request-id"))
        self.emit("model_response_headers", "error" if code >= 400 else "started",
                  f"收到模型服务 HTTP {code} 响应头。",
                  http_status=code, provider_request_id=request_id)

    def on_llm_end(self, response, **kwargs) -> None:
        if self.started is not None:
            self.emit("model_request_completed", "success", "模型响应已读取。")

    def on_llm_error(self, error, **kwargs) -> None:
        fields = failure_metadata(error)
        self.emit("model_request_failed", "error",
                  f"模型请求失败：{fields['failure_kind']}（{fields['exception_type']}）。", **fields)


def model_trace_options(timeout: float) -> dict:
    sink = _sink.get()
    if sink is None:
        return {"http_client": ModelHTTPClient(timeout=timeout)}
    trace = ModelTrace(sink, _node.get(), timeout)
    client = ModelHTTPClient(timeout=timeout, event_hooks={"request": [trace.request], "response": [trace.response]})
    clients = _clients.get()
    if clients is not None:
        clients.append(client)
    return {"callbacks": [trace], "http_client": client}
