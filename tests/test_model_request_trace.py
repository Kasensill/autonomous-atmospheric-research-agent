import json
from threading import Event

import httpx
import pytest
from langgraph.graph import END, START, StateGraph
from openai import InternalServerError

from asgi_helpers import MemoryClient
from test_service_api import wait
from atmospheric_research_agent.agent.graph import _instrument_node
from atmospheric_research_agent.agent.model_trace import failure_metadata, model_trace_options
from atmospheric_research_agent.agent.state import ResearchState
from atmospheric_research_agent.service import engine
from atmospheric_research_agent.service.api import create_app


def test_verify_request_visible_before_response_and_failure_preserves_trace(tmp_path, monkeypatch):
    entered, release = Event(), Event()

    def verify(state):
        options = model_trace_options(90)
        trace = options["callbacks"][0]
        request = httpx.Request("POST", "https://example.invalid/secret-path?token=private",
                                headers={"Authorization": "Bearer private-secret"}, content="private-prompt")
        trace.request(request)
        entered.set()
        assert release.wait(5)
        response = httpx.Response(503, request=request, headers={"x-request-id": "safe-request-id"},
                                  content=b"private-response")
        trace.response(response)
        error = InternalServerError("private-error-body", response=response, body={"secret": "private"})
        trace.on_llm_error(error)
        raise error

    def build(*, checkpointer):
        graph = StateGraph(ResearchState)
        graph.add_node("verify_report", _instrument_node("verify_report", verify))
        graph.add_edge(START, "verify_report")
        graph.add_edge("verify_report", END)
        return graph.compile(checkpointer=checkpointer)

    monkeypatch.setattr(engine, "build_research_graph", build)
    with MemoryClient(create_app(db_path=tmp_path / "research.sqlite3")) as client:
        response = client.post("/runs", json={"question": "诊断测试"},
                               headers={"Idempotency-Key": "diagnostic"})
        run_id = response.json()["run_id"]
        endpoint = f"/runs/{run_id}/trace"
        try:
            assert entered.wait(5)
            started = client.get(endpoint).json()["events"][-1]
            assert started["event"] == "model_request_started"
            assert started["graph_node"] == "verify_report" and started["timeout_seconds"] == 90
        finally:
            release.set()
        failed = wait(client, run_id, "failed")
        assert "HTTP 503" in failed["error"]["message"]
        later = client.get(endpoint, params={"after": started["sequence"]}).json()["events"]
        headers, error = later[:2]
        assert headers["http_status"] == error["http_status"] == 503
        assert headers["provider_request_id"] == error["provider_request_id"] == "safe-request-id"
        assert headers["model_call_id"] == error["model_call_id"] == started["model_call_id"]
        assert error["failure_kind"] == "http_5xx"
        assert "private" not in json.dumps(later)


def test_wrapped_read_timeout_keeps_transport_phase():
    request = httpx.Request("POST", "https://example.invalid")
    try:
        try:
            raise httpx.ReadTimeout("private", request=request)
        except httpx.ReadTimeout as error:
            raise RuntimeError("private") from error
    except RuntimeError as error:
        assert failure_metadata(error) == {"exception_type": "RuntimeError", "failure_kind": "read_timeout"}


def test_real_client_wrapper_preserves_mocked_502_metadata():
    from langchain_openai import ChatOpenAI
    from atmospheric_research_agent.agent.model_trace import ModelTrace

    events = []
    trace = ModelTrace(events.append, "verify_report", 90)
    transport = httpx.MockTransport(lambda request: httpx.Response(
        502, json={"error": {"message": "private-server-message"}}, headers={"x-request-id": "probe-502"}))
    with httpx.Client(transport=transport, event_hooks={
        "request": [trace.request], "response": [trace.response],
    }) as http_client:
        model = ChatOpenAI(model="offline", api_key="offline-placeholder",
                           base_url="https://offline.invalid", max_retries=0,
                           http_client=http_client, callbacks=[trace])
        with pytest.raises(Exception) as captured:
            model.invoke("private-prompt")
    assert failure_metadata(captured.value)["http_status"] == 502
    assert [event["event"] for event in events] == [
        "model_request_started", "model_response_headers", "model_request_failed"]
    assert events[-1]["provider_request_id"] == "probe-502"
    assert "private" not in json.dumps(events)


@pytest.mark.parametrize("recovered", [True, False])
def test_invalid_model_response_replayed_once(recovered):
    from atmospheric_research_agent.agent.model_trace import (
        InvalidModelResponseError,
        ModelHTTPClient,
    )

    requests = []

    def respond(request):
        requests.append(request.read())
        if recovered and len(requests) == 2:
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                              content=b"private-invalid-response")

    with ModelHTTPClient(transport=httpx.MockTransport(respond)) as client:
        if recovered:
            assert client.post("https://offline.invalid", json={"model": "offline"}).json() == {
                "choices": [],
            }
        else:
            with pytest.raises(InvalidModelResponseError) as captured:
                client.post("https://offline.invalid", json={"model": "offline"})
            assert "private-invalid-response" not in str(captured.value)

    assert len(requests) == 2
    assert [json.loads(body) for body in requests] == [{"model": "offline"}] * 2


def test_503_retries_once_after_backoff_and_traces_it(monkeypatch):
    from atmospheric_research_agent.agent import model_trace

    actions, events = [], []
    monkeypatch.setattr(model_trace, "sleep", lambda seconds: actions.append(seconds))
    trace = model_trace.ModelTrace(events.append, "agent", 60)

    def respond(request):
        actions.append("send")
        return httpx.Response(503)

    with model_trace.ModelHTTPClient(transport=httpx.MockTransport(respond),
                                    event_hooks={"request": [trace.request]}) as client:
        assert client.post("https://offline.invalid", json={"model": "offline"}).status_code == 503

    assert actions == ["send", 2, "send"]
    assert events[1]["retry"] is True
    assert events[1]["retry_backoff_seconds"] == 2


@pytest.mark.parametrize("recovered", [True, False])
def test_fast_bad_gateway_is_replayed_once(recovered):
    from atmospheric_research_agent.agent.model_trace import ModelHTTPClient

    requests = []

    def respond(request):
        requests.append(request)
        if recovered and len(requests) == 2:
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(502, json={"error": "offline bad gateway"})

    with ModelHTTPClient(transport=httpx.MockTransport(respond)) as client:
        response = client.post("https://offline.invalid", json={"model": "offline"})

    assert response.status_code == (200 if recovered else 502)
    assert len(requests) == 2
