"""PDF 拒绝、网页失败诊断及 Trace 状态的纯离线回归。"""

import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.graph import _instrument_node
from atmospheric_research_agent.agent.run import format_final_result
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.tools import source_reader


def fake_reader(monkeypatch, *, data=b"<html>private body</html>", headers=None, status=200,
                content=None, download_error=None, extract_error=None, no_response=False):
    calls = []

    def download(url, *, decode, with_headers):
        calls.append("download")
        assert decode is False and with_headers is True
        if download_error:
            raise download_error
        return None if no_response else SimpleNamespace(data=data, headers=headers or {}, status=status)

    def extract(*args, **kwargs):
        calls.append("extract")
        if extract_error:
            raise extract_error
        return content

    monkeypatch.setitem(sys.modules, "trafilatura", SimpleNamespace(
        fetch_response=download, extract=extract,
        extract_metadata=lambda *args, **kwargs: SimpleNamespace(
            title="来源", author=None, sitename=None, hostname="example.org", date=None),
    ))
    return calls


@pytest.mark.parametrize("url", [
    "https://example.org/report.pdf", "https://example.org/report.PDF?download=1",
    "[PDF](https://example.org/report.pdf#page=2)",
])
def test_pdf_suffix_is_rejected_without_download(monkeypatch, url):
    calls = fake_reader(monkeypatch)
    result = source_reader.read_public_source(url)
    assert result["message"] == source_reader.PDF_UNSUPPORTED_MESSAGE
    assert result["status"] == "error"
    assert result["diagnostics"]["pdf_detection"] == {
        "url_suffix": True, "content_header": False, "content_type": False,
    }
    assert result["diagnostics"]["downloaded_bytes"] == 0
    assert not result["diagnostics"]["download_attempted"]
    assert calls == []


@pytest.mark.parametrize("data,headers,header_hit,type_hit", [
    (b"%PDF-1.7\nprivate body", {}, True, False),
    (b"private body", {"Content-Type": "Application/PDF; charset=binary"}, False, True),
    (b"%PDF-1.7\nprivate body", {"content-type": "application/pdf"}, True, True),
])
def test_pdf_magic_or_response_type_is_rejected_before_extraction(monkeypatch, data, headers, header_hit, type_hit):
    calls = fake_reader(monkeypatch, data=data, headers=headers)
    result = source_reader.read_public_source("https://example.org/download?id=123")
    assert result["message"] == source_reader.PDF_UNSUPPORTED_MESSAGE
    assert result["diagnostics"]["pdf_detection"] == {
        "url_suffix": False, "content_header": header_hit, "content_type": type_hit,
    }
    assert result["diagnostics"]["downloaded_bytes"] == len(data)
    assert calls == ["download"]
    assert "private body" not in json.dumps(result)
    assert "%PDF" not in json.dumps(result)


def test_html_extraction_failure_has_non_pdf_message_and_byte_count(monkeypatch):
    data = "<html>不能提取的正文</html>".encode("utf-8")
    fake_reader(monkeypatch, data=data, headers={"content-type": "text/html; charset=utf-8"})
    result = source_reader.read_public_source("https://example.org/article")
    assert result["message"] == "网页正文读取失败：未能从网页提取可读正文。"
    assert not any(result["diagnostics"]["pdf_detection"].values())
    assert result["diagnostics"]["downloaded_bytes"] == len(data)
    assert result["diagnostics"]["content_type"] == "text/html"
    assert "不能提取的正文" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.parametrize("options,expected", [
    ({"no_response": True}, "网页下载失败或未获得响应内容。"),
    ({"status": 403}, "网页下载失败：HTTP 403。"),
    ({"download_error": OSError("private body")}, "网页下载失败（OSError）。"),
    ({"extract_error": ValueError("private body")}, "网页正文提取失败（ValueError）。"),
])
def test_download_and_extractor_errors_keep_diagnostics_without_body(monkeypatch, options, expected):
    fake_reader(monkeypatch, **options)
    result = source_reader.read_public_source("https://example.org/article")
    assert result["status"] == "error"
    assert result["message"] == expected
    assert result["diagnostics"]["download_attempted"]
    assert "downloaded_bytes" in result["diagnostics"]
    assert "private body" not in json.dumps(result)
    if options.get("no_response") or options.get("download_error"):
        assert result["diagnostics"]["downloaded_bytes"] is None


def test_invalid_url_and_missing_dependency_have_diagnostics(monkeypatch):
    result = source_reader.read_public_source("http://localhost/private")
    assert result["status"] == "error" and result["diagnostics"]["downloaded_bytes"] == 0
    monkeypatch.setitem(sys.modules, "trafilatura", None)
    result = source_reader.read_public_source("https://example.org/article")
    assert "未安装 trafilatura" in result["message"]
    assert result["diagnostics"]["downloaded_bytes"] == 0
    assert not result["diagnostics"]["download_attempted"]


def test_source_failure_trace_keeps_diagnostics_and_error_status(monkeypatch):
    fake_reader(monkeypatch, data=b"%PDF-1.7\nprivate body")
    result = source_reader.read_public_source("https://example.org/download")
    state = create_initial_state("读取公开来源")
    state["messages"] = [ToolMessage(
        name="read_source", tool_call_id="read_1", status="error",
        content=json.dumps(result, ensure_ascii=False),
    )]
    update = _instrument_node("record_results", nodes.record_results_node)(state)
    failed = next(e for e in update["trace"] if e["event"] == "tool_failed")
    assert failed["status"] == state["messages"][0].status == "error"
    assert failed["source_diagnostics"] == result["diagnostics"]
    assert next(e for e in update["trace"] if e["event"] == "node_completed")["status"] == "success"
    assert update["evidence"] == []
    assert update["tool_call_count"] == 1  # 失败调用依然消耗预算。
    output = format_final_result({**state, **update}, debug=True)
    assert '"source_diagnostics"' in output and '"content_header": true' in output
    assert "private body" not in output


@pytest.mark.parametrize("tool_name", ["search_web", "search_local_knowledge", "get_current_weather"])
def test_other_tool_failure_events_use_same_error_status(tool_name):
    state = create_initial_state("工具失败")
    state["messages"] = [ToolMessage(name=tool_name, tool_call_id="failed", status="error",
                                     content=json.dumps({"status": "error", "message": "离线失败"}))]
    update = _instrument_node("record_results", nodes.record_results_node)(state)
    assert next(e for e in update["trace"] if e["event"] == "tool_failed")["status"] == "error"
