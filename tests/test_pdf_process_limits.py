"""不依赖 pypdf 的进程控制、缺依赖及诊断测试；全部离线。"""

import json
import subprocess
import sys
from time import monotonic
from types import SimpleNamespace

from atmospheric_research_agent.agent.tools import pdf_reader, source_reader


def test_missing_pypdf_is_a_diagnostic_tool_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    result = pdf_reader._extract_pdf_bytes(b"%PDF-test")
    assert result["status"] == "error"
    assert "未安装 pypdf" in result["message"]
    assert result["diagnostics"]["downloaded_bytes"] == len(b"%PDF-test")


def test_real_subprocess_timeout_terminates_wait_without_background_thread(monkeypatch):
    real_run = subprocess.run
    monkeypatch.setattr(pdf_reader, "PDF_PARSE_TIMEOUT_SECONDS", 0.1)

    def sleeping_process(command, **kwargs):
        assert kwargs["timeout"] == 0.1
        assert kwargs["stderr"] is subprocess.DEVNULL
        return real_run([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)

    monkeypatch.setattr(pdf_reader.subprocess, "run", sleeping_process)
    start = monotonic()
    result = pdf_reader.read_pdf_text(b"%PDF-test")
    assert monotonic() - start < 5
    assert result["status"] == "error" and "解析超时" in result["message"]


def test_oversize_does_not_start_worker(monkeypatch):
    monkeypatch.setattr(pdf_reader, "PDF_MAX_BYTES", 4)

    def forbidden(*args, **kwargs):
        raise AssertionError("不应启动解析进程")

    monkeypatch.setattr(pdf_reader.subprocess, "run", forbidden)
    assert "大小上限" in pdf_reader.read_pdf_text(b"%PDF-more")["message"]


def test_worker_invalid_output_does_not_expose_body(monkeypatch):
    monkeypatch.setattr(pdf_reader.subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout=b"private unstructured PDF body",
    ))
    result = pdf_reader.read_pdf_text(b"%PDF-test")
    assert result["status"] == "error"
    assert "private unstructured" not in json.dumps(result)


def test_parser_failure_reaches_source_tool_with_existing_diagnostics(monkeypatch):
    monkeypatch.setattr(pdf_reader, "read_pdf_text", lambda _: {
        "status": "error", "message": "PDF 已加密，无法读取。", "diagnostics": {"processed_pages": 0},
    })
    diagnostics = {"pdf_detection": {"url_suffix": True}, "downloaded_bytes": 9}
    result = source_reader._read_pdf_source(b"%PDF-test", "https://example.org/report.pdf", diagnostics)
    assert result["status"] == "error" and "已加密" in result["message"]
    assert result["diagnostics"]["pdf"]["processed_pages"] == 0
    assert result["diagnostics"]["pdf_detection"]["url_suffix"]


def test_range_notice_counts_towards_final_evidence_limit(monkeypatch):
    monkeypatch.setattr(source_reader, "MAX_SOURCE_CHARACTERS", 100)
    monkeypatch.setattr(pdf_reader, "read_pdf_text", lambda _: {
        "status": "success", "content": "[第 1 页]\n" + "a" * 80,
        "message": "仅读取了前 1 页，已截断。", "truncated": True,
        "diagnostics": {"processed_page_range": [1, 1]},
    })
    result = source_reader._read_pdf_source(b"%PDF", "https://example.org/file.pdf", {})
    assert len(result["content"]) <= 100
    assert result["diagnostics"]["pdf"]["evidence_character_limit_reached"]
    assert "已截断" in result["content"]
