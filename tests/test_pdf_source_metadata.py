"""PDF 显示元数据清理的离线测试，不改变正文提取或依赖实际 PDF。"""

import json

import pytest
from langchain_core.messages import ToolMessage

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.tools import pdf_reader, source_reader


@pytest.mark.parametrize("title", [
    None, "", " \n ", "report.pdf", "REPORT.PDF", "Microsoft Word - report.docx",
    "报告.pdf", "https://example.org/report.pdf", "https://example.org/download?id=1",
    "www.example.org/report", "example.org/report", "https%3A%2F%2Fexample.org%2Freport.pdf",
    "C:\\exports\\report.pdf", "report%2Epdf", "download",
])
def test_missing_file_or_url_title_falls_back_to_domain(title):
    name, _ = source_reader._pdf_display_metadata(title, None, "https://mem.gov.cn/download")
    assert name == "PDF：mem.gov.cn"


@pytest.mark.parametrize("title", ["郑州特大暴雨灾害调查报告", "Meteorological investigation report"])
def test_descriptive_pdf_metadata_title_is_preferred(title):
    name, author = source_reader._pdf_display_metadata(" " + title + "\n", "气象研究机构", "https://example.org/report.pdf")
    assert name == title
    assert author == "气象研究机构"


@pytest.mark.parametrize("author", [None, "", " USER ", "admin", "Administrator", "unknown", "anonymous", "N/A", "default user"])
def test_empty_or_generic_author_is_suppressed(author):
    _, displayed = source_reader._pdf_display_metadata("正式报告", author, "https://example.org/report.pdf")
    assert displayed is None


def test_source_display_cleanup_preserves_body_location_and_diagnostics(monkeypatch):
    content = "[第 1 页]\n提取的证据正文完全保持原样。"
    monkeypatch.setattr(pdf_reader, "read_pdf_text", lambda _: {
        "status": "success", "content": content, "truncated": False, "message": "",
        "title": "https://mem.gov.cn/report.pdf", "author": "user",
        "diagnostics": {"processed_page_range": [1, 1]},
    })
    url = "https://mem.gov.cn/report.pdf"
    result = source_reader._read_pdf_source(b"%PDF-test", url, {"downloaded_bytes": 9})
    assert result["content"] == content
    assert not result["truncated"]
    assert result["source"]["display_name"] == "PDF：mem.gov.cn"
    assert result["source"]["author"] is None
    assert result["source"]["location"] == url
    assert result["diagnostics"]["pdf"]["processed_page_range"] == [1, 1]
    state = create_initial_state("读取 PDF")
    state["messages"] = [ToolMessage(name="read_source", tool_call_id="pdf", content=json.dumps(result))]
    state.update(nodes.record_results_node(state))
    assert state["evidence"][0]["content"] == content
    assert "author_or_organization" not in state["evidence"][0]
    references = nodes._reference_section(nodes._numbered_evidence(state), report="引用 [E1]")
    assert "PDF：mem.gov.cn" in references
    assert "来源：user" not in references


def test_real_author_remains_visible_in_reference_list(monkeypatch):
    monkeypatch.setattr(pdf_reader, "read_pdf_text", lambda _: {
        "status": "success", "content": "[第 1 页]\n正文。", "truncated": False,
        "title": "正式气象调查报告", "author": "研究机构甲", "diagnostics": {},
    })
    result = source_reader._read_pdf_source(b"%PDF-test", "https://example.org/report.pdf", {})
    state = create_initial_state("读取 PDF")
    state["messages"] = [ToolMessage(name="read_source", tool_call_id="pdf", content=json.dumps(result))]
    state.update(nodes.record_results_node(state))
    references = nodes._reference_section(nodes._numbered_evidence(state), report="引用 [E1]")
    assert "正式气象调查报告" in references and "来源：研究机构甲" in references
