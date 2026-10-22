"""需要 pypdf 的 PDF 提取回归；缺依赖时逐项跳过，不联网安装。"""

from io import BytesIO
import json
import sys
from types import SimpleNamespace

import pytest

from atmospheric_research_agent.agent.tools import pdf_reader, source_reader
from atmospheric_research_agent.agent.settings import MAX_SOURCE_CHARACTERS


@pytest.fixture(autouse=True)
def pdf_library():
    return pytest.importorskip("pypdf")


def make_pdf(pdf_library, texts, *, password=None):
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = pdf_library.PdfWriter()
    for text in texts:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                     NameObject("/Subtype"): NameObject("/Type1"),
                                     NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
            })
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def fake_pages(monkeypatch, pdf_library, texts):
    pages = [SimpleNamespace(extract_text=lambda value=value: value) for value in texts]
    monkeypatch.setattr(pdf_library, "PdfReader", lambda *args, **kwargs: SimpleNamespace(
        pages=pages, is_encrypted=False, metadata={},
    ))


def test_generated_text_pdf_extracts_page_markers_and_skips_blank_page(pdf_library):
    data = make_pdf(pdf_library, ["Government weather analysis with enough readable text.", None,
                                  "Second analysis page with verified event information."])
    result = pdf_reader._extract_pdf_bytes(data)
    assert result["status"] == "success"
    assert "[第 1 页]" in result["content"] and "[第 3 页]" in result["content"]
    assert "[第 2 页]" not in result["content"]
    assert result["diagnostics"]["skipped_empty_pages"] == [2]
    assert result["diagnostics"]["processed_page_range"] == [1, 3]
    assert not result["truncated"]


def test_generated_pdf_page_limit_has_explicit_partial_read_notice(pdf_library):
    data = make_pdf(pdf_library, ["Enough readable weather event facts for every page."] * 32)
    result = pdf_reader._extract_pdf_bytes(data)
    assert result["status"] == "success" and result["truncated"]
    assert "仅读取了前 30 页" in result["message"] and "已截断" in result["message"]
    assert "[第 30 页]" in result["content"] and "[第 31 页]" not in result["content"]
    assert result["diagnostics"]["processed_page_range"] == [1, 30]


def test_generated_pdf_character_limit_is_explicit(monkeypatch, pdf_library):
    monkeypatch.setattr(pdf_reader, "PDF_MAX_EXTRACTED_CHARACTERS", 200)
    result = pdf_reader._extract_pdf_bytes(make_pdf(pdf_library, ["Weather facts " * 20] * 2))
    assert result["status"] == "success" and result["truncated"]
    assert len(result["content"]) <= 200
    assert "200 字符上限，已截断" in result["message"]


def test_scanned_or_blank_pdf_is_only_described_as_suspected(pdf_library):
    result = pdf_reader._extract_pdf_bytes(make_pdf(pdf_library, [None, None, None]))
    assert result["status"] == "error"
    assert result["message"] == "疑似扫描版或无文字层，需要 OCR。"
    assert "content" not in result


def test_mostly_sparse_pages_trigger_ocr_warning(monkeypatch, pdf_library):
    fake_pages(monkeypatch, pdf_library, [""] * 4 + ["Readable weather facts with more than twenty letters."])
    result = pdf_reader._extract_pdf_bytes(b"%PDF-fake")
    assert "疑似扫描版" in result["message"]
    assert result["diagnostics"]["quality"]["sparse_page_ratio"] == 0.8


@pytest.mark.parametrize("bad_text", ["\ufffd" * 20, "\x01" * 20, "(cid:123)" * 10])
def test_encoding_defects_fail_without_body_logging(monkeypatch, pdf_library, bad_text):
    fake_pages(monkeypatch, pdf_library, ["可读的政府天气报告内容" * 10 + bad_text])
    result = pdf_reader._extract_pdf_bytes(b"%PDF-fake")
    assert result["status"] == "error" and result["message"] == "字体编码异常，无法可靠提取。"
    assert "content" not in result
    assert "可读的政府天气报告内容" not in json.dumps(result, ensure_ascii=False)


def test_oversized_pdf_is_rejected_before_parser(monkeypatch, pdf_library):
    monkeypatch.setattr(pdf_library, "PdfReader", lambda *a, **kw: pytest.fail("超大文件不应解析"))
    data = b"%PDF" + b"x" * pdf_reader.PDF_MAX_BYTES
    assert "超过 10MB" in pdf_reader._extract_pdf_bytes(data)["message"]
    assert "超过 10MB" in pdf_reader.read_pdf_text(data)["message"]


def test_encrypted_pdf_has_distinct_error(pdf_library):
    data = make_pdf(pdf_library, ["Private PDF content must not enter failure diagnostics."], password="offline-test-only")
    result = pdf_reader._extract_pdf_bytes(data)
    assert result["status"] == "error" and "PDF 已加密，无法读取" in result["message"]
    assert "Private PDF content" not in json.dumps(result)


def test_parser_exception_is_safe_and_distinct(monkeypatch, pdf_library):
    def broken(*args, **kwargs):
        raise ValueError("private body must never be logged")

    monkeypatch.setattr(pdf_library, "PdfReader", broken)
    result = pdf_reader._extract_pdf_bytes(b"%PDF-broken")
    assert result["message"] == "PDF 解析异常，无法提取可读文本。"
    assert result["diagnostics"]["exception_type"] == "ValueError"
    assert "private body" not in json.dumps(result)


def test_extracted_text_limit_uses_independent_constant(monkeypatch, pdf_library):
    fake_pages(monkeypatch, pdf_library, ["天气事实" * 30_000])
    result = pdf_reader._extract_pdf_bytes(b"%PDF-fake")
    assert result["status"] == "success" and result["truncated"]
    assert len(result["content"]) == 100_000
    assert "100000 字符上限" in result["message"]


def test_final_evidence_limit_and_page_ranges_are_not_full_document_claims(monkeypatch, pdf_library):
    fake_pages(monkeypatch, pdf_library, ["Page one content. " * 700, "Page two content. " * 700])
    parsed = pdf_reader._extract_pdf_bytes(b"%PDF-fake")
    monkeypatch.setattr(pdf_reader, "read_pdf_text", lambda _: parsed)
    result = source_reader._read_pdf_source(b"%PDF-fake", "https://example.org/report.pdf", {})
    assert result["status"] == "success" and result["truncated"]
    assert len(result["content"]) <= MAX_SOURCE_CHARACTERS
    assert "已截断" in result["content"] and "未按相关度选段" in result["message"]
    assert result["diagnostics"]["pdf"]["processed_page_range"] == [1, 2]
    assert result["diagnostics"]["pdf"]["returned_page_numbers"] == [1]


def test_source_reader_generated_pdf_through_real_worker(monkeypatch, pdf_library):
    data = make_pdf(pdf_library, ["Official weather event report with sufficient readable facts."])
    monkeypatch.setitem(sys.modules, "trafilatura", SimpleNamespace(
        fetch_response=lambda *a, **kw: SimpleNamespace(data=data, status=200, headers={"Content-Type": "application/pdf"}),
    ))
    result = source_reader.read_public_source("https://example.org/report.pdf")
    assert result["status"] == "success"
    assert "[第 1 页]" in result["content"]
    assert result["diagnostics"]["pdf_detection"]["url_suffix"]
    assert result["diagnostics"]["downloaded_bytes"] == len(data)
