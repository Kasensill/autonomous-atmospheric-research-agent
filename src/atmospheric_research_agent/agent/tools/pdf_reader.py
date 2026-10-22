"""有文字层 PDF 的受限提取；子进程超时后终止，不执行 OCR。"""

from io import BytesIO
import json
import re
import subprocess
import sys
import unicodedata

from atmospheric_research_agent.agent.settings import (
    PDF_MAX_BYTES, PDF_MAX_PAGES, PDF_MAX_EXTRACTED_CHARACTERS,
    PDF_PARSE_TIMEOUT_SECONDS, PDF_MIN_PAGE_TEXT_CHARACTERS,
    PDF_MAX_SPARSE_PAGE_RATIO, PDF_MAX_GARBLED_RATIO, PDF_MIN_CID_OCCURRENCES,
)


def _error(message: str, diagnostics: dict) -> dict:
    total = diagnostics.get("total_pages")
    processed = diagnostics.get("processed_pages", 0)
    if total and processed and processed < total:
        message += f" 仅读取了前 {processed} 页（共 {total} 页），未完整读取。"
    if diagnostics.get("extraction_character_limit_reached"):
        message += f" 提取文本达到 {PDF_MAX_EXTRACTED_CHARACTERS} 字符上限，已截断。"
    return {"status": "error", "message": message, "diagnostics": diagnostics}


def _extract_pdf_bytes(data: bytes) -> dict:
    """仅在解析进程内运行；可用假 PdfReader 做离线边界测试。"""
    diagnostics = {"downloaded_bytes": len(data), "total_pages": None,
                   "processed_pages": 0, "processed_page_range": [], "text_page_numbers": [],
                   "skipped_empty_pages": [], "extracted_characters": 0}
    if len(data) > PDF_MAX_BYTES:
        return _error(f"PDF 文件超过 {PDF_MAX_BYTES / (1024 * 1024):g}MB 大小上限，未解析。", diagnostics)
    try:
        from pypdf import PdfReader
    except ImportError:
        return _error("未安装 pypdf，请安装项目依赖后重试。", diagnostics)
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = bool(reader.decrypt(""))
            except Exception:
                unlocked = False
            if not unlocked:
                return _error("PDF 已加密，无法读取；请提供可读取的未加密版本。", diagnostics)
        total_pages = len(reader.pages)
        diagnostics["total_pages"] = total_pages
        blocks = []
        text_parts = []
        length = sparse_pages = 0
        character_limit = False
        for index in range(min(total_pages, PDF_MAX_PAGES)):
            remaining = PDF_MAX_EXTRACTED_CHARACTERS - length
            # 为页码标记留足空间，不输出半个标记。
            marker = f"[第 {index + 1} 页]\n"
            if remaining <= len(marker) + 2:
                character_limit = True
                break
            text = reader.pages[index].extract_text() or ""
            diagnostics["processed_pages"] = index + 1
            diagnostics["processed_page_range"] = [1, index + 1]
            if sum(ch.isalnum() for ch in text) < PDF_MIN_PAGE_TEXT_CHARACTERS:
                sparse_pages += 1
            if not text.strip():
                diagnostics["skipped_empty_pages"].append(index + 1)
                continue
            separator = "\n\n" if blocks else ""
            block = separator + marker + text
            if len(block) > remaining:
                block = block[:remaining]
                character_limit = True
            blocks.append(block)
            text_parts.append(block[len(separator) + len(marker):])
            diagnostics["text_page_numbers"].append(index + 1)
            length += len(block)
            if character_limit:
                break
        content = "".join(blocks)
        quality_text = "".join(text_parts)
        denominator = max(1, len(quality_text))
        cid_count = len(re.findall(r"\(cid:", quality_text, flags=re.IGNORECASE))
        quality = {
            "replacement_ratio": quality_text.count("\ufffd") / denominator,
            "control_ratio": sum(unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in quality_text) / denominator,
            "cid_prefix_ratio": cid_count * len("(cid:") / denominator,
            "cid_occurrences": cid_count,
            "sparse_page_ratio": sparse_pages / max(1, diagnostics["processed_pages"]),
        }
        diagnostics.update(quality=quality, extracted_characters=len(content),
                           page_limit_reached=total_pages > PDF_MAX_PAGES,
                           extraction_character_limit_reached=character_limit)
        if (quality["replacement_ratio"] > PDF_MAX_GARBLED_RATIO
                or quality["control_ratio"] > PDF_MAX_GARBLED_RATIO
                or (cid_count >= PDF_MIN_CID_OCCURRENCES and quality["cid_prefix_ratio"] > PDF_MAX_GARBLED_RATIO)):
            return _error("字体编码异常，无法可靠提取。", diagnostics)
        if not quality_text.strip() or quality["sparse_page_ratio"] >= PDF_MAX_SPARSE_PAGE_RATIO:
            return _error("疑似扫描版或无文字层，需要 OCR。", diagnostics)
        truncated = diagnostics["processed_pages"] < total_pages or character_limit
        messages = []
        if diagnostics["processed_pages"] < total_pages:
            messages.append(f"仅读取了前 {diagnostics['processed_pages']} 页（共 {total_pages} 页），已截断。")
        if character_limit:
            messages.append(f"提取文本达到 {PDF_MAX_EXTRACTED_CHARACTERS} 字符上限，已截断。")
        metadata = reader.metadata or {}
        title = metadata.get("/Title")
        author = metadata.get("/Author")
        return {"status": "success", "content": content, "truncated": truncated,
                "message": " ".join(messages), "diagnostics": diagnostics,
                "title": title[:250] if isinstance(title, str) else None,
                "author": author[:250] if isinstance(author, str) else None}
    except Exception as error:
        # 不输出第三方异常原文；其中可能含 PDF 原始内容。
        diagnostics["exception_type"] = type(error).__name__
        return _error("PDF 解析异常，无法提取可读文本。", diagnostics)


def read_pdf_text(data: bytes) -> dict:
    """限制解析进程的等待时间；不包括下载，不承诺严格内存隔离。"""
    diagnostics = {"downloaded_bytes": len(data), "parse_timeout_seconds": PDF_PARSE_TIMEOUT_SECONDS}
    if len(data) > PDF_MAX_BYTES:
        return _error(f"PDF 文件超过 {PDF_MAX_BYTES / (1024 * 1024):g}MB 大小上限，未解析。", diagnostics)
    try:
        # subprocess.run 超时会 kill 并 wait 子进程；不留下继续解析的后台线程。
        completed = subprocess.run(
            [sys.executable, "-m", "atmospheric_research_agent.agent.tools.pdf_reader"],
            input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=PDF_PARSE_TIMEOUT_SECONDS, check=False,
        )
        if completed.returncode != 0:
            return _error("PDF 解析进程异常退出。", diagnostics)
        result = json.loads(completed.stdout)
        if not isinstance(result, dict) or result.get("status") not in {"success", "error"}:
            raise ValueError("invalid parser result")
        if not isinstance(result.get("diagnostics"), dict):
            raise ValueError("invalid parser diagnostics")
        if result["status"] == "success" and not isinstance(result.get("content"), str):
            raise ValueError("invalid parser content")
        result.setdefault("diagnostics", {}).update(parse_timeout_seconds=PDF_PARSE_TIMEOUT_SECONDS)
        return result
    except subprocess.TimeoutExpired:
        return _error(f"PDF 解析超时（{PDF_PARSE_TIMEOUT_SECONDS} 秒），已终止解析进程。", diagnostics)
    except (OSError, ValueError, TypeError):
        return _error("PDF 解析进程异常，未取得有效结果。", diagnostics)


if __name__ == "__main__":
    # 标准输入只接收 PDF 字节；标准输出仅用于结构化结果，不记录到日志。
    payload = sys.stdin.buffer.read(PDF_MAX_BYTES + 1)
    sys.stdout.write(json.dumps(_extract_pdf_bytes(payload), ensure_ascii=True))
