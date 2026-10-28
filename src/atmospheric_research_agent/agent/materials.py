"""命令行材料的运行时加载：只打开显式路径，不持久化、不执行材料中的指令。"""

from hashlib import sha256
from pathlib import Path
import re
import unicodedata

from .settings import (
    MATERIAL_MAX_FILES, MATERIAL_MAX_CHARS_TOTAL, MATERIAL_MAX_GARBLED_RATIO,
    MATERIAL_REFERENCE_WORDS, MATERIAL_TEXT_TASK_WORDS, MATERIAL_VERIFICATION_WORDS,
    MAX_SOURCE_CHARACTERS, MATERIAL_BEGIN, MATERIAL_END,
)
from .tools.pdf_reader import read_pdf_text


def escape_material_markers(text: str) -> str:
    """等长转义边界标记，避免原文闭合材料块；不解析其中的路径或指令。"""
    for marker in (MATERIAL_BEGIN, MATERIAL_END):
        text = text.replace(marker, marker.replace("<", "＜").replace(">", "＞"))
    return text


def decide_material_role(question: str, requested: str = "auto") -> tuple[str, str]:
    if requested not in {"auto", "clue", "analysis", "both"}:
        raise ValueError("material-role 必须是 auto、clue、analysis 或 both。")
    if requested != "auto":
        return requested, "命令行显式指定"
    reference = any(word in question for word in MATERIAL_REFERENCE_WORDS)
    text_task = any(word in question for word in MATERIAL_TEXT_TASK_WORDS)
    verification = any(word in question for word in MATERIAL_VERIFICATION_WORDS)
    if text_task and verification:
        return "both", "同时包含文本任务词与核实词"
    if reference and text_task and not verification:
        return "analysis", "同时包含材料指代词与文本任务词，且没有核实词"
    return "clue", "未满足文本分析条件，保守作为待核实线索"


def _text_error(text: str) -> str | None:
    if not text.strip():
        return "材料为空，没有可用正文。"
    bad = text.count("\ufffd") + sum(
        unicodedata.category(ch) == "Cc" and ch not in "\n\r\t" for ch in text
    )
    if bad / max(1, len(text)) > MATERIAL_MAX_GARBLED_RATIO:
        return "材料含大量乱码或异常控制字符，无法可靠读取。"
    return None


def _pieces(text: str, pdf: bool) -> list[tuple[str, str, int | None]]:
    if pdf:
        markers = list(re.finditer(r"(?m)^\[第 (\d+) 页\]\n", text))
        return [(text[marker.start():markers[index + 1].start() if index + 1 < len(markers) else len(text)],
                 f"第 {marker[1]} 页", int(marker[1])) for index, marker in enumerate(markers)]
    # 保留分段分隔符；定位按非空段落计，长段落再按证据正文上限切片。
    return [(match.group(), f"第 {index} 段", None) for index, match in enumerate(
        (m for m in re.finditer(r"[\s\S]+?(?:\r?\n[ \t]*\r?\n|\Z)", text) if m.group().strip()), 1
    )]


def load_materials(paths: list[str | Path], question: str, requested_role: str = "auto") -> dict:
    """返回可并入初始状态的证据、覆盖摘要和 Trace；加载不消耗研究预算。"""
    role, reason = decide_material_role(question, requested_role)
    trace = [{"node": "material_loader", "event": "material_role_decided",
              "detail": f"材料角色：{role}；{reason}。", "role": role, "reason": reason}]
    evidence, materials = [], []
    remaining = MATERIAL_MAX_CHARS_TOTAL
    seen = set()
    for index, supplied_path in enumerate(paths):
        path = Path(supplied_path)
        name = path.name
        error = None
        if index >= MATERIAL_MAX_FILES:
            error = f"超过材料数量上限 {MATERIAL_MAX_FILES}，未读取。"
        elif path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            error = "不支持此材料类型；仅支持 TXT、MD 和有文字层的 PDF。"
        if error:
            trace.append({"node": "material_loader", "event": "material_skipped",
                          "detail": f"{name}：{error}", "display_name": name, "status": "error"})
            continue
        try:
            # 唯一文件读取入口；内容中的路径、链接、文件名从不用于 I/O。
            data = path.read_bytes()
        except OSError:
            trace.append({"node": "material_loader", "event": "material_skipped",
                          "detail": f"{name}：无法读取材料文件。", "display_name": name, "status": "error"})
            continue
        material_id = sha256(data).hexdigest()[:8]
        if material_id in seen:
            trace.append({"node": "material_loader", "event": "material_skipped",
                          "detail": f"{name}：与本批已加载材料内容相同，未重复送入。", "status": "error"})
            continue
        pdf = path.suffix.lower() == ".pdf"
        diagnostics = {}
        parse_note = ""
        if pdf:
            parsed = read_pdf_text(data)
            error = parsed.get("message") if parsed.get("status") != "success" else None
            text = parsed.get("content", "") if not error else ""
            diagnostics = parsed.get("diagnostics", {})
            parse_note = parsed.get("message") or ""
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text, error = "", "文本不是有效 UTF-8，无法可靠读取；请转换编码后重试。"
        error = error or _text_error(text)
        if error:
            trace.append({"node": "material_loader", "event": "material_skipped",
                          "detail": f"{name}：{error}", "display_name": name,
                          "material_id": material_id, "status": "error"})
            continue
        pieces = _pieces(escape_material_markers(text), pdf)
        sent, ranges, sent_pages = 0, [], set()
        for content, locator, page in pieces:
            offset = 0
            while offset < len(content) and remaining:
                part = content[offset:offset + min(MAX_SOURCE_CHARACTERS, remaining)]
                located = f"{locator}，字符 {offset + 1}–{offset + len(part)}"
                evidence.append({
                    "evidence_id": f"material-{material_id}-{len(evidence) + 1}",
                    "content": part, "source_type": "user_material", "material_id": material_id,
                    "source_display_name": name, "source_location": name, "locator": located,
                    "verification_status": "user_provided_unverified", "role": role,
                    "tool_name": "material_loader",
                })
                sent += len(part)
                remaining -= len(part)
                offset += len(part)
                ranges.append(located)
                if page is not None:
                    sent_pages.add(page)
        record = {"material_id": material_id, "display_name": name, "role": role,
                  "verification_status": "user_provided_unverified", "parsed_characters": len(text),
                  "parsed_pages": diagnostics.get("processed_pages", 0),
                  "document_pages": diagnostics.get("total_pages", 0),
                  "sent_characters": sent, "sent_pages": sorted(sent_pages), "sent_ranges": ranges,
                  "parse_note": parse_note, "truncated": bool(parsed.get("truncated")) if pdf else False}
        record["truncated"] = record["truncated"] or sent < len(text)
        materials.append(record)
        seen.add(material_id)
        trace.extend([
            {"node": "material_loader", "event": "material_loaded", "material_id": material_id,
             "detail": f"已解析 {name}（{material_id}）；用户提供、未独立核实。"},
            {"node": "material_loader", "event": "material_coverage", "material_id": material_id,
             "detail": coverage_text([record]), "coverage": record},
        ])
    return {"materials": materials, "material_role": role, "evidence": evidence, "trace": trace}


def _compact_numbers(numbers: list[int], unit: str) -> str:
    """只合并连续编号，不把未送入的页或段补进范围。"""
    values = sorted(set(numbers))
    groups = []
    for value in values:
        if groups and value == groups[-1][1] + 1:
            groups[-1][1] = value
        else:
            groups.append([value, value])
    return "第 " + "、".join(str(start) if start == end else f"{start}–{end}" for start, end in groups) + f" {unit}"


def coverage_text(materials: list[dict]) -> str:
    """面向报告、提示词与默认界面的摘要；精确 sent_ranges 原样保留在记录中。"""
    lines = []
    for record in materials:
        pdf = bool(record.get("document_pages")) or record["display_name"].lower().endswith(".pdf")
        parsed = (f"{record['parsed_pages']} 页／" if pdf else "") + f"{record['parsed_characters']} 字符"
        if not record["sent_characters"]:
            scope = "未送入正文"
        elif pdf and record.get("sent_pages"):
            scope = "涉及" + _compact_numbers(record["sent_pages"], "页")
        else:
            paragraphs = [int(match[1]) for span in record["sent_ranges"]
                          if (match := re.match(r"第 (\d+) 段[，,]", span))]
            scope = ("涉及" + _compact_numbers(paragraphs, "段") if paragraphs
                     else "已送入片段，精确定位见材料记录")
        lines.append(
            f"材料覆盖：{record['display_name']}；已解析 {parsed}；"
            f"本轮送入模型 {record['sent_characters']} 字符（{scope}）。"
            + (f" {record['parse_note']}" if record.get("parse_note") else "")
            + (" 已截断，仅分析已送入片段。" if record.get("truncated") else "")
        )
    return "\n".join(lines)
