"""将人工确认许可与页范围的 PDF 文字层导入语料，不调用模型、不下载文件。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def extract_page(page) -> str:
    """按 PDF 原生文本块顺序保留段落；不把二维公式／图像转写成可靠公式。"""
    paragraphs = []
    for block in page.get_text("blocks", sort=False):
        if block[6] != 0:
            continue
        text = block[4].strip()
        if re.match(r"^(?:REFERENCES|References)(?:\s|$)", text):
            paragraphs.append("REFERENCES")
            break
        if not text or re.fullmatch(r"\d+(?:-\d+)?", text):
            continue
        if len(re.findall(r"[A-Za-z]", text)) < 15:
            continue  # 图内坐标数字和孤立标签，不是连贯正文。
        if re.search(r"(?:@|Downloaded from|wileyonlinelibrary\.com|Frontiers in Environmental Science \|)", text):
            continue
        text = text.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\u00ad", "")
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        # 保留连字符，不猜测哪些是复合词，哪些是排版断词。
        text = re.sub(r"-\n(?=[a-z])", "-", text)
        text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
        paragraphs.append(text)
    return "\n\n".join(paragraphs)


def import_manifest(manifest: Path, raw_dir: Path, knowledge_base: Path) -> list[dict]:
    import pymupdf

    records = json.loads(manifest.read_text(encoding="utf-8"))
    inventory = []
    for record in records:
        source = raw_dir / record["pdf"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"原件摘要不匹配：{record['pdf']}")
        if record["license"] not in {"CC-BY-4.0", "CC-BY", "US-GOV-PUBLIC-DOMAIN"}:
            raise ValueError("尚未确认可用的来源许可。")
        metadata = {key: record.get(key, "") for key in (
            "source_url", "source_author", "source_title", "published_at", "license",
            "license_url", "content_kind", "source_scope", "doi",
        )}
        metadata.update(source_pdf_sha256=digest, source_language="en")
        lines = [f"# {record['title']}",
                 "<!-- corpus-meta: " + json.dumps(metadata, ensure_ascii=False) + " -->", "",
                 "## 资料说明", "",
                 f"原作者：{record['source_author']}。原文：{record['source_title']}。",
                 f"来源：{record['source_url']}。许可：{record['license']}（{record['license_url']}）。",
                 "以下为英文原文文字层选段，不是 AI 总结或翻译；页码是 PDF 文件页序，不是印刷页码。",
                 "提取变更：移除图像、孤立坐标／页码、联系邮箱及下载水印，合并段内排版换行；保留原文论断。",
                 "PDF 双栏与跨页断词可能保留排版痕迹；图、表及二维公式请回原文核对，不据文本自行恢复数值或运算。",
                 f"适用范围：{record['source_scope']}", "", "## 原文", ""]
        pages = []
        with pymupdf.open(source) as document:
            for first, last in record["pages"]:
                if first < 1 or last > len(document) or first > last:
                    raise ValueError(f"页码越界：{record['pdf']}")
                for number in range(first, last + 1):
                    text = extract_page(document[number - 1])
                    if not text:
                        continue
                    if text.count("\ufffd") / max(1, len(text)) > .01:
                        raise ValueError(f"疑似乱码：{record['pdf']} 第 {number} 页")
                    # 参考文献保留在原件，不作为可支持科学结论的检索正文。
                    text = re.split(r"(?m)(?:^|\n\n)(?:REFERENCES|References)\s*(?:\n|$)", text)[0]
                    if text.strip():
                        lines.extend([f"### PDF第 {number} 页", "", text.strip(), ""])
                        pages.append(number)
                    if re.search(r"(?:^|\n\n)(?:REFERENCES|References)(?:\n|$)", extract_page(document[number - 1])):
                        break
        target = knowledge_base / record["target"]
        if not target.resolve().is_relative_to(knowledge_base.resolve()):
            raise ValueError("输出不在知识库内。")
        target.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines).rstrip() + "\n"
        target.write_text(content, encoding="utf-8")
        inventory.append({"file": record["target"], "characters": len(content),
                          "pdf_pages": pages, "source_sha256": digest})
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--knowledge-base", type=Path, default=Path("data/knowledge_base"))
    args = parser.parse_args()
    print(json.dumps(import_manifest(args.manifest, args.raw_dir, args.knowledge_base), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
