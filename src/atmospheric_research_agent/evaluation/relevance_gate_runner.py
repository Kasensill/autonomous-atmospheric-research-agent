"""用明确正负标签研究 rerank 门槛；校准与留出按 query 分开，保存原始分数。"""

import json
import re
from pathlib import Path
from datetime import datetime

from ..agent.retrieval.rerank import call_qwen_rerank
from ..agent.settings import RERANK_MODEL

ROOT = Path(__file__).resolve().parents[3]


def section_text(location: str, section: str) -> str:
    text = (ROOT / "data/knowledge_base" / location).read_text(encoding="utf-8")
    match = re.search(r"^## " + re.escape(section) + r"\s*\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    if not match:
        raise ValueError(f"找不到章节：{location} > {section}")
    title = next(line for line in text.splitlines() if line.startswith("# "))
    return f"{title}\n## {section}\n{match.group(1).strip()}"


def counts(rows: list[dict], threshold: float) -> dict:
    positives = [row for row in rows if row["relevant"]]
    negatives = [row for row in rows if not row["relevant"]]
    return {"positive_count": len(positives), "negative_count": len(negatives),
            "false_rejects": sum(row["score"] < threshold for row in positives),
            "false_accepts": sum(row["score"] >= threshold for row in negatives),
            "accepted_noise_ratio": sum(not row["relevant"] for row in rows if row["score"] >= threshold)
                / max(1, sum(row["score"] >= threshold for row in rows))}


def main():
    cases = json.loads((ROOT / "evals/relevance_gate_cases.json").read_text(encoding="utf-8"))
    rows = []
    errors = []
    for case in cases:
        labelled = [(ref, True) for ref in case["relevant"]] + [(ref, False) for ref in case["irrelevant"]]
        try:
            docs = [section_text(*ref) for ref, _ in labelled]
            results = call_qwen_rerank(case["query"], docs, top_k=len(docs))
            if {r["index"] for r in results} != set(range(len(docs))):
                raise ValueError("Rerank 未返回全部候选分数")
            for result in results:
                reference, relevant = labelled[result["index"]]
                rows.append({"case_id": case["id"], "split": case["split"], "query": case["query"],
                             "source": reference, "relevant": relevant, "score": result["relevance_score"]})
        except Exception as error:
            errors.append({"case_id": case["id"], "error": str(error)})
    calibration = [r for r in rows if r["split"] == "calibration"]
    holdout = [r for r in rows if r["split"] == "holdout"]
    thresholds = [i / 100 for i in range(1, 100)]
    # 仅用校准集选择阈值，优先不误拒正例，再减少误收；并列选较低值。
    threshold = min(thresholds, key=lambda t: (counts(calibration, t)["false_rejects"], counts(calibration, t)["false_accepts"], t)) if calibration else None
    payload = {"generated_at": datetime.now().astimezone().isoformat(), "model": RERANK_MODEL,
               "threshold": threshold, "rows": rows, "errors": errors,
               "before": {"calibration": counts(calibration, 0), "holdout": counts(holdout, 0)},
               "after": {"calibration": counts(calibration, threshold), "holdout": counts(holdout, threshold)} if threshold else {},
               "scope": "小型显式正负 query-passage 对；非完整检索候选分布，非独立人工盲测。"}
    output = ROOT / "docs/validation/2026-10-20/relevance_gate_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
