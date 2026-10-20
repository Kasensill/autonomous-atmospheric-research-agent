"""固定原有十二题，同一真实候选列表对照门槛前后，避免两次 API 波动。"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from .retrieval_cases import load_retrieval_cases
from .retrieval_metrics import calculate_retrieval_metrics, mean_metrics
from ..agent.tools.local_knowledge import compare_local_retrievers
from ..agent.retrieval.rerank import filter_relevance
from ..agent.settings import RERANK_MIN_SCORE

ROOT = Path(__file__).resolve().parents[3]


def main():
    before, after, results, errors = [], [], [], []
    for case in load_retrieval_cases(ROOT / "evals/retrieval_cases.json"):
        try:
            raw = compare_local_retrievers(case.question, include_rerank=True)
            if raw.get("rerank_status") != "success":
                raise RuntimeError(raw.get("rerank_message", "检索或 Rerank 失败"))
            candidates = raw["reranked_candidates"][:4]
            filtered = filter_relevance(candidates, RERANK_MIN_SCORE)
            metrics = [calculate_retrieval_metrics(items, case.expected_sources, top_k=4,
                       irrelevant_sources=case.irrelevant_sources) for items in (candidates, filtered)]
            before.append(metrics[0]); after.append(metrics[1])
            results.append({"id": case.case_id, "query": case.question, "before": asdict(metrics[0]),
                            "after": asdict(metrics[1]), "before_count": len(candidates), "after_count": len(filtered),
                            "candidates": candidates})
        except Exception as error:
            errors.append({"id": case.case_id, "error": str(error)})
    payload = {"generated_at": datetime.now().astimezone().isoformat(), "threshold": RERANK_MIN_SCORE,
               "completed": len(results), "errors": errors, "cases": results,
               "before": mean_metrics(before) if before else {}, "after": mean_metrics(after) if after else {}}
    path = ROOT / "docs/validation/2026-10-20/retrieval_gate_effect.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
