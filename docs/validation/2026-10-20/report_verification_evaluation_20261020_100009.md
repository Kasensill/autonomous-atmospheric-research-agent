# Reflection / Verification 评测结果

- 生成时间：2026-10-20T09:59:57+08:00
- 案例数：8
- 完成案例：8
- Verdict 一致率：1.000
- 危险放行（应 revise 却 pass）：0
- 保守误拦（应 pass 却 revise）：0

## 逐题对照

| 案例 | 类型 | 人工标注 | 实际 verdict | 是否一致 |
| --- | --- | --- | --- | --- |
| supported_qualified_claim | supported_claim | pass | pass | 是 |
| absolute_overstatement | overstatement | revise | revise | 是 |
| missing_citation | citation_missing | revise | revise | 是 |
| nonexistent_citation | citation_invalid | revise | revise | 是 |
| claim_evidence_mismatch | citation_semantic_mismatch | revise | revise | 是 |
| model_conditions_not_station_observation | data_kind_mismatch | revise | revise | 是 |
| cold_without_climatology | internal_contradiction | revise | revise | 是 |
| qualified_model_weather | supported_weather | pass | pass | 是 |
