# Reflection / Verification 评测结果

- 生成时间：2026-10-20T09:59:08+08:00
- 案例数：8
- 完成案例：0
- Verdict 一致率：0.000
- 危险放行（应 revise 却 pass）：0
- 保守误拦（应 pass 却 revise）：0

## 逐题对照

| 案例 | 类型 | 人工标注 | 实际 verdict | 是否一致 |
| --- | --- | --- | --- | --- |
| supported_qualified_claim | supported_claim | pass | ERROR | 运行失败 |
| absolute_overstatement | overstatement | revise | ERROR | 运行失败 |
| missing_citation | citation_missing | revise | ERROR | 运行失败 |
| nonexistent_citation | citation_invalid | revise | ERROR | 运行失败 |
| claim_evidence_mismatch | citation_semantic_mismatch | revise | ERROR | 运行失败 |
| model_conditions_not_station_observation | data_kind_mismatch | revise | ERROR | 运行失败 |
| cold_without_climatology | internal_contradiction | revise | ERROR | 运行失败 |
| qualified_model_weather | supported_weather | pass | ERROR | 运行失败 |

## 需复盘：supported_qualified_claim

- 运行错误：Connection error.

## 需复盘：absolute_overstatement

- 运行错误：Connection error.

## 需复盘：missing_citation

- 运行错误：Connection error.

## 需复盘：nonexistent_citation

- 运行错误：Connection error.

## 需复盘：claim_evidence_mismatch

- 运行错误：Connection error.

## 需复盘：model_conditions_not_station_observation

- 运行错误：Connection error.

## 需复盘：cold_without_climatology

- 运行错误：Connection error.

## 需复盘：qualified_model_weather

- 运行错误：Connection error.
