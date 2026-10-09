# 本地检索评测（V1.6）

这里的评测只衡量内部知识库的检索质量，不衡量网页搜索、完整 Agent 的工具决策或最终报告文笔。

## 比较对象

同一条问题会分别比较：

1. `vector`：纯向量检索；
2. `hybrid_rrf`：向量召回与 BM25 召回经 RRF 融合；
3. `hybrid_rrf_rerank`：RRF 候选再经 Qwen Rerank 精排。

三者均按 Top 4 计算指标。前两者是对照组；当前生产工具的默认路径是第三种。

## 标注原则

`retrieval_cases.json` 中的 `expected_sources` 标注为“这个问题应被召回的现有知识片段”，
而不是唯一标准答案。每一条通过 `location + section` 精确定位，避免同一篇资料中无关章节被
误判为命中。

问题集同时包含精确术语、语义改写、多机制和结论边界四类问题。部分条目会显式标出当前
知识库的资料缺口；这些案例用于区分“检索失败”和“资料不足”，不应被误解为模型必然失败。

## 指标

- `Hit@K`：Top-K 至少有一条人工标注的相关片段；
- `MRR@K`：Top-K 内首条相关片段名次的倒数；
- `Coverage@K`：所有人工标注的相关片段中，Top-K 实际覆盖的比例。

指标计算代码位于 `src/atmospheric_research_agent/evaluation/retrieval_metrics.py`，其单元测试
不访问外部模型或向量库。

## 运行真实对比

在已完成入库、且 `.env` 具备 DashScope Embedding 与 Rerank 配置的前提下，运行：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.evaluation.retrieval_runner --top-k 4
```

它会对 12 道题各运行一次真实的双路检索和 Rerank，再拆分为 A/B/C 对照：纯向量、RRF、
RRF + Rerank。终端会给出两份报告路径；`evals/results/` 下的 Markdown 供人阅读，JSON 供
后续分析。报告文件是本地运行产物，默认不提交到 Git。

## 报告核验评测（V2）

`report_verification_cases.json` 是 Reflection / Verification 的最小人工评测集。它不评估
最终报告文笔，而是测试审查器能否区分合格草稿和五类常见风险：绝对化过度主张、缺失引用、
不存在的引用编号，以及“编号存在但正文不支持该主张”的语义错配。下一步会用同一批案例
真实调用审查模型，统计它对 `pass / revise` 的判断是否与人工标注一致：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.evaluation.report_verification_runner
```

结果除 Verdict 一致率外，还会单独报告“危险放行”：人工标注应为 `revise`，审查器却返回
`pass`。对研究报告来说，这比保守误拦更需要优先排查。
