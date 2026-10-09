# V3：三 Agent 编排计划

> 状态：已确定设计，尚未开始把现有主图重构为子图。

## 目标

把当前单 Agent 图中已经存在、但职责不同的研究、证据判断和报告交付能力，
拆为三个可审计的专业 Agent。拆分的目的不是增加模型数量，而是让“缺什么
证据”能够以结构化反馈回到正确的研究循环。

第一版的 Coordinator 使用程序条件路由，不额外引入一个协调 LLM。这样既能
保持调度可解释，也避免在尚无真实多领域任务时制造无价值的模型决策。

## 角色

### 1. Research Agent

从现有的 `agent -> tools -> record_results -> agent` Tool Calling Loop 演化而来。
它负责选择和调用研究工具，整理正式 evidence，并输出 `ResearchPacket`。
它不再独自决定“证据是否已足够写报告”。

### 2. Evidence Review Agent

从现有 `assess_evidence` 节点演化而来。它读取用户问题与 `ResearchPacket`，
输出覆盖点、缺失点、潜在冲突以及明确的 `feedback_to_research`。

当 verdict 为 `research_more` 时，Coordinator 把这份反馈作为下一次 Research
Agent 的输入。这样“证据不足”不再只是一次泛泛回路，而是定向补查。

### 3. Report Agent

把当前 `generate_draft -> verify_report -> revise_report` 收束为一个独立交付单元。
它只依据已通过 Evidence Review 的正式 evidence 写作、核验、受控修订；若发现
问题只能通过补证据解决，则返回给 Coordinator，而不是凭空补写。

## 目标图

```text
START
  ↓
Coordinator（程序路由）
  ↓
Research Agent
  ↓
Evidence Review Agent
  ├── research_more → Coordinator → Research Agent
  ├── clarify → request_clarification → END
  └── ready_for_report
        ↓
     Report Agent
       ├── pass → store_memory → END
       └── needs_evidence → Coordinator → Research Agent
```

## 交接规则

- Agent 之间传结构化 Packet，不传完整内部 `messages`。
- Research Agent 的工具循环、预算、Trace 保持独立。
- Memory 只能提供背景；只有本轮 `evidence` 可支持报告引用。
- 每一次定向补查必须带明确缺口，且受研究轮次/工具调用预算限制。
- V3 完成后必须与当前单 Agent 图做评测比较，不能只因结构更复杂就认为效果更好。

## 实施顺序

1. 调整交接契约为 `ResearchPacket`、`EvidenceReviewPacket`、`ReportPacket` 与
   `ResearchFeedbackTask`。
2. 将现有 Research Tool Calling Loop 封装为 Research Subgraph。
3. 实现 Evidence Review Subgraph 与结构化反馈注入。
4. 封装 Report Subgraph。
5. 用程序条件边实现最小 Coordinator。
6. 增加单 Agent vs 三 Agent 的回归评测。

## 不在本阶段

- Event/Data Agent、历史再分析资料等事件研究能力；它们是未来可插拔扩展。
- 多模型并行、投票、辩论等高级 Multi-Agent 模式。
- 数据库持久化、异步队列与服务部署。
