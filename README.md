# Autonomous Research Agent

一个面向大气科学领域的学习型 Research Agent。它的目标不是“搜索后直接总结”，而是
把问题、工具调用、证据、评估和带来源的报告串成一条可解释的 LangGraph 工作流。

## 当前状态：V1 核心闭环已完成

V1 已具备真实的本地知识库、外部工具调用与证据驱动报告能力。它仍是学习项目，
不应被视为科学结论生产系统或天气预警系统。

```text
用户问题
  ↓
agent ── 缺少用户关键信息 ──→ request_clarification ──→ END（等待用户）
  │
  ├── 请求工具 ──→ tools ──→ record_results ──→ agent
  │                    │
  │                    └── 只把可引用结果写入 evidence
  │
  └── 暂不调用工具 ──→ research_guard
                              ├── 无证据且仍有预算 ──→ agent
                              ├── 有证据 ──→ assess_evidence
                              │                  ├── 继续研究 ──→ agent
                              │                  ├── 需要澄清 ──→ request_clarification
                              │                  └── 可报告 ──→ generate_report ──→ END
                              └── 预算耗尽 ──→ generate_report（受限结论）──→ END
```

## 可用工具

- `search_local_knowledge(query)`：本地知识库的向量召回 + BM25 + RRF + Qwen Rerank。
- `get_current_weather(location)`：通过 Open-Meteo 获取指定地点的模型当前条件。
- `search_web(query)`：通过 Tavily 寻找公开网页候选；候选摘要不是正式证据。
- `read_source(url)`：读取候选网页正文与可获得的标题、机构、作者、日期。

只有本地知识片段、天气规范化结果和实际读取到的网页正文会写入 `evidence`。网页搜索
结果仅供 Agent 选择下一步读取哪些 URL。

知识库 Markdown 中的“资料来源”“参考资料”“参考文献”小节会保留在原文件中供人类
追溯，但不会作为可检索 Chunk 进入 Chroma 或 BM25，避免书目列表污染知识候选。

## 项目结构

```text
data/
  knowledge_base/                 # 50 篇大气科学 Markdown 资料
  vector_store/                   # Chroma 运行数据（不提交）
src/atmospheric_research_agent/
  ingestion/build_index.py        # Markdown → 分块 → Embedding → Chroma
  agent/
    state.py                       # ResearchState：图中的共享公共契约
    settings.py                    # 预算、Top-K、超时等集中运行配置
    nodes.py                       # Agent、工具、证据、评估、报告节点
    graph.py                       # 节点和边组成的 LangGraph
    run.py                         # 命令行入口
    tools/                         # 四个实际工具
tests/                             # 单元测试与图流程测试
examples/                          # Tool Calling 学习示例
```

## 配置

复制 `.env.example` 为 `.env`，填写所需密钥；`.env` 不应提交。

```text
DEEPSEEK_API_KEY=             # 运行 Agent、证据评估、报告生成
DASHSCOPE_API_KEY=            # 建库和本地向量查询
# 可选：qwen3.7-text-rerank 需要业务空间专用地址时再填写
DASHSCOPE_RERANK_BASE_URL=
TAVILY_API_KEY=               # 公开网页搜索；不使用网页搜索时可不填
```

`agent/settings.py` 集中维护不会交给模型修改的在线运行规则，例如研究轮次、工具调用
上限、本地检索 Top-K、网页读取长度和网络超时。离线入库的分块参数留在
`ingestion/build_index.py`，因为它是独立的数据准备流程。

DeepSeek 的 Agent Tool Calling 明确关闭思考模式：该 API 模式要求每轮回传专有的
`reasoning_content`，而项目以可审阅的 State、工具结果与节点轨迹记录研究过程，不依赖
隐藏思维链。

## 从知识库到向量库

先运行不消耗 API 的资料检查：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.ingestion.build_index --dry-run
```

确认后重建本地 Chroma collection：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.ingestion.build_index --reset
```

## 运行研究

只使用内部知识库：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.agent.run "为什么副热带高压会带来高温少雨？请仅查询内部知识库，并给出带来源的研究结论。" --debug
```

混合使用内部知识库与公开来源：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.agent.run "为什么副热带高压常带来高温少雨？请同时使用内部知识库和至少一个来自大学、政府气象机构或权威百科的公开网页来源；不要使用社交媒体或商业天气 API 首页，并给出带来源的研究结论。" --debug
```

`--debug` 会显示研究轮次、工具调用数、正式证据数与节点轨迹，适合学习图实际如何运行。

## 测试

```powershell
.venv\Scripts\python -m pytest
```

测试不发送真实模型请求、不访问 Tavily，也不读取公开网页；外部依赖都由假对象替代。

## 当前边界与版本路线图

- 本地检索已升级为 **向量召回 + 中文 BM25 + RRF + qwen3.7-text-rerank**；
  检索排名只表示问题与片段的相关性，不等于来源可信度。当前仅有 200 个可检索
  知识 Chunk，缺少专题资料时应生成受限结论，而不是外推为完整结论。
- 当前已有检索与报告核验评测，以及按 ``run_id`` 关联的节点 Trace、耗时和业务事件；它们仍是
  本地命令行可读版本，尚未接入数据库、监控平台、告警或成本仪表盘。
- 网页正文读取具有 V1 基础的 URL 安全过滤；不等于生产级的全量 SSRF 防护。
- 当前 Task Memory 只保存已通过核验报告的项目摘要，并按轻量词项召回；它是跨运行背景，绝不是
  本次可引用的正式 evidence，也尚未实现数据库、跨用户隔离或任务中断恢复。

当前确定的核心 Agent 学习与实现顺序如下。每一阶段都建立在前一阶段已经可运行、可观察的基础上，
避免为了堆叠概念而过早引入复杂架构：

1. **V1.6：检索评测与数据补强（已完成）**：固定问题集对比纯向量、RRF、RRF + Rerank 的召回质量，
   并根据评测补充了低空急流等专题资料。
2. **V2：Reflection / Verification（已完成）**：报告先生成草稿，再由结构化审查与程序引用完整性检查共同
   决定定稿、受控修订或安全降级；另有最小人工标注评测集。
3. **V2.1：Task Memory 与 Observability（第一版已完成）**：每次运行有 run_id、Trace、节点耗时与总耗时；
   已通过核验的报告摘要会保存为本地项目记忆，并在后续相关任务开始时作为背景召回。
4. **V3：Multi-Agent（设计中）**：先按职责把 Coordinator / Planning、Research、Evidence Review、
   Report Writer、Report Verification 拆为可审计的子 Agent，并通过结构化交接契约、定向补查与评测
   证明拆分的价值；Event/Data Agent 属于后续事件研究方向的可插拔扩展，不替代这条近期主线。

MCP、FastAPI、Docker 与前端属于服务和产品交付层：它们会复用稳定的 `run_research()` 核心入口，
但不抢在上述核心能力之前实现。

### 核心能力之后的高级能力地图

当单 Agent、Reflection、任务记忆和职责明确的 Multi-Agent 已稳定后，后续能力按以下方向扩展。
它们不是另一套脱离基础的知识，而是把同样的 State、节点、边、工具与评测能力扩展到更可靠的生产系统。

- **评测与可观测性**：固定问题集、期望证据、检索与报告质量指标、耗时和成本记录；不能只凭单次回答观感判断质量。
- **持久化与可恢复运行**：将 State、检查点、研究记录写入数据库，使中断任务可以从最近节点恢复。
- **Human-in-the-loop**：在关键来源、研究计划或最终结论处暂停，等待人工确认后继续。
- **可靠工具系统**：权限分级、参数校验、超时、重试、幂等性与沙箱；工具开始涉及文件、代码或外部写操作时尤其关键。
- **长期与异步任务**：以后台任务、队列和进度状态支持分钟到小时级研究，并允许查看、取消和恢复。
- **分层 Memory**：从任务记忆扩展到用户偏好和项目知识；每层都要明确记忆写入、检索、过期与影响决策的规则。
- **更严格的验证器**：检查引用是否真实支撑主张、来源是否冲突、是否有过度推断，而不只检查是否存在引用。
- **高级 Multi-Agent 协作**：并行研究、辩论、投票、协调与任务分配；拆分必须用评测证明其质量或效率收益。
- **Graph RAG 与结构化知识**：资料中的实体与关系需要多跳推理时，再引入图谱，而不是替换已经有效的文本检索。
- **多模态 Agent**：读取天气图、卫星云图、PDF 表格、雷达图等非纯文本证据。
- **部署与产品化**：FastAPI、MCP、Docker、前端、认证、限流、日志、监控与成本控制。

SFT、LoRA、DPO、RLHF 等模型训练技术属于并行但更偏模型层的路线。只有在提示、工具、检索与工作流
优化仍无法稳定解决某类问题时，才作为后续投入方向；当前不影响本项目的 Agent 工程主线。
