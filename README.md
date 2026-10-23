# Autonomous Research Agent

面向大气科学的证据驱动研究 Agent：用 LangGraph 工作流输出带来源的报告，缺证据时受限交付，无法可靠交付时说明失败。

## 它和普通搜索后总结有什么不同

一个面向大气科学领域的学习型 Research Agent。它的目标不是“搜索后直接总结”，而是
把问题、工具调用、证据、评估和带来源的报告串成一条可解释的 LangGraph 工作流。
它不把搜索摘要直接当证据，也不把模型自检当作事实认证；下面的设计取舍明确了取证、核验和交付边界。

## 五个设计取舍

### 证据与候选分离

`search_web` 只产生候选，网页须经 `read_source` 取得正文后才能成为证据，避免把摘要当作已读取的资料。用户陈述不是证据；具体事件须另找事实资料，本地知识库只解释机制、不充当事件实况，这些来源使用边界也通过提示词约束模型。见[事件取证与用户陈述](docs/architecture/research_workflow.md#事件取证与用户陈述)。

### 预算由程序控制

研究轮次、工具调用、修订、澄清和递归保护上限由 `settings.py` 常量决定，程序算好剩余预算再告诉模型，避免依赖模型自报额度。研究预算耗尽后基于已有证据转向交付，证据不足则受限交付或说明失败；递归超限另由入口捕获并说明中止，不保证生成报告。见[程序额度](docs/architecture/research_workflow.md#程序额度)与[配置常量](src/atmospheric_research_agent/agent/settings.py)。

### 分级核验

语义核验只把编造、与证据矛盾、引用不存在或错配、数字／单位／时间错误、证据冲突未说明列为必须修改；措辞偏强等只记建议、不因此拦截，避免反复改写却无法收敛。程序按必须修改项是否为空规范化 `verdict`，引用完整性与简要篇幅仍有程序硬检查，不能把模型判断视为独立事实认证。见[核验分级、复查与交付](docs/architecture/research_workflow.md#核验分级复查与交付)。

### 三种结束方式

正常报告、带程序提示的受限报告和失败说明分开交付，让用户区分“可支持的部分”与“未能可靠完成”。受限报告不写入长期记忆，避免把证据不全的结论固化为后续背景；失败说明也不存记忆。见[交付与记忆边界](docs/architecture/research_workflow.md#核验分级复查与交付)。

### 同任务追问恢复与可观测性

缺少关键信息时暂停追问，回答后在同一任务内保留状态与预算恢复，减少重新开始造成的重复取证。每次运行记录节点级 Trace，debug 模式实时追加节点序列；检查点仅在进程内有效，实时日志不等于可跨进程恢复。见[同任务暂停与恢复](docs/architecture/HITL_minimal.md)与[诊断材料](docs/architecture/research_workflow.md#--debug-诊断材料)。

## 一个真实排错案例

一次追问恢复后，流程因步数超限中止；补充诊断后才发现 Agent 无工具、证据评估要求继续的循环。
根因是有证据的空转未消费研究轮次；修复将此类空转计次，并把预算耗尽后的继续请求规范化为受限报告。
递归保护也按合法路径重新核算；Agent 为什么未调用工具仍未查清，见[定位、验证与遗留](docs/architecture/debug_log.md)。

## 输出示例

公开记录保留真实交付与被拦截的首稿，展示成功交付与失败边界，不代表总体成功率。

### 测试实录

两次真实运行展示了缺少关键条件时暂停追问用户并恢复，以及针对虚构气象站问题，核验环节拦下首稿中无证据支持的细节、修订后交付受限结论的过程。
查看[测试实录](docs/evidence/test_evidence_2026-10-20.md)，其中保留脱敏后的报告与完整 Trace，并说明已知局限。

## 快速开始

### 从 clone 到能跑

需要 Python >=3.11；以下使用 Windows PowerShell，在 clone 后进入项目目录：

```powershell
git clone https://github.com/Kasensill/autonomous-atmospheric-research-agent.git
cd autonomous-atmospheric-research-agent
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest --basetemp .pytest-tmp
.venv\Scripts\python -m atmospheric_research_agent.ingestion.build_index --dry-run
```

`dev` 是 `pyproject.toml` 定义的测试依赖组。安装依赖需要下载软件包；上述离线测试与 dry-run 不调用模型或研究 API。随后按[配置](#配置)填写凭据，建库后再运行研究。

| 能力 | 凭据要求 |
| --- | --- |
| 离线测试、知识资料 `--dry-run` 检查 | 不需要任何 key；需先安装依赖 |
| Agent、证据评估、报告生成／核验／修订 | `DEEPSEEK_API_KEY` |
| 本地向量建库、向量查询与 Rerank | `DASHSCOPE_API_KEY`；Rerank 专用地址按配置需要填写 |
| 网页候选搜索 | `TAVILY_API_KEY` |
| Open-Meteo 天气工具、HTML／PDF 来源读取 | 当前实现不需要 key，但需要联网；完整研究仍需模型凭据 |

离线测试不调用 API，不等于完整研究不需要 API。

## 完整流程图

### 当前状态：研究闭环、报告核验与同任务 HITL 已实现

当前能力统一入口如下；项目仍用于学习，不应被视为科学结论生产系统或天气预警系统。

- 证据驱动研究：工具结果整理为正式证据，报告以证据编号注明来源。
- 同任务追问恢复：缺少关键条件时暂停等待回答，在同一进程中保留状态与预算继续研究。
- 分级核验与修订：模型区分必须修改项与建议，程序检查引用并限制修订次数。
- 受限报告：证据不足时只交付可支持的部分，核验失败则输出失败说明而非未核验草稿。
- 程序化预算：研究轮次、工具调用与澄清额度由程序控制，并为无工具空转计次。
- 可观测性与排错记录：Trace、实时节点日志和超限诊断帮助定位问题，排错结论见[排错记录](docs/architecture/debug_log.md)。

### 图 1：研究与追问

```mermaid
flowchart TD
    __start__(["开始（__start__）"])
    recall_memory["召回背景（recall_memory）"]
    agent["研究决策（agent）<br/>先处理 ask_user"]
    tools["工具执行（tools）"]
    record_results["整理证据与计数<br/>（record_results）"]
    research_guard{"证据守卫<br/>（research_guard）"}
    assess_evidence{"证据评估<br/>（assess_evidence）"}
    request_clarification["追问、暂停与恢复<br/>（request_clarification）"]
    generate_draft["进入图 2：草稿入口<br/>（generate_draft）"]
    finalize_limited_report["进入图 2：失败说明入口<br/>（finalize_limited_report）"]

    __start__ --> recall_memory
    recall_memory --> agent
    agent -->|"有效追问，额度未满"| request_clarification
    agent -->|"无待追问；普通请求整批可执行"| tools
    agent -->|"无普通请求或批次预算不足"| research_guard
    tools --> record_results
    record_results --> agent
    request_clarification -->|"用户回答，预算保留"| agent
    research_guard -->|"有可用证据"| assess_evidence
    research_guard -->|"无证据；澄清已满或预算耗尽"| finalize_limited_report
    research_guard -->|"无证据；两类额度均未满"| agent
    assess_evidence -->|"clarify，额度未满"| request_clarification
    assess_evidence -->|"full_report / limited_report<br/>或 continue 预算耗尽<br/>或 clarify 额度已满"| generate_draft
    assess_evidence -->|"continue，预算尚余"| agent

    classDef interaction fill:#ede9fe,stroke:#7c3aed,color:#111827
    classDef handoff fill:#e0f2fe,stroke:#0284c7,stroke-dasharray:5 3,color:#111827
    classDef failure fill:#fee2e2,stroke:#dc2626,color:#111827
    class request_clarification interaction
    class generate_draft handoff
    class finalize_limited_report failure
```

图 1 的两个出口沿用真实节点 ID：`generate_draft` 对应图 2 的草稿入口，`finalize_limited_report` 对应图 2 的失败说明入口，不是新增节点。`tools` 合并执行四个研究工具：本地检索、当前天气、网页搜索、HTML／文字层 PDF 读取；`ask_user` 不在其中。有效追问优先；超限或参数无效的追问不取消同批普通工具，普通工具仍检查预算。

### 图 2：报告核验与结束

```mermaid
flowchart TD
    generate_draft["生成 full / limited 草稿<br/>（generate_draft）"]
    verify_report{"模型与程序核验<br/>（verify_report）"}
    revise_report["修订必须修改项<br/>（revise_report）"]
    finalize_report["定稿并附来源（finalize_report）<br/>limited 加受限提示"]
    finalize_limited_report["失败说明（finalize_limited_report）<br/>不交付未核验草稿；不存记忆"]
    store_memory["保存或跳过记忆（store_memory）<br/>仅 full、核验通过且非时效性任务写入<br/>limited 不写入"]
    __end__(["结束（__end__）"])

    generate_draft --> verify_report
    verify_report -->|"pass"| finalize_report
    verify_report -->|"revise，次数未满"| revise_report
    verify_report -->|"revise，次数已满"| finalize_limited_report
    revise_report --> verify_report
    finalize_report --> store_memory
    store_memory -->|"full：正常结束"| __end__
    store_memory -->|"limited：受限报告结束"| __end__
    finalize_limited_report -->|"失败说明结束"| __end__

    classDef delivery fill:#e0f2fe,stroke:#0284c7,color:#111827
    classDef failure fill:#fee2e2,stroke:#dc2626,color:#111827
    class finalize_report,store_memory delivery
    class finalize_limited_report failure
    linkStyle 6 stroke:#15803d,stroke-width:3px,color:#15803d
    linkStyle 7 stroke:#b45309,stroke-width:3px,stroke-dasharray:6 3,color:#b45309
    linkStyle 8 stroke:#dc2626,stroke-width:3px,color:#dc2626
```

绿色实线为 full 正常结束，橙色虚线为 limited 受限报告结束，红色实线为失败说明结束。full 与 limited 共用 `finalize_report → store_memory → END`；双色结束线仅展示结果区别，不是新增路由。limited 经过记忆节点但跳过写入，失败说明不经过记忆节点。

完整分支说明、预算与暂停恢复机制见[当前研究流程图](docs/architecture/research_workflow.md)。

## 可用工具

- `search_local_knowledge(query)`：本地知识库的向量召回 + BM25 + RRF + Qwen Rerank。
- `get_current_weather(location)`：通过 Open-Meteo 获取指定地点的模型当前条件。
- `search_web(query)`：通过 Tavily 寻找公开网页候选；候选摘要不是正式证据。
- `read_source(url)`：读取候选网页正文与可获得的标题、机构、作者、日期。

`read_source` 支持有文字层的 PDF（有页数、大小、字符上限），不做 OCR；限制与失败提示见架构文档的 PDF 一节。

只有本地知识片段、天气规范化结果和实际读取到的网页正文会写入 `evidence`。网页搜索
结果仅供 Agent 选择下一步读取哪些 URL。

知识库 Markdown 中的“资料来源”“参考资料”“参考文献”小节会保留在原文件中供人类
追溯，但不会作为可检索 Chunk 进入 Chroma 或 BM25，避免书目列表污染知识候选。

## 项目结构

```text
data/
  knowledge_base/                 # 51 篇 Markdown 知识摘要
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

完整交互入口，首次问题与澄清回答均由用户输入，调用真实模型和工具：

```powershell
.venv\Scripts\python examples/05_interactive_research.py
```

正式模块入口也支持追问后继续：缺少地点等必要信息时会在当前终端等待回答，不需要重新发起任务。
HITL 使用 `InMemorySaver`、不变的 `thread_id`、`interrupt()` 和 `Command(resume=...)`。
恢复保留证据、消息、记忆、研究轮次与工具计数；回答追加到消息和澄清记录，传给评估、写作与核验。
人工等待时间不计入执行耗时。退出进程后不能恢复；尚未加入 SQLite、前置意图节点或旧证据自动筛选。
设计与验证详情见 [最小 HITL 说明](docs/architecture/HITL_minimal.md)。

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
.venv\Scripts\python -m pytest --basetemp .pytest-tmp
```

测试不发送真实模型请求、不访问 Tavily，也不读取公开网页；外部依赖都由假对象替代。

当前基线：2026-10-23，基于提交 `f005e86`，Python 3.11.9，离线测试 **341 passed**；复现命令：`.venv\Scripts\python -m pytest --basetemp .pytest-tmp`。

## 当前边界与版本路线图

- 本地检索已升级为 **向量召回 + 中文 BM25 + RRF + qwen3.7-text-rerank**；
  检索排名只表示问题与片段的相关性，不等于来源可信度。当前语料为 204 个可检索
  知识 Chunk（由 `--dry-run` 得到，日期 2026-10-23；不是已建库实测）。缺少专题资料时应生成受限结论，而不是外推为完整结论。
- 当前已有检索与报告核验评测，以及按 ``run_id`` 关联的节点 Trace、耗时和业务事件；它们仍是
  本地命令行可读版本，尚未接入数据库、监控平台、告警或成本仪表盘。
- 网页正文读取具有 V1 基础的 URL 安全过滤；不等于生产级的全量 SSRF 防护。
- 当前 Task Memory 只保存已通过核验报告的项目摘要，并按轻量词项召回；它是跨运行背景，绝不是
  本次可引用的正式 evidence，也尚未实现数据库或跨用户隔离。任务恢复另由内存检查点负责，不能跨进程。

### 历史修复与质量验证（2026-10-20）

**输出质量第一批修复（2026-10-20）**

- 天气 evidence 补齐已请求的云量、风向；来源继续标为模型当前条件。
- 最终来源清单仅列正文引用的证据，保持 E 编号；保留原始检索材料用于追溯。
- 工具 Trace 展示工具名、受限参数和结果计数；条件路由记录预算阻止原因，便于解释“提出调用但未执行”。
- 简要请求正文上限 900 字符（不含来源），超长进入既有修订路径；核验加强前后自洽与产品性质检查。
- 当前天气及受限报告跳过长期记忆写入；旧天气快照不自动召回，原文件保留。通用词过滤与最低重合规则降低随意召回。
- 完全重复去重补齐同批重复；这不是语义去重。核验正负例扩充至八条。
- 离线回归 102 项通过；真实核验评测 8/8 完成、8/8 与标签一致、危险放行 0、误拦 0。
  仅为小型自建集的本次结果；首次沙箱连接失败的报告也保留，未算入质量得分。

以上为历史阶段验证，非当前通过数。

设计、逐条处理和验证边界见 [外部输出评价改进记录](docs/architecture/output_review_2026-10-20.md)。

**输出质量第二批：研究后实施（2026-10-20）**

| 条目 | 实际处理 | 验证与限制 |
| --- | --- | --- |
| 检索相关性门槛 | Rerank 成功时过滤低于 0.07 的候选，可返回零条 | 八题按 query 分校准/留出；留出负例误收 8→3，仍非零，阈值暂定 |
| 对原有召回的影响 | 同一候选做过滤前后对照 | 十二题全部完成；Hit 1.000、MRR 0.9583、Coverage 0.8889 前后相同 |
| 降级检索 | 向量/BM25/RRF 保持可用，明确门槛未应用 | 不拿不同尺度分数套阈值，降级结果仍有噪声风险 |
| 检索噪声指标 | 增加显式负来源、已标噪声占比和标注覆盖 | 未标注不当无关；标注规模仍小 |
| 过期当前天气 | 保存带时区有效时间，模型/报告读取适用证据视图 | 原始记录不删、引用不重编号；两小时业务窗口可配置，通用资料不受影响 |
| 恢复后补查 | 模型输入已有适用证据、上轮缺口及预算 | 尚无语义查询去重或强制工具缓存，不宣称彻底消除过度检索 |
| 受限原因 | 程序预算原因与模型推测的能力缺口/冲突/资料不足分开记录 | 可同时出现；模型判断不是程序确认的事实 |
| 写作与修订一致性 | 与核验共享模型产品性质、比较基准及简要要求 | 首次端到端因“实况”误称被拦截；修复后第二次通过，失败记录保留 |
| 天气快照记忆 | 合格交付也不写入长期知识记忆 | 实测出现 memory_skipped；检查点与运行记录仍保留 |
| 气压/气候态、语义约束变更过滤 | 未增加 | 单点气压不足以完成区域归因；多年基准与地点/年份更正需独立设计 |
| jieba 预热 | 保持已有进程内缓存 | 没有 Serverless 部署，不提前引入部署改造 |

最终离线回归 **110 passed**。真实端到端第二次验收：正文609字符、2轮研究、3次工具调用、8条证据、
一次追问恢复、零次修订、未存天气记忆，九项检查通过。首次验收未通过，不计为成功；
两次为不同随机运行，并非严格性能 A/B，不能把用时/条数变化归为单个改动的因果收益。
模型审查 pass 仍不能代替科学内容人工审核。

以上为历史阶段验证，非当前通过数。

### 原始结果与复现

可核对的原始结果：原始运行产物仅保存在本地，不随仓库发布。

- 核验器八题结果
- 相关性门槛校准（含全部原始分数）
- 原有十二题门槛前后对照
- 首次完整运行：核验失败，安全收尾
- 修复后完整运行：609字符正文与Trace

复现研究性验收（将调用配置的 API 并产生费用；无需读取或展示密钥）：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.evaluation.relevance_gate_runner
.venv\Scripts\python -m atmospheric_research_agent.evaluation.retrieval_gate_effect
.venv\Scripts\python -m atmospheric_research_agent.evaluation.live_output_review
```

### 核心实现顺序

当前确定的核心 Agent 学习与实现顺序如下。每一阶段都建立在前一阶段已经可运行、可观察的基础上，
避免为了堆叠概念而过早引入复杂架构：

1. **V1.6：检索评测与数据补强（已完成）**：固定问题集对比纯向量、RRF、RRF + Rerank 的召回质量，
   并根据评测补充了低空急流等专题资料。
2. **V2：Reflection / Verification（已完成）**：报告先生成草稿，再由结构化审查与程序引用完整性检查共同
   决定定稿、受控修订或安全降级；另有最小人工标注评测集。
3. **V2.1：Task Memory 与 Observability（第一版已完成）**：每次运行有 run_id、Trace、节点耗时与总耗时；
   完整且非时效性、已通过核验的报告摘要会保存为本地项目记忆，并在后续相关任务开始时作为背景召回。
   **HITL 增量（2026-10-20）**：两个澄清入口接入同任务暂停恢复；证据评估的澄清必须提供具体追问。
   完整离线回归 94 项通过；2026-10-19 的一次真实模型验收完成地点追问、北京回答、天气取证及报告交付。
   这是机制验收，不证明模型永不重复提问、数据永不过期或报告科学结论绝对正确。
   以上为历史阶段验证，非当前通过数。
4. **V3：Multi-Agent（规划中，未实现）**：此前按 Coordinator / Planning、Research、Evidence Review、
   Report Writer、Report Verification 五类职责描述；具体拆分以[三 Agent 规划](docs/architecture/V3_three_agent_plan.md)为准，
   不表示当前已实现子 Agent。Event/Data Agent 属于后续事件研究方向的可插拔扩展，不替代这条近期主线。

MCP、FastAPI、Docker 与前端属于服务和产品交付层：它们会复用稳定的 `run_research()` 核心入口，
但不抢在上述核心能力之前实现。

### 远期能力地图

**核心能力之后的高级能力地图**

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
