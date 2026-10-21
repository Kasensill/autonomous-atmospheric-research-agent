# 当前研究流程图

本图以 `graph.py` 实际注册的节点、固定边和条件边为准。先通过 `build_research_graph().get_graph().draw_mermaid()` 离线生成骨架，再补充中文标签和路由条件；构图没有执行 `invoke`，不调用模型或外部 API。

## 完整流程

```mermaid
flowchart TD
    __start__(["开始（__start__）"])
    recall_memory["召回背景记忆（recall_memory）"]
    agent["研究决策与追问协议处理（agent）<br/>ask_user 仅表达追问，不进入工具执行"]
    tools["工具执行（tools）<br/>本地检索 / 当前天气 / 网页搜索 / 网页读取"]
    record_results["整理证据与更新研究计数（record_results）"]
    research_guard{"可用证据与额度守卫（research_guard）"}
    assess_evidence{"证据评估与下一步判断（assess_evidence）"}
    request_clarification["人工追问与暂停恢复（request_clarification）<br/>interrupt 等待；Command 恢复"]
    generate_draft["生成完整或受限草稿（generate_draft）<br/>report_mode = full / limited"]
    verify_report{"模型与程序核验（verify_report）"}
    revise_report["按审查意见修订草稿（revise_report）"]
    finalize_report["定稿并附引用来源（finalize_report）<br/>limited 额外附加受限提示"]
    store_memory["按规则保存或跳过记忆（store_memory）<br/>时效性天气任务不存长期记忆"]
    finalize_limited_report["生成失败说明（finalize_limited_report）<br/>不是已核验的受限报告；不存记忆"]
    __end__(["结束（__end__）"])

    __start__ --> recall_memory
    recall_memory --> agent
    agent -->|"request_clarification：有效 ask_user 且澄清未满<br/>同批普通工具取消，不计数"| request_clarification
    agent -->|"tools：没有有效待追问；普通请求整批预算允许<br/>ask_user 超限或无效不取消普通工具"| tools
    agent -->|"research_guard：没有待追问且无普通请求<br/>或普通请求整批预算不足"| research_guard
    tools --> record_results
    record_results --> agent
    request_clarification -->|"回答写回原任务；预算保留"| agent
    research_guard -->|"assess_evidence：有可用证据，优先于额度检查"| assess_evidence
    research_guard -->|"finalize_limited_report：无可用证据<br/>先判澄清已满，再判研究预算耗尽"| finalize_limited_report
    research_guard -->|"agent：无可用证据；澄清未满且预算尚余"| agent
    assess_evidence -->|"request_clarification：clarify 且澄清未满"| request_clarification
    assess_evidence -->|"generate_draft：full_report / limited_report<br/>或 continue 且预算耗尽；超限 clarify 改 limited_report"| generate_draft
    assess_evidence -->|"agent：continue 且研究预算尚余"| agent
    generate_draft --> verify_report
    verify_report -->|"finalize_report：pass"| finalize_report
    verify_report -->|"revise_report：revise 且修订次数未满"| revise_report
    verify_report -->|"finalize_limited_report：revise 且修订次数已满"| finalize_limited_report
    revise_report --> verify_report
    finalize_report --> store_memory
    store_memory -->|"正常结束：report_mode 为 full"| __end__
    store_memory -->|"受限报告结束：report_mode 为 limited"| __end__
    finalize_limited_report -->|"失败说明结束"| __end__

    classDef interaction fill:#ede9fe,stroke:#7c3aed,color:#111827
    classDef sharedDelivery fill:#e0f2fe,stroke:#0284c7,color:#111827
    classDef failure fill:#fee2e2,stroke:#dc2626,color:#111827
    class request_clarification interaction
    class finalize_report,store_memory sharedDelivery
    class finalize_limited_report failure
    linkStyle 20 stroke:#15803d,stroke-width:3px,color:#15803d
    linkStyle 21 stroke:#b45309,stroke-width:3px,stroke-dasharray:6 3,color:#b45309
    linkStyle 22 stroke:#dc2626,stroke-width:3px,color:#dc2626
```

绿色实线表示完整报告正常结束，橙色虚线表示受限报告结束，红色实线表示失败说明结束。完整、受限报告共用 `finalize_report → store_memory → END`；图中把最后一条固定边按结果画成两条彩色展示线，**不是新增条件边或路由函数**。实际只有一个 `__end__`。`__start__`、`__end__` 是 LangGraph 骨架中的内置节点 ID，其余 13 个 ID 均为主图注册名称。[代码：graph.py:265](../../src/atmospheric_research_agent/agent/graph.py#L265)

## 分支含义与先后顺序

| 分流位置 | 当前实现 |
| --- | --- |
| `agent` 的追问处理 | `ask_user` 声明绑定给模型但不在四个研究工具的执行列表内。有效且额度未满时，在 `agent` 为本批每个请求补一条 `ToolMessage`；追问回应为“等待用户，非回答；已转交人工澄清流程”，普通工具回应为取消，均不产生证据或计入研究预算。多个有效追问合并为一次。见 [nodes.py:154](../../src/atmospheric_research_agent/agent/nodes.py#L154)、[nodes.py:302](../../src/atmospheric_research_agent/agent/nodes.py#L302)。 |
| `route_after_agent` | 先检查待追问且澄清未满，再检查普通工具和批次预算。超限时只拒绝 `ask_user`；参数无效时回错误消息、不暂停；这两种情况同批普通工具照常走预算，预算不足或没有普通请求才去守卫。`tools` 按调用 ID 跳过已回应请求，误入执行层的 `ask_user` 也只安全报错。见 [nodes.py:369](../../src/atmospheric_research_agent/agent/nodes.py#L369)、[nodes.py:432](../../src/atmospheric_research_agent/agent/nodes.py#L432)、[nodes.py:469](../../src/atmospheric_research_agent/agent/nodes.py#L469)。 |
| `route_after_research_guard` | 先判断有无 `usable_evidence`，有则评估；无则先检查澄清上限，再检查研究预算，任一满足进入失败说明，否则回 `agent`。守卫在“澄清已满且无可用证据”时不增加研究轮次；其他无证据尝试会加一轮，再由路由判断预算。见 [nodes.py:654](../../src/atmospheric_research_agent/agent/nodes.py#L654)、[nodes.py:689](../../src/atmospheric_research_agent/agent/nodes.py#L689)。 |
| `assess_evidence` 的澄清处理 | 与 `agent` 共用已完成问答历史。模型若返回 `clarify` 但额度已满，评估节点程序覆盖为 `sufficient=False, next_step=limited_report`，清除待追问并记录上限事件；不是无证据失败说明。见 [nodes.py:787](../../src/atmospheric_research_agent/agent/nodes.py#L787)。 |
| `route_after_evidence_assessment` | `clarify` 未满去追问、已满去草稿（路由额外防守）；`full_report / limited_report` 去草稿；其余 `continue` 在预算耗尽时去草稿，否则回 `agent`。评估入口要求有可用证据。见 [nodes.py:780](../../src/atmospheric_research_agent/agent/nodes.py#L780)、[nodes.py:815](../../src/atmospheric_research_agent/agent/nodes.py#L815)。 |
| `generate_draft` 的报告模式 | 只有 `sufficient=True` 且 `next_step=full_report` 才是 `full`，其他为 `limited`。二者都必须核验；受限报告在定稿时由程序加“当前证据不全”提示。见 [nodes.py:836](../../src/atmospheric_research_agent/agent/nodes.py#L836)、[nodes.py:1103](../../src/atmospheric_research_agent/agent/nodes.py#L1103)、[nodes.py:1198](../../src/atmospheric_research_agent/agent/nodes.py#L1198)。 |
| `route_after_report_verification` | `pass` 定稿；否则修订次数未满就修订并重新核验，已满就失败说明。模型核验之外，程序检查引用完整性和简要报告长度，发现问题可强制改成 `revise`。见 [nodes.py:1046](../../src/atmospheric_research_agent/agent/nodes.py#L1046)、[nodes.py:1088](../../src/atmospheric_research_agent/agent/nodes.py#L1088)。 |
| 两类交付 | `finalize_report` 交付核验通过的完整或受限报告，随后处理记忆；当前天气任务跳过存储。`finalize_limited_report` 按核验失败、无证据且澄清已满、无证据预算耗尽或异常状态生成失败说明，直接结束，不经过记忆。澄清上限原因不写成预算耗尽。见 [nodes.py:1225](../../src/atmospheric_research_agent/agent/nodes.py#L1225)、[nodes.py:1272](../../src/atmospheric_research_agent/agent/nodes.py#L1272)。 |

图上条件边标签的冒号前是路由函数真实返回值，冒号后是判断依据；`pass / revise`、`clarify / continue / full_report / limited_report` 是相应结构化判断字段，不是独立业务节点。未捕获异常和图递归上限异常不属于注册的结束分支，本图不将它们画成正常交付。

## interrupt 与同任务恢复

`interrupt` 在 `request_clarification` 内，不在 `tools`。追问工具的协议回应已在 `agent` 关闭；暂停节点在 `interrupt` 前只读取并校验追问，不调用模型或工具、不写 State。用户回答后追加消息、问答历史，清空待追问和旧评估，沿固定边回 `agent`，不重走入口，不重置预算。[代码：nodes.py:1322](../../src/atmospheric_research_agent/agent/nodes.py#L1322)、[graph.py:268](../../src/atmospheric_research_agent/agent/graph.py#L268)

正式入口使用 `InMemorySaver`、同一图实例和不变的 `thread_id`，通过 `Command(resume=...)` 恢复。等待不标记任务完成，运行时长排除用户输入等待。检查点仅在进程内有效，进程退出后暂停任务不能恢复；这与跨运行的项目摘要记忆不是一回事。详见 [HITL 说明](HITL_minimal.md)。[代码：run.py:78](../../src/atmospheric_research_agent/agent/run.py#L78)

## 程序额度

| 常量 | 当前值 | 用途 |
| --- | --- | --- |
| `MAX_RESEARCH_ROUNDS` | 3 | 一批实际工具结果记一轮；无证据守卫也可记空尝试，不是每进入一次 `agent` 都加一。 |
| `MAX_TOOL_CALLS` | 8 | 实际研究工具请求计数上限；整批放不下就不执行该批。 |
| `MAX_CLARIFICATIONS` | 2 | 两入口共用 `len(clarification_history)`，即已完成问答次数。 |
| `MAX_REPORT_REVISION_ROUNDS` | 1 | 草稿最多修订一次，然后再次核验。 |
| `GRAPH_RECURSION_LIMIT` | 30 | 入口传入的图执行步数保护，不是研究轮数或追问次数。 |

定义来源：[settings.py:12](../../src/atmospheric_research_agent/agent/settings.py#L12)。预算函数来源：[settings.py:20](../../src/atmospheric_research_agent/agent/settings.py#L20)。研究计数来源：[nodes.py:645](../../src/atmospheric_research_agent/agent/nodes.py#L645)、[nodes.py:654](../../src/atmospheric_research_agent/agent/nodes.py#L654)。递归上限传入位置：[run.py:88](../../src/atmospheric_research_agent/agent/run.py#L88)。

研究预算在轮次或工具次数任一达到上限时耗尽；执行普通工具批次还要求轮次未满，且“已执行工具数 + 本批普通请求数”不超过 8。追问、控制回应和因澄清取消的工具不计数。`usable_evidence` 会排除过期或缺少可靠时间的天气记录；原始证据仍保留。[代码：settings.py:30](../../src/atmospheric_research_agent/agent/settings.py#L30)、[nodes.py:483](../../src/atmospheric_research_agent/agent/nodes.py#L483)、[evidence.py:25](../../src/atmospheric_research_agent/agent/evidence.py#L25)

## 生成版本与复现

依据代码提交：`c690f56bc8515806fb22980007829704387f80b3`（生成日期：2026-10-21）。本次仅更新文档，不修改业务实现。

在项目根目录执行以下命令可重新生成原始 Mermaid 骨架，只打印文本，不执行研究、不联网：

```powershell
.venv\Scripts\python -c "from atmospheric_research_agent.agent.graph import build_research_graph; print(build_research_graph().get_graph().draw_mermaid())"
```

原始骨架不会包含本页人工补充的中文、条件说明和三类交付配色。更新图时需重新核对四个路由函数及节点内澄清预处理；README 与本页使用同一张整理图。本机未找到 `mmdc`，未做渲染验证；已人工检查引号、节点 ID、边标签及样式索引，并离线核对节点和边与生成骨架一致（末尾固定结束边的双色展示去重后核对）。
