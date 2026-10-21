# 最小人工澄清：同任务暂停与恢复

本实现使用进程内检查点暂停、恢复同一次研究；没有独立的前置意图节点、SQLite 检查点或前端。主图从 `recall_memory` 进入 `agent`，入口使用 `InMemorySaver`。已有当前天气过期过滤，但没有按用户纠正的地点、年份自动重评估旧证据的机制。[代码：graph.py:211](../../src/atmospheric_research_agent/agent/graph.py#L211)、[run.py:78](../../src/atmospheric_research_agent/agent/run.py#L78)、[evidence.py:25](../../src/atmospheric_research_agent/agent/evidence.py#L25)

## 流程

`agent / assess_evidence → request_clarification → interrupt → 用户回答 → agent`

### 两个澄清入口与判断顺序

- `agent` 通过 `ask_user(question=...)` 表达追问，而不是旧的 JSON `decision=clarify`。`ask_user` 只是一份工具声明，随研究工具一起绑定给模型，不在 `RESEARCH_TOOLS` 执行列表中。旧 JSON 不符合当前无工具决策协议时，会安全降级到 `research_guard`，不会发起追问。[代码：clarification.py:5](../../src/atmospheric_research_agent/agent/tools/clarification.py#L5)、[tools/__init__.py:10](../../src/atmospheric_research_agent/agent/tools/__init__.py#L10)、[nodes.py:154](../../src/atmospheric_research_agent/agent/nodes.py#L154)、[nodes.py:352](../../src/atmospheric_research_agent/agent/nodes.py#L352)
- `route_after_agent` 先检查待追问问题且额度未满，再检查普通工具请求及其批次预算；没有可执行的普通工具请求时进入 `research_guard`。因此有效追问优先于研究预算判断，即使研究预算已满，仍可使用剩余澄清额度。[代码：nodes.py:369](../../src/atmospheric_research_agent/agent/nodes.py#L369)
- `assess_evidence` 保留结构化的 `next_step=clarify` 入口，不调用 `ask_user`；选择它时必须给出非空 `clarification_question`。额度未满时转入同一个 `request_clarification` 节点。[代码：nodes.py:112](../../src/atmospheric_research_agent/agent/nodes.py#L112)、[nodes.py:795](../../src/atmospheric_research_agent/agent/nodes.py#L795)、[nodes.py:824](../../src/atmospheric_research_agent/agent/nodes.py#L824)

### 同批工具请求与协议回应

| 情况 | 当前处理 | 代码位置 |
| --- | --- | --- |
| 额度未满，`ask_user` 参数有效 | 多个追问去重后合并成一次追问。同批普通工具全部取消、不执行、不计数；在 `agent` 中为每个请求的 `tool_call_id` 补一条 `ToolMessage`，然后转向暂停。 | [nodes.py:302](../../src/atmospheric_research_agent/agent/nodes.py#L302)、[nodes.py:315](../../src/atmospheric_research_agent/agent/nodes.py#L315) |
| 已达到澄清上限 | 只拒绝 `ask_user`，回应“已达上限，不可再追问，请基于现有信息继续”；同批普通工具不取消，照常检查预算。预算容纳不了时补拒绝回应并转守卫，容纳得下时进入 `tools`。 | [nodes.py:309](../../src/atmospheric_research_agent/agent/nodes.py#L309)、[nodes.py:327](../../src/atmospheric_research_agent/agent/nodes.py#L327)、[nodes.py:384](../../src/atmospheric_research_agent/agent/nodes.py#L384) |
| 额度未满，但 `question` 为空、纯空白或不是字符串 | 回错误 `ToolMessage`，不设置追问、不暂停；同批普通工具不取消，仍按正常预算检查。若只有无效 `ask_user`，进入 `research_guard`。同批多个 `ask_user` 中只要有一个参数无效，本批就不发起追问。 | [nodes.py:306](../../src/atmospheric_research_agent/agent/nodes.py#L306)、[nodes.py:318](../../src/atmospheric_research_agent/agent/nodes.py#L318)、[nodes.py:327](../../src/atmospheric_research_agent/agent/nodes.py#L327)、[nodes.py:393](../../src/atmospheric_research_agent/agent/nodes.py#L393) |

有效追问的控制回应原文是“等待用户，非回答；已转交人工澄清流程”，即**这不是用户的回答**；取消的普通工具回应是“本轮因澄清优先取消，未执行”。这些回应带 `protocol_only=True`，用于关闭模型工具调用协议，不是证据，也不计研究轮次或工具次数；它们不会沿有效追问路径进入 `record_results`。[代码：nodes.py:315](../../src/atmospheric_research_agent/agent/nodes.py#L315)、[nodes.py:396](../../src/atmospheric_research_agent/agent/nodes.py#L396)、[nodes.py:483](../../src/atmospheric_research_agent/agent/nodes.py#L483)、[graph.py:215](../../src/atmospheric_research_agent/agent/graph.py#L215)

超限或参数无效时，`agent` 先关闭 `ask_user` 的协议；保留下来的普通工具由 `tools` 回应。`tools` 按 `tool_call_id` 跳过已有回应的请求，避免重复补消息。正常路径不在 `tools` 执行 `ask_user`；即使它误入执行层，也只返回错误并记录 `control_tool_not_executable`，不会调用研究工具实现。`record_results` 排除控制回应，只统计实际研究工具结果。[代码：nodes.py:432](../../src/atmospheric_research_agent/agent/nodes.py#L432)、[nodes.py:469](../../src/atmospheric_research_agent/agent/nodes.py#L469)、[nodes.py:483](../../src/atmospheric_research_agent/agent/nodes.py#L483)、[nodes.py:592](../../src/atmospheric_research_agent/agent/nodes.py#L592)

### 共用两次额度与超限出口

- `MAX_CLARIFICATIONS=2`，`agent` 和 `assess_evidence` 共用 `len(clarification_history)` 判断，没有各自独立计数器。记录在用户回答后追加；多个有效 `ask_user` 合并成一次问答，只增加一条历史。追问、拒绝和取消请求本身不消耗研究轮次及工具次数，也不重置已有预算。[代码：settings.py:17](../../src/atmospheric_research_agent/agent/settings.py#L17)、[settings.py:20](../../src/atmospheric_research_agent/agent/settings.py#L20)、[nodes.py:308](../../src/atmospheric_research_agent/agent/nodes.py#L308)、[nodes.py:1334](../../src/atmospheric_research_agent/agent/nodes.py#L1334)
- 达到上限后，`agent` 的规则要求不再追问；评估提示单独写“当前不可再追问用户。”，未满时写“当前可以追问用户（已追问 N 次，还可追问 M 次）。”。程序另在评估 trace 写入 `clarification_quota`（`used`、`limit`、`remaining`）；模型的自由解释 `reason` 不决定额度。[代码：nodes.py:189](../../src/atmospheric_research_agent/agent/nodes.py#L189)、[nodes.py:722](../../src/atmospheric_research_agent/agent/nodes.py#L722)、[nodes.py:768](../../src/atmospheric_research_agent/agent/nodes.py#L768)、[nodes.py:803](../../src/atmospheric_research_agent/agent/nodes.py#L803)
- 超限仍请求追问会记录 `clarification_limit_reached`。到达守卫时若没有**可用证据**（包括原证据全部为过期天气），直接进入 `finalize_limited_report` 失败说明，不增加研究轮次；原因写“澄清次数达到上限”，不是预算耗尽。有可用证据则进入评估；超限并不阻止同批普通工具先按预算执行。[代码：nodes.py:196](../../src/atmospheric_research_agent/agent/nodes.py#L196)、[nodes.py:654](../../src/atmospheric_research_agent/agent/nodes.py#L654)、[nodes.py:689](../../src/atmospheric_research_agent/agent/nodes.py#L689)、[nodes.py:1286](../../src/atmospheric_research_agent/agent/nodes.py#L1286)
- 若评估模型在额度已满时仍返回 `clarify`，程序将评估改为 `sufficient=False, next_step=limited_report`，清除待追问，转入 `generate_draft` 生成受限草稿；这不是直接输出失败说明，后续仍经核验。路由本身也防守超限 `clarify`，不会再暂停。[代码：nodes.py:787](../../src/atmospheric_research_agent/agent/nodes.py#L787)、[nodes.py:824](../../src/atmospheric_research_agent/agent/nodes.py#L824)、[graph.py:252](../../src/atmospheric_research_agent/agent/graph.py#L252)

### 暂停、恢复与重入安全

- 运行入口配置 `InMemorySaver`，以初始 `run_id` 作为不变的 `thread_id`。`interrupt` 仍在 `request_clarification` 内；入口读取回答，用同一个图实例、同一配置和 `Command(resume=...)` 恢复。[代码：run.py:78](../../src/atmospheric_research_agent/agent/run.py#L78)、[nodes.py:1329](../../src/atmospheric_research_agent/agent/nodes.py#L1329)、[run.py:91](../../src/atmospheric_research_agent/agent/run.py#L91)
- 协议关闭在此前的 `agent` 节点完成，不是在暂停节点内补回应。`request_clarification` 重入时，`interrupt` 前只有读取和校验待追问文本，没有模型、工具调用或 State 写入；外层观测包装器也在节点正常返回后才汇总 trace。不会因这一节点重入再次执行前一批工具或重复关闭协议。[代码：nodes.py:315](../../src/atmospheric_research_agent/agent/nodes.py#L315)、[nodes.py:1322](../../src/atmospheric_research_agent/agent/nodes.py#L1322)、[graph.py:52](../../src/atmospheric_research_agent/agent/graph.py#L52)
- 暂停期间不执行后续研究，不写完成时间，不记录 `run_completed`。恢复时保留已有证据、消息、计划、召回记忆和预算；固定边直接回 `agent`，不重新经过 `START / recall_memory`。[代码：run.py:91](../../src/atmospheric_research_agent/agent/run.py#L91)、[graph.py:268](../../src/atmospheric_research_agent/agent/graph.py#L268)
- 非空回答追加到 `messages` 和 `clarification_history`，清空待回答问题和旧 `assessment`，不覆盖原始 `question`；评估、写作、核验、修订通过 `_task_question` 接收用户补充。两个列表使用追加型 reducer。[代码：nodes.py:1330](../../src/atmospheric_research_agent/agent/nodes.py#L1330)、[state.py:180](../../src/atmospheric_research_agent/agent/state.py#L180)、[state.py:207](../../src/atmospheric_research_agent/agent/state.py#L207)、[nodes.py:703](../../src/atmospheric_research_agent/agent/nodes.py#L703)、[nodes.py:758](../../src/atmospheric_research_agent/agent/nodes.py#L758)、[nodes.py:958](../../src/atmospheric_research_agent/agent/nodes.py#L958)、[nodes.py:1004](../../src/atmospheric_research_agent/agent/nodes.py#L1004)、[nodes.py:1160](../../src/atmospheric_research_agent/agent/nodes.py#L1160)
- CLI 的运行耗时排除等待用户输入的时间。[代码：run.py:95](../../src/atmospheric_research_agent/agent/run.py#L95)

## 使用

完整人工交互测试（真实研究图、真实模型和工具，首次问题及后续回答全部由你输入）：

```powershell
.venv\Scripts\python examples/05_interactive_research.py
```

脚本不会强制触发澄清；模型判断需要补充时才暂停。结束后显示报告、运行轨迹和本次问答记录。
它调用正式 `run_research`，没有替换任何业务节点；不是模拟图。该脚本只把轨迹打印到终端，没有传入 `run_research(debug=True)`。[代码：05_interactive_research.py:41](../../examples/05_interactive_research.py#L41)

先运行不消耗 API 的交互演示，输入年份后查看暂停前后的证据与计数对比：

```powershell
.venv\Scripts\python examples/03_clarification_resume.py
```

演示使用简化图、模拟资料以及项目真实的澄清节点，直接设置 `clarification_question`；它验证恢复机制，不覆盖正式 `ask_user` 决策、优先级和共用额度，也不代表真实模型或工具已通过测试。[代码：03_clarification_resume.py:17](../../examples/03_clarification_resume.py#L17)
自动复现可追加 `--answer "2021 年"`。下列命令则运行真实研究图，会调用外部模型和工具：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.agent.run "请查询当前天气并给出带来源的简要分析。" --debug
```

没有地点且还有澄清额度时，提示词要求先调用 `ask_user`。在“你的回答”后输入城市，按回车继续；实际输出仍由模型决定。此正式命令的 `--debug` 还会在结束后保存诊断材料到已忽略的 `docs/validation/<日期>/<run_id>/`。[代码：nodes.py:68](../../src/atmospheric_research_agent/agent/nodes.py#L68)、[run.py:80](../../src/atmospheric_research_agent/agent/run.py#L80)、[run.py:118](../../src/atmospheric_research_agent/agent/run.py#L118)

带自动检查的真实模型验收（会调用 API）：

```powershell
.venv\Scripts\python examples/04_live_clarification.py
```

也可追加 `--answer 北京` 自动回答。脚本检查一次澄清、回答记录、清除追问、最终输出和天气证据。
若重复追问会停止验收，这是该脚本要求“恰好一次澄清”的限制，不是主流程的两次上限。它沿用正常流程；当前天气任务会跳过长期记忆存储。[代码：04_live_clarification.py:17](../../examples/04_live_clarification.py#L17)、[04_live_clarification.py:30](../../examples/04_live_clarification.py#L30)、[nodes.py:1235](../../src/atmospheric_research_agent/agent/nodes.py#L1235)

直接调用图时应传入 checkpointer 与 thread_id，并保存该图实例和配置用于恢复。
`run_research(..., answer_provider=...)` 可由调用者提供回答函数。没有回答函数时返回
暂停结果，不会伪装为任务完成；这个便利函数不提供跨调用恢复句柄，外部应用需要自行保留图实例和恢复配置。[代码：run.py:78](../../src/atmospheric_research_agent/agent/run.py#L78)、[run.py:91](../../src/atmospheric_research_agent/agent/run.py#L91)

## 边界

内存检查点仅在进程存活、原图实例及检查点仍可用时有效，恢复必须使用同一 `thread_id`。退出程序、Ctrl+C 或关闭终端导致进程结束后，暂停任务的检查点会丢失，重启不能恢复。[代码：run.py:78](../../src/atmospheric_research_agent/agent/run.py#L78)

两次上限限制的是已完成问答数量，不保证这两次问题内容不重复；合并同批追问的去重不等于跨轮语义去重。额度检查在主流程的入口和路由，不在 `request_clarification_node` 内；直接调用该节点不会自行限制追问次数。[代码：settings.py:20](../../src/atmospheric_research_agent/agent/settings.py#L20)、[nodes.py:308](../../src/atmospheric_research_agent/agent/nodes.py#L308)、[nodes.py:1322](../../src/atmospheric_research_agent/agent/nodes.py#L1322)

如果回答纠正了此前研究对象、地点或年份，恢复节点不会按这些语义条件自动删除或标记旧证据失效。它只追加回答、清空旧评估并记录当前过期天气数量；保留记录不代表这些证据仍然适用。[代码：nodes.py:1333](../../src/atmospheric_research_agent/agent/nodes.py#L1333)

2026-10-20 后续输出质量迭代已增加“当前天气按有效时间过期”的读取视图过滤，
但它不等于地点/年份等语义约束改变后的重评估；原始记录保留，`usable_evidence` 排除过期或时间无法可靠判断的天气记录。详见 [输出审查记录](output_review_2026-10-20.md)。[代码：evidence.py:8](../../src/atmospheric_research_agent/agent/evidence.py#L8)、[evidence.py:25](../../src/atmospheric_research_agent/agent/evidence.py#L25)

## 离线验证

```powershell
.venv\Scripts\python -m pytest tests/test_clarification_priority.py tests/test_clarification_resume.py tests/test_assessment_quota.py tests/test_agent_node.py tests/test_graph.py tests/test_run.py tests/test_state.py --basetemp .pytest-tmp
```

离线测试覆盖两个澄清入口、同任务暂停恢复、证据和预算保留、补充条件传递，以及追问问题非空校验；优先级测试另覆盖混合调用、参数无效、超限、每个 `tool_call_id` 恰好一条回应及恢复后的协议完整性。额度测试覆盖明确状态句和模型解释与程序额度不一致时的路由。这里列出的是现有测试覆盖范围，不代表真实 API 运行验收。[测试：test_clarification_resume.py](../../tests/test_clarification_resume.py)、[test_clarification_priority.py](../../tests/test_clarification_priority.py)、[test_assessment_quota.py](../../tests/test_assessment_quota.py)
