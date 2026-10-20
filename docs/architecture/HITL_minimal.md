# 最小人工澄清：同任务暂停与恢复

本阶段只实现同一进程内的暂停恢复，不增加前置意图节点、SQLite、前端或证据筛选机制。

## 流程

`agent / assess_evidence → request_clarification → interrupt → 用户回答 → agent`

- 运行入口配置 `InMemorySaver`，以初始 `run_id` 作为本次任务不变的 `thread_id`。
- `interrupt` 把追问返回入口；入口读取命令行回答，用同一配置和 `Command(resume=...)` 恢复。
- 暂停期间不执行后续研究，不写完成时间，不记录 `run_completed`。
- 恢复时保留已有证据、消息、计划、历史记忆和预算；不重新经过 START/recall_memory。
- 回答追加到消息和 `clarification_history`，清空待回答问题和旧证据评估。
- 评估、写作、核验与修订的输入均带上澄清记录；原始 question 不覆盖。
- 评估模型选择 clarify 时必须提供非空的具体追问。
- 暂停节点恢复时从头重入，中断之前不调用模型、工具或增加计数。
- CLI 的运行耗时排除等待用户输入的时间。

## 使用

完整人工交互测试（真实研究图、真实模型和工具，首次问题及后续回答全部由你输入）：

```powershell
.venv\Scripts\python examples/05_interactive_research.py
```

脚本不会强制触发澄清；模型判断需要补充时才暂停。结束后显示报告、运行轨迹和本次问答记录。
它调用正式 `run_research`，没有替换任何业务节点；不是模拟图。

先运行不消耗 API 的交互演示，输入年份后查看暂停前后的证据与计数对比：

```powershell
.venv\Scripts\python examples/03_clarification_resume.py
```

演示使用简化图、模拟资料以及项目真实的澄清节点；它验证恢复机制，不代表真实模型或工具已通过测试。
自动复现可追加 `--answer "2021 年"`。下列命令则运行真实研究图，会调用外部模型和工具：

```powershell
.venv\Scripts\python -m atmospheric_research_agent.agent.run "请查询当前天气并给出带来源的简要分析。" --debug
```

没有地点时应提出追问。在“你的回答”后输入城市，按回车继续。实际是否追问由模型决定。

带自动检查的真实模型验收（会调用 API）：

```powershell
.venv\Scripts\python examples/04_live_clarification.py
```

也可追加 `--answer 北京` 自动回答。脚本检查一次澄清、回答记录、清除追问、最终输出和天气证据。
若重复追问会停止验收，避免自动无限回答。它沿用正常流程，合格报告可能写入项目记忆。

直接调用图时应传入 checkpointer 与 thread_id，并保存该图实例和配置用于恢复。
`run_research(..., answer_provider=...)` 可由调用者提供回答函数。没有回答函数时返回
暂停结果，不会伪装为任务完成；这个便利函数不提供跨调用恢复句柄，外部应用应直接管理图。

## 边界

内存检查点仅在进程存活期间有效。退出程序、Ctrl+C 或关闭终端后不能恢复。
本版不保证模型永不重复提问，也未实现约束变化后的证据自动筛选。
如果回答纠正了此前研究对象，旧证据不会自动删除或失效，这属于后续改进。

2026-10-20 后续输出质量迭代已增加“当前天气按有效时间过期”的读取视图过滤，
但它不等于地点/年份等语义约束改变后的重评估。详见 [输出审查记录](output_review_2026-10-20.md)。

## 离线验证

```powershell
.venv\Scripts\python -m pytest tests/test_clarification_resume.py tests/test_agent_node.py tests/test_graph.py tests/test_run.py tests/test_state.py --basetemp .pytest-tmp
```

测试以假模型驱动真实 LangGraph，覆盖两个澄清入口、暂停后没有写草稿、同任务恢复、
证据和预算保留、回答追加、补充条件传递以及澄清缺少具体问题时的校验。
