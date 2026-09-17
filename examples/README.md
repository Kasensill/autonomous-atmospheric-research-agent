# Tool Calling 学习示例

## 第一步：观察工具请求

在项目根目录运行：

```powershell
.venv\Scripts\python -m pip install -e ".[learning]"
.venv\Scripts\python examples/01_request_tool.py
```

每次运行会调用一次真实模型，产生对应 API 费用。工具不会执行。

根目录 `.env` 必须有 `DEEPSEEK_API_KEY`。可选配置：

```dotenv
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
```

默认使用 DeepSeek 官方地址及当前文档中的模型名称。第三方服务请配置自己的地址、模型名称，并确认兼容工具调用以及 thinking 参数。不要提交或分享真实密钥。

阅读顺序：`add_numbers` 定义能力 → `bind_tools` 准备说明 → `invoke` 请求模型 → 查看 `content` 和 `tool_calls`。

预期观察：请求中包含 `name`、`args` 和 `id`。即使模型输出了计算答案，也不能说明本地函数执行过。这个文件没有 Tool Node，也没有执行工具的调用。

## 第二步：手动完成一次工具往返

```powershell
.venv\Scripts\python examples/02_manual_tool_roundtrip.py
```

这次会有两次模型调用：第一次请求工具；程序手动执行 `add_numbers` 后，把结果写成带相同调用 ID 的 `ToolMessage`；第二次模型根据该结果回答。它仍未使用 LangGraph。
