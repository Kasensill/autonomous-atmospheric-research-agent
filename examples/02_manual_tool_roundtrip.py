"""第二步：手动执行一次工具请求，并把结果送回模型。

本文件还没有 LangGraph。它把未来 Tool Node 所做的事直接写出来，
便于观察每条消息如何进入下一轮模型调用。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


def make_model() -> ChatOpenAI:
    """读取本地配置，并创建可提出工具请求的模型对象。"""

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请在项目根目录 .env 中配置 DEEPSEEK_API_KEY。")

    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        temperature=0,
        timeout=60,
        max_retries=0,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )


def main() -> None:
    # bind_tools 仍然只是给模型提供工具说明，不会执行 add_numbers。
    model_with_tools = make_model().bind_tools([add_numbers])
    messages = [
        HumanMessage(content="请使用 add_numbers 工具计算 137 加 286。"),
    ]

    # 第一轮：模型提出工具调用请求。
    first_response = model_with_tools.invoke(messages)
    print("第一轮模型请求：", first_response.tool_calls)
    if len(first_response.tool_calls) != 1:
        raise SystemExit("本示例预期得到一条工具请求，请重新运行或检查模型工具调用支持。")

    tool_call = first_response.tool_calls[0]

    # 这三行就是未来 Tool Node 的核心职责：找工具、传参数、得到结果。
    tool_result = add_numbers.invoke(tool_call["args"])
    print("本地工具实际返回：", tool_result)

    # ToolMessage 用同一个 ID 将结果对应回模型刚才的请求。
    tool_message = ToolMessage(
        content=str(tool_result),
        tool_call_id=tool_call["id"],
        name=tool_call["name"],
    )

    # 第二轮模型会看到：用户请求、自己的 Tool Call、工具执行结果。
    messages.extend([first_response, tool_message])
    final_response = model_with_tools.invoke(messages)

    print("第二轮模型回答：", final_response.content)
    print("第二轮是否还请求工具：", bool(final_response.tool_calls))


if __name__ == "__main__":
    main()
