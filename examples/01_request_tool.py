"""第一步：观察模型提出的工具请求；本文件不会执行加法工具。"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


def main() -> None:
    # 显式定位项目根目录，因此从其他目录启动也能读取正确配置。
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请在项目根目录 .env 中配置 DEEPSEEK_API_KEY。")

    model = ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
        temperature=0,
        timeout=60,
        max_retries=0,
        max_tokens=512,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # bind_tools 只准备工具说明，此时既没有请求模型，也没有执行工具。
    model_with_tools = model.bind_tools([add_numbers])

    # invoke 才发出模型请求。用提示要求调用，保留实际结果供观察。
    messages = [("user", "请使用 add_numbers 工具计算 137 加 286。")]
    try:
        response = model_with_tools.invoke(messages)
    except Exception as error:
        # 不打印原始异常或请求对象，避免连接配置出现在日志中。
        raise SystemExit(
            f"模型请求失败（{type(error).__name__}）。请检查地址、模型、密钥、余额和网络。"
        ) from None

    print("content:", response.content)
    print("tool_calls:")
    print(json.dumps(response.tool_calls, ensure_ascii=False, indent=2))
    if response.invalid_tool_calls:
        print("检测到无法解析的工具请求，请检查模型的工具调用支持。")
    elif not response.tool_calls:
        print("本次模型没有提出工具请求；这不表示工具已经执行。")
    print("示例结束：没有调用 add_numbers.invoke，工具尚未执行。")


if __name__ == "__main__":
    main()
