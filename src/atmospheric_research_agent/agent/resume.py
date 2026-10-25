"""在 LangGraph 写入恢复值前校验应用的文本问答协议。"""

from typing import Any

from langgraph.types import Command


def _validate_input(value: Any) -> None:
    if not isinstance(value, Command):
        return
    # 保留仅更新状态／跳转的 Command；裸 Command(resume=None) 不是合法回答。
    if value.resume is None and (value.update is not None or value.goto):
        return
    answers = list(value.resume.values()) if isinstance(value.resume, dict) else [value.resume]
    if not answers or any(not isinstance(answer, str) or not answer.strip() for answer in answers):
        raise ValueError("澄清回答必须是非空文本；非法值未提交到检查点，请重新输入。")


class ResumeValidatedGraph:
    """保留编译图的状态查询；同步／异步执行入口先拒绝非法恢复值。

    节点内校验仍保留作防御，但它发生在框架接收恢复值之后，不能代替入口校验。
    不修改 LangGraph、检查点或图的节点与连线。
    """

    def __init__(self, graph: Any) -> None:
        self._graph = graph

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        _validate_input(input)
        return self._graph.invoke(input, config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        _validate_input(input)
        return await self._graph.ainvoke(input, config, **kwargs)

    def stream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        _validate_input(input)
        return self._graph.stream(input, config, **kwargs)

    def astream(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        _validate_input(input)
        return self._graph.astream(input, config, **kwargs)
