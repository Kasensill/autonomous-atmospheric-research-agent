"""与界面无关的同步研究执行；一个实例独占一张图与检查点。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Callable
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from atmospheric_research_agent.agent.graph import build_research_graph
from atmospheric_research_agent.agent.settings import GRAPH_RECURSION_LIMIT
from atmospheric_research_agent.agent.state import create_initial_state


class ResearchSession:
    """由调用方串行驱动；不提供跨进程恢复或硬取消。"""

    def __init__(self, question: str, *, run_id: str | None = None,
                 on_error: Callable[[Exception], None] | None = None,
                 material_paths: list[str | Path] | None = None, material_role: str = 'auto',
                 material_snapshot: dict | None = None, conversation_context: list | None = None) -> None:
        self.state: dict[str, Any] = dict(create_initial_state(question))
        if run_id is not None:
            self.state["run_id"] = run_id
        if material_paths:
            from ..agent.materials import load_materials
            loaded = load_materials(material_paths, question, material_role)
            for key in ('evidence', 'materials', 'material_role'):
                self.state[key] = loaded[key]
            self.state['trace'].extend(loaded['trace'])
        if material_snapshot is not None:
            from copy import deepcopy
            from ..agent.materials import decide_material_role
            role, reason = decide_material_role(question, material_role)
            self.state['materials'] = deepcopy(material_snapshot.get('materials', []))
            self.state['evidence'] = deepcopy(material_snapshot.get('evidence', []))
            self.state['material_role'] = role
            for item in self.state['evidence']:
                item['role'] = role
            self.state['trace'].append({'node': 'conversation', 'event': 'material_role_decided',
                'detail': f'复用已解析材料；本轮角色：{role}；{reason}', 'role': role})
        if conversation_context:
            self.state['conversation_context'] = conversation_context
            self.state['trace'].append({'node': 'conversation', 'event': 'conversation_context_loaded',
                'detail': '历史仅作背景，不继承旧证据、旧引用或已用预算。',
                'turn_count': len(conversation_context)})
        self.on_error = on_error
        self.checkpointer = InMemorySaver()
        self.graph = build_research_graph(checkpointer=self.checkpointer)
        self.config = {
            "recursion_limit": GRAPH_RECURSION_LIMIT,
            "configurable": {"thread_id": self.state["run_id"]},
        }
        self.status = "ready"
        self.question = ""
        self.error = ""
        self.elapsed_ms = 0.0

    def advance(self, answer: str | None = None,
                on_progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        """首次执行或恢复一次，运行到下一次 interrupt／结束。

        同步执行避免页面为同一任务启动多个后台线程。已结束／失败的任务不重试，
        防止一个模糊的“重试”重复调用模型、工具或写入记忆。
        """
        if self.status == "waiting":
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("请先填写非空回答。")
            payload: Any = Command(resume=answer.strip())
        elif self.status == "ready" and answer is None:
            payload = self.state
        else:
            raise ValueError("当前任务不能再次执行；请新建研究任务。")

        self.status = "running"
        self.question = ""
        started = perf_counter()
        interrupts: tuple = ()
        try:
            # values 是已应用 reducer 的完整状态，不能再手动追加 evidence/trace。
            for values in self.graph.stream(payload, config=self.config, stream_mode="values"):
                if "__interrupt__" in values:
                    interrupts = tuple(values["__interrupt__"])
                self.state.update({key: value for key, value in values.items()
                                   if key != "__interrupt__"})
                if on_progress is not None:
                    on_progress(self.state)

            snapshot = self.graph.get_state(self.config)
            self.state.update(snapshot.values)
            interrupts = tuple(getattr(snapshot, "interrupts", ())) or interrupts
            if not interrupts:
                interrupts = tuple(item for task in snapshot.tasks for item in task.interrupts)
            if interrupts:
                value = interrupts[0].value
                self.question = value.get("question", "") if isinstance(value, dict) else str(value)
                if not self.question:
                    raise RuntimeError("暂停状态缺少追问内容。")
                self.status = "waiting"
            elif snapshot.next:
                raise RuntimeError("图尚有待执行节点，但未返回追问。")
            elif not self.state.get("final_report"):
                raise RuntimeError("图已结束但没有最终输出。")
            else:
                self.status = "completed"
                self.state["finished_at"] = datetime.now(UTC).isoformat()
        except Exception as error:
            if self.on_error is not None:
                self.on_error(error)
            # 不把可能带 URL 凭据／请求参数的异常原文输出到网页或下载包。
            self.error = (
                "流程超过步数上限被中止，未交付报告。"
                if isinstance(error, GraphRecursionError)
                else f"本轮执行未完成（{type(error).__name__}）。请检查终端配置与服务可用性。"
            )
            self.status = "failed"
            try:
                self.state.update(self.graph.get_state(self.config).values)
            except Exception:
                # 检查点读取失败时仍保留最后一份流式状态。
                pass
            self.state["trace"] = [*self.state.get("trace", []), {
                "node": "web_session", "event": "execution_failed", "status": "error",
                "timestamp": datetime.now(UTC).isoformat(), "detail": self.error,
            }]
        finally:
            self.elapsed_ms += (perf_counter() - started) * 1000
            # Streamlit 停止／重绘会用 BaseException 中止脚本，不属于普通异常。
            # 不自动重新执行这一批，避免未知完成程度的工具被重复调用。
            if self.status == "running":
                self.status = "failed"
                self.error = "页面执行被中断，当前任务不自动重试。请保留已有材料后新建任务。"
            if self.status in {"completed", "failed"}:
                self.state["run_duration_ms"] = round(self.elapsed_ms, 2)

    @property
    def ending(self) -> str:
        if self.status == "failed":
            return "执行中止"
        if self.status == "waiting":
            return "等待补充信息"
        if self.status != "completed":
            return "研究进行中"
        if any(item.get("node") == "finalize_limited_report"
               for item in self.state.get("trace", [])):
            return "失败说明"
        return "受限报告" if self.state.get("report_mode") == "limited" else "完整报告"

    def export_json(self, field: str) -> str:
        if field not in {"trace", "evidence"}:
            raise ValueError("只允许导出 Trace 或证据。")
        return json.dumps({"run_id": self.state["run_id"], field: self.state.get(field, [])},
                          ensure_ascii=False, indent=2)

    def close(self) -> None:
        """仅在执行结束或暂停时调用，释放所有图检查点。"""
        self.checkpointer.delete_thread(self.state["run_id"])
        self.graph = None
