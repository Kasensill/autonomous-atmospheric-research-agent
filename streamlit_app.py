"""启动：python -m streamlit run streamlit_app.py --server.address 127.0.0.1"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from atmospheric_research_agent.agent.settings import (
    MAX_CLARIFICATIONS, MAX_RESEARCH_ROUNDS, MAX_TOOL_CALLS,
)
from atmospheric_research_agent.ui.session import ResearchSession


NODE_LABELS = {
    "recall_memory": "召回研究背景", "agent": "规划与选择工具",
    "tools": "执行研究工具", "record_results": "整理正式证据",
    "research_guard": "检查证据与程序预算", "assess_evidence": "评估证据缺口",
    "request_clarification": "等待你补充条件", "generate_draft": "起草研究报告",
    "verify_report": "核验报告", "revise_report": "按核验意见修订",
    "finalize_report": "定稿", "finalize_limited_report": "生成失败说明",
    "store_memory": "检查长期记忆准入",
}


def reset_task() -> None:
    """按钮回调在页面重绘前清理界面输入，不删除磁盘材料或长期记忆。"""
    for key in list(st.session_state):
        if key in {"research_task", "question_draft", "confirm_new_task"} or key.startswith(
            ("active_question_", "clarification_answer_")
        ):
            del st.session_state[key]


def request_new_task() -> None:
    task = st.session_state.get("research_task")
    if task is not None and task.status not in {"completed", "failed"}:
        st.session_state["confirm_new_task"] = True
    else:
        reset_task()


def render_progress(state: dict, activity, counters) -> None:
    completed = [item for item in state.get("trace", []) if item.get("event") == "node_completed"]
    if completed:
        latest = completed[-1]
        activity.info(f"已完成：{NODE_LABELS.get(latest['node'], latest['node'])} · 正在继续研究")
    counters.caption(
        f"研究轮次 {state.get('research_round', 0)}/{MAX_RESEARCH_ROUNDS}　·　"
        f"工具调用 {state.get('tool_call_count', 0)}/{MAX_TOOL_CALLS}　·　"
        f"已收集 {len(state.get('evidence', []))} 条原始证据"
    )


def render_results(task: ResearchSession) -> None:
    state = task.state
    a, b, c, d = st.columns(4)
    a.metric("交付状态", task.ending)
    b.metric("已收集证据", len(state.get("evidence", [])))
    c.metric("工具调用", f"{state.get('tool_call_count', 0)} / {MAX_TOOL_CALLS}")
    d.metric("执行耗时", f"{task.elapsed_ms / 1000:.1f} 秒")
    st.caption("耗时不含等待回答；证据数量包含原始记录，不代表全部适用或结论可靠。")

    report, evidence = st.tabs(["研究报告", "证据与核验"])
    with report:
        if task.status == "completed":
            if task.ending == "失败说明":
                st.warning("本次未形成可交付的研究报告，下面是失败说明。")
            elif task.ending == "受限报告":
                st.warning("证据不完整，只交付可支持的部分；受限报告不写入长期记忆。")
            else:
                st.success("报告已通过系统核验；这不等于事实得到独立认证。")
            st.markdown(state["final_report"])
            st.download_button("下载 Markdown", state["final_report"],
                               file_name=f"research_{state['run_id']}.md", mime="text/markdown")
        elif task.status == "failed":
            st.error(task.error)
            st.info("已保留本会话可读取的证据和轨迹。不自动重试，避免重复请求；可下载后新建任务。")
        else:
            st.info("研究尚未结束。补充条件后继续同一任务，草稿不会作为最终报告交付。")
    with evidence:
        review = state.get("report_review")
        if review:
            st.caption("模型审查意见与程序硬检查的合并结果")
            st.json(review, expanded=False)
        for index, record in enumerate(state.get("evidence", []), 1):
            with st.expander(f"E{index} · {record.get('source_display_name', '未命名来源')}"):
                st.text(record.get("source_location", ""))
                st.caption(f"有效时间：{record.get('valid_at') or '未提供／不适用'}")
                st.text(record.get("content", ""))
        if not state.get("evidence"):
            st.caption("尚未取得正式证据。搜索候选不在这里冒充证据。")
        st.download_button("下载原始证据 JSON", task.export_json("evidence"),
                           file_name=f"evidence_{state['run_id']}.json", mime="application/json")
    # 放在结果页底部，不依赖当前选择哪个报告／证据页签。
    st.divider()
    st.subheader("本次研究记录")
    with st.expander("问题与补充记录", expanded=False):
        st.write("研究问题：", state["question"])
        history = state.get("clarification_history", [])
        if not history:
            st.caption("本次没有已完成的追问。")
        for index, item in enumerate(history, 1):
            st.markdown(f"**补充条件 {index}**")
            st.write("追问：", item.get("question", ""))
            st.write("回答：", item.get("answer", ""))
    with st.expander("运行记录与完整 Trace", expanded=False):
        st.caption(f"开始时间（UTC）：{state.get('started_at', '未记录')} · "
                   f"结束时间（UTC）：{state.get('finished_at') or '尚未结束'}")
        st.caption(f"研究轮次：{state.get('research_round', 0)} / {MAX_RESEARCH_ROUNDS} · "
                   f"报告修订：{state.get('report_revision_count', 0)} 次")
        events = state.get("trace", [])
        st.caption("Trace 是节点和业务事件记录；下载文件保留完整事件字段。")
        rows = [{"序号": index, "节点": item.get("node", ""),
                 "事件": item.get("event", ""), "状态": item.get("status", ""),
                 "说明": str(item.get("detail", ""))[:300]}
                for index, item in enumerate(events, 1)]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.json(events, expanded=False)
        else:
            st.caption("本次尚无 Trace 事件。")
        st.download_button("下载完整 Trace", task.export_json("trace"),
                           file_name=f"trace_{state['run_id']}.json", mime="application/json")
    st.caption(f"任务编号：{state['run_id']}")


def main() -> None:
    st.set_page_config(page_title="大气研究工作台", page_icon="🌤️", layout="wide")
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    st.markdown("""<style>
        .block-container {max-width: 1240px; padding-top: 2.3rem;}
        [data-testid="stMetric"] {border: 1px solid #b7c9dd55; border-radius: 12px; padding: 16px;}
        [data-testid="stSidebar"] {border-right: 1px solid #b7c9dd33;}
        </style>""", unsafe_allow_html=True)
    task = st.session_state.get("research_task")
    with st.sidebar:
        st.markdown("### 🌤️ 大气研究工作台")
        st.caption("证据驱动 · 有限研究 · 可追溯交付")
        st.button("＋ 新建研究", type="primary", use_container_width=True,
                  disabled=task is not None and task.status == "running",
                  on_click=request_new_task)
        if st.session_state.get("confirm_new_task"):
            st.warning("新建会放弃当前未完成任务，无法从本页面继续；已用 API 费用不会退回。")
            confirm, cancel = st.columns(2)
            confirm.button("确认新建", on_click=reset_task, use_container_width=True)
            cancel.button("继续当前任务", on_click=lambda: st.session_state.pop("confirm_new_task", None),
                          use_container_width=True)
        st.divider()
        st.markdown("**连接配置**")
        for name, label in [("DEEPSEEK_API_KEY", "研究模型"),
                            ("DASHSCOPE_API_KEY", "向量检索与重排"),
                            ("TAVILY_API_KEY", "网页搜索")]:
            st.caption(f"{'● 已配置' if os.getenv(name) else '○ 未配置'} · {label}")
        st.caption("这里只检查配置是否存在，不代表凭据或服务已验证。密钥在本地 .env 配置。")
        st.divider()
        st.markdown("**程序预算**")
        st.caption(f"研究 {MAX_RESEARCH_ROUNDS} 轮 · 工具 {MAX_TOOL_CALLS} 次 · 追问 {MAX_CLARIFICATIONS} 次")
        st.caption("由现有研究图控制，界面不修改预算。")
        st.divider()
        st.caption("本地单人工作台。刷新、断线或重启可能丢失会话，请及时下载。图状态按会话隔离；长期项目记忆仍复用本地存储，不提供多用户隔离。")

    if task is None:
        st.title("把问题，交给证据。")
    else:
        title_question = " ".join(task.state["question"].split())
        title_question = title_question[:90] + "…" if len(title_question) > 90 else title_question
        st.title(f"当前问题：{title_question}")
    st.write("提出大气科学问题，查看取证过程，获得带来源的研究报告。")
    st.caption("可研究一般机制或具体事件；缺少关键条件时会暂停追问，证据不足时受限交付或说明失败。")
    submitted = False
    question = ""
    if task is None:
        with st.form("research_question"):
            question = st.text_area("研究问题", height=135, key="question_draft",
                                    placeholder="例如：为什么副热带高压主体常晴热少雨，边缘也一样吗？")
            submitted = st.form_submit_button("开始研究", type="primary")
            st.caption("开始后会调用已配置的模型和工具，可能产生 API 费用。本版支持文本问题和公开链接，不支持文件上传。")
    else:
        # 任务的原始问题是唯一数据来源，不依赖编辑控件在重绘时的默认值。
        st.text_area("研究问题", value=task.state["question"], height=135,
                     key=f"active_question_{task.state['run_id']}", disabled=True)

    execute = task is not None and task.status == "ready"
    answer = None
    if submitted:
        if not question.strip():
            st.warning("请填写研究问题。")
        elif not os.getenv("DEEPSEEK_API_KEY"):
            st.error("尚未配置 DEEPSEEK_API_KEY，请先在本地 .env 中配置研究模型。")
        else:
            task = ResearchSession(question)
            st.session_state["research_task"] = task
            # 先重绘为当前任务标题与只读原问题，再开始运行。
            st.rerun()

    if task is None:
        st.divider()
        left, middle, right = st.columns(3)
        left.markdown("**01 · 先取证**\n\n搜索摘要只是候选，读取与整理后的内容才成为正式证据。")
        middle.markdown("**02 · 再核验**\n\n报告先起草再核验，有必须修改项时受控修订。")
        right.markdown("**03 · 如实交付**\n\n区分完整报告、受限报告和失败说明，并保留运行轨迹。")
        return

    if task.status == "waiting":
        st.warning("研究已暂停，等待你补充信息。已有证据和已用预算保留。")
        st.write(task.question)
        history_count = len(task.state.get("clarification_history", []))
        st.caption(f"补充条件 · 第 {history_count + 1} 次")
        with st.form(f"clarify_{task.state['run_id']}_{history_count}"):
            # 每轮独立 key：下一次追问为空；本轮空回答／未成功接收则保留原输入。
            answer = st.text_area("你的补充", key=f"clarification_answer_{task.state['run_id']}_{history_count}",
                                  placeholder="请回答上面的问题，例如地点、日期或研究对象。")
            resume = st.form_submit_button("提交回答，继续原任务", type="primary")
        if resume:
            if not answer.strip():
                st.warning("回答不能为空；原任务仍在等待。")
            else:
                execute = True

    if execute:
        activity = st.empty()
        counters = st.empty()
        with st.status("正在研究，请等待节点完成…", expanded=True) as status:
            task.advance(answer, on_progress=lambda state: render_progress(state, activity, counters))
            status.update(label=task.ending, state="error" if task.status == "failed" else "complete")
        st.rerun()

    render_results(task)


if __name__ == "__main__":
    main()
