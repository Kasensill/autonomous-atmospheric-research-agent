"""启动：python -m streamlit run streamlit_app.py --server.address 127.0.0.1"""

from uuid import uuid4
from hashlib import sha256
from html import escape
import re

import streamlit as st

from atmospheric_research_agent.agent.settings import (
    MAX_CLARIFICATIONS, MAX_RESEARCH_ROUNDS, MAX_TOOL_CALLS,
    MATERIAL_MAX_FILES, MATERIAL_UPLOAD_MAX_BYTES,
)
from atmospheric_research_agent.ui.client import ApiError, RemoteResearchTask, ResearchClient
from atmospheric_research_agent.agent.materials import coverage_text


NODE_LABELS = {
    "recall_memory": "召回研究背景", "agent": "规划与选择工具",
    "tools": "执行研究工具", "record_results": "整理正式证据",
    "research_guard": "检查证据与程序预算", "assess_evidence": "评估证据缺口",
    "request_clarification": "等待你补充条件", "generate_draft": "起草研究报告",
    "verify_report": "核验报告", "revise_report": "按核验意见修订",
    "finalize_report": "定稿", "finalize_limited_report": "生成失败说明",
    "store_memory": "检查长期记忆准入",
}

MATERIAL_ROLE_LABELS = {
    'auto': '自动判断（不明确时作为待核实线索）',
    'clue': '待核实线索：用于检索，不能直接证明事实',
    'analysis': '分析对象：总结、解读原文，不替原文背书',
    'both': '两者并用：分析原文并核实其中说法',
}


def reset_task() -> None:
    """按钮回调在页面重绘前清理界面输入，不删除磁盘材料或长期记忆。"""
    for key in list(st.session_state):
        if key in {"research_task", "question_draft", "confirm_new_task", "create_request", "material_role_choice"} or key.startswith(
            ("active_question_", "clarification_answer_", "material_uploads_")
        ):
            del st.session_state[key]
    st.session_state['material_upload_generation'] = st.session_state.get('material_upload_generation', 0) + 1


def request_new_task() -> None:
    task = st.session_state.get("research_task")
    if task is not None and task.status not in {"completed", "failed", "abandoned", "interrupted"}:
        st.session_state["confirm_new_task"] = True
    else:
        reset_task()


def clear_delete_request() -> None:
    st.session_state.pop("pending_delete", None)


@st.dialog("删除历史任务", on_dismiss=clear_delete_request)
def confirm_delete_task(client: ResearchClient, view: dict, *, last_on_page: bool) -> None:
    st.markdown(
        f'<div class="delete-task-copy"><div class="delete-task-question">{escape(view["question"])}</div>'
        '<p class="delete-task-description">任务档案、检查点和关联记忆将一并删除。其他任务保留。</p></div>',
        unsafe_allow_html=True,
    )
    with st.container(key="delete_task_actions", horizontal=True, horizontal_alignment="right", gap="small"):
        cancelled = st.button("取消")
        confirmed = st.button("确认删除", type="primary")
    if cancelled:
        clear_delete_request()
        st.rerun()
    if confirmed:
        try:
            client.request("DELETE", f'/runs/{view["run_id"]}')
        except ApiError as error:
            st.error(str(error))
            return
        clear_delete_request()
        task = st.session_state.get("research_task")
        if task is not None and task.state["run_id"] == view["run_id"]:
            reset_task()
        if last_on_page:
            st.session_state["history_offset"] = max(0, st.session_state.get("history_offset", 0) - 20)
        st.rerun()


def render_progress(state: dict, activity, counters) -> None:
    completed = [item for item in state.get("trace", []) if item.get("event") == "node_completed"]
    latest_event = next((item for item in reversed(state.get("trace", []))
                         if item.get("event") in {"node_completed", "model_request_started", "model_response_headers"}), None)
    if latest_event and latest_event.get("event") in {"model_request_started", "model_response_headers"}:
        name = latest_event.get("graph_node") or "模型调用"
        activity.info(f"{NODE_LABELS.get(name, name)} · {latest_event['detail']}")
    elif completed:
        latest = completed[-1]
        activity.info(f"已完成：{NODE_LABELS.get(latest['node'], latest['node'])} · 正在继续研究")
    counters.caption(
        f"研究轮次 {state.get('research_round', 0)}/{MAX_RESEARCH_ROUNDS}　·　"
        f"工具调用 {state.get('tool_call_count', 0)}/{MAX_TOOL_CALLS}　·　"
        f"已收集 {len(state.get('evidence', []))} 条原始证据"
    )


def render_results(task: RemoteResearchTask) -> None:
    state = task.state
    a, b, c, d = st.columns(4)
    a.metric("交付状态", task.ending)
    b.metric("已收集证据", len(state.get("evidence", [])))
    c.metric("工具调用", f"{state.get('tool_call_count', 0)} / {MAX_TOOL_CALLS}")
    d.metric("执行耗时", f"{task.elapsed_ms / 1000:.1f} 秒")
    st.caption("耗时不含等待回答；证据数量包含原始记录，不代表全部适用或结论可靠。")

    for event in state.get('trace', []):
        if event.get('event') == 'material_skipped':
            st.warning('材料未采用：' + str(event.get('detail', '解析失败。')))
    report, evidence, material_tab = st.tabs(["研究报告", "证据与核验", "材料与覆盖"])
    with report:
        if task.status == "completed":
            if task.ending == "历史回答":
                st.info("以下内容来自历史档案或项目记忆。")
            elif task.ending == "失败说明":
                st.warning("本次未形成可交付的研究报告，下面是失败说明。")
            elif task.ending == "受限报告":
                st.warning("证据不完整，只交付可支持的部分；受限报告不写入长期记忆。")
            else:
                st.success("报告已通过系统核验；这不等于事实得到独立认证。")
            with st.container(key="report_content"):
                st.markdown(state["final_report"])
            st.download_button("下载 Markdown", state["final_report"],
                               file_name=f"research_{state['run_id']}.md", mime="text/markdown")
        elif task.status in {"failed", "interrupted"}:
            st.error(task.error)
            st.info("已保留可查询的进度与轨迹。不自动重试，避免重复请求；可新建任务。")
        elif task.status == "abandoned":
            st.info("任务已放弃，没有生成报告。")
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
                if record.get("historical_run_id"):
                    st.caption(f"复用自历史任务 {record['historical_run_id']} · 档案条目 #{record['historical_evidence_index']}")
                if record.get('source_type') == 'user_material':
                    st.caption(f"用户提供、未独立核实 · {record.get('locator', '')}")
                st.caption(f"有效时间：{record.get('valid_at') or '未提供／不适用'}")
                st.text(record.get("content", ""))
        if not state.get("evidence"):
            st.caption("没有可展示的正式证据。" if task.status == "completed"
                       else "证据正文在任务完成后提供；目前可查询进度和 Trace。")
        st.download_button("下载原始证据 JSON", task.export_json("evidence"),
                           file_name=f"evidence_{state['run_id']}.json", mime="application/json",
                           disabled=task.status != "completed")
    with material_tab:
        if not state.get('material_files'):
            st.caption('本任务没有上传材料。')
        else:
            st.write('材料用途：', MATERIAL_ROLE_LABELS.get(state.get('material_role'), '正在判定'))
            st.caption('材料由用户提供、未独立核实；含材料的运行不写入长期记忆。')
            for name in state['material_files']:
                st.text(name)
            for item in state.get('materials', []):
                st.write(item['display_name'])
                st.caption(coverage_text([item]))
                with st.expander('查看精确范围 · ' + item['display_name'], expanded=False):
                    st.text('\n'.join(item['sent_ranges']) or '没有送入正文')
            if not state.get('materials'):
                st.caption('解析与覆盖信息在任务完成后提供；解析失败会在上方提示。')
    # 放在结果页底部，不依赖当前选择哪个报告／证据页签。
    st.divider()
    st.subheader("本次研究记录")
    with st.expander("问题与补充记录", expanded=False):
        st.write("研究问题：", state["question"])
        history = state.get("clarification_history", [])
        if not history:
            st.caption("本次没有已完成的追问。" if task.status == "completed"
                       else f"已有 {task.view['clarification_count']} 次补充；详细问答记录在任务完成后提供。")
        for index, item in enumerate(history, 1):
            st.markdown(f"**补充条件 {index}**")
            st.write("追问：", item.get("question", ""))
            st.write("回答：", item.get("answer", ""))
    if state.get('conversation_context'):
        with st.expander('本轮使用的历史背景', expanded=False):
            st.caption('仅作背景，旧报告不作为本轮正式证据；truncated 表示该轮背景已截断。')
            st.json(state['conversation_context'], expanded=False)
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


def abandon_and_reset() -> None:
    task = st.session_state.get("research_task")
    try:
        if task is not None and task.status == "waiting":
            task.client.abandon(task.state["run_id"])
        reset_task()
    except ApiError as error:
        st.session_state["api_error"] = str(error)


@st.fragment(run_every="2s")
def poll_task(task: RemoteResearchTask) -> None:
    try:
        if task.refresh():
            st.rerun()
        st.caption("正在查询服务端进度；关闭页面不会主动取消研究。")
    except ApiError as error:
        st.warning(str(error))


def main() -> None:
    st.set_page_config(page_title="大气研究工作台", page_icon="🌤️", layout="wide")
    st.markdown("""<style>
        .block-container {max-width: 1240px; padding-top: 2.3rem;}
        [data-testid="stMetric"] {border: 1px solid #b7c9dd55; border-radius: 12px; padding: 16px;}
        [data-testid="stSidebar"] {border-right: 1px solid #b7c9dd33;}
        [data-testid="stSidebarUserContent"] {padding: 1rem .8rem;}
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .3rem;}
        .st-key-research_home {
            width: min(100%, 42rem); margin-inline: auto;
            padding-top: clamp(2rem, 13vh, 8rem); gap: .75rem;
        }
        .st-key-research_home .research-home-title,
        .st-key-research_heading .research-task-title {
            font-size: 30px; font-weight: 500; line-height: 1.4;
            white-space: normal; overflow-wrap: anywhere;
        }
        .st-key-report_content h1 {font-size: 1.75rem; line-height: 1.35;}
        .st-key-report_content h2 {font-size: 1.4rem; line-height: 1.4;}
        .st-key-report_content h3 {font-size: 1.15rem; line-height: 1.4;}
        .st-key-research_home .research-home-title {
            text-align: center; margin: 0; padding: 0 0 1rem;
        }
        .st-key-research_home [data-testid="stForm"] {
            border-radius: 1.15rem; padding: 1rem;
        }
        .st-key-research_home [data-testid="stCaptionContainer"] {text-align: center;}
        .block-container:has(.st-key-research_heading) {padding-top: 5rem;}
        .st-key-research_heading .research-task-title {
            padding: 0;
            width: 100%; margin: 0 0 calc(.75rem + 10px); text-align: left;
        }
        .st-key-sidebar_brand .sidebar-brand {
            display: flex; align-items: center; justify-content: center;
            min-height: 2.5rem; padding: .25rem 0;
            font-size: 1.1rem; font-weight: 600; line-height: 1.35;
        }
        .st-key-sidebar_brand .sidebar-brand-frame {padding-bottom: 1rem;}
        .st-key-sidebar_new button {
            display: flex; align-items: center; justify-content: center; margin-top: .75rem;
        }
        .st-key-history_heading {padding-top: 1.15rem; padding-bottom: .25rem;}
        [data-testid="stSidebar"] [class*="st-key-open_"],
        [data-testid="stSidebar"] [class*="st-key-open_"] :is(div, span):has(button[data-testid^="stBaseButton-"]) {
            width: 100% !important; min-width: 0; max-width: 100%;
        }
        [data-testid="stSidebar"] [class*="st-key-open_"] button {
            display: flex; align-items: center; justify-content: flex-start;
            width: 100% !important; min-height: 2rem; padding: .25rem .65rem;
            border: 1px solid transparent; border-radius: .5rem; background: transparent;
        }
        [data-testid="stSidebar"] [class*="st-key-open_"] button > *,
        [data-testid="stSidebar"] [class*="st-key-open_"] button [data-testid="stMarkdownContainer"] {
            flex: 1 1 0; width: 100%; min-width: 0; overflow: hidden; text-align: left;
        }
        [data-testid="stSidebar"] [class*="st-key-open_"] button span:has([data-testid="stMarkdownContainer"]) {
            width: 100%; min-width: 0; justify-content: flex-start;
        }
        [data-testid="stSidebar"] [class*="st-key-open_"] button p {
            display: block; width: 100%; margin: 0;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: left;
        }
        [data-testid="stSidebar"] [class*="st-key-open_"] button:hover {
            background: color-mix(in srgb, currentColor 8%, transparent); color: inherit;
        }
        [data-testid="stSidebar"] [class*="st-key-open_"] button[kind="primary"] {
            background: color-mix(in srgb, currentColor 13%, transparent); color: inherit;
        }
        [data-testid="stSidebar"] [class*="st-key-open_"] button:focus-visible {
            outline: 2px solid currentColor; outline-offset: -2px;
        }
        [class*="st-key-history_row_"] [class*="st-key-open_"] button[data-testid^="stBaseButton-"]::before {
            content: var(--history-status);
            flex: 0 0 2.4rem; box-sizing: border-box; text-align: center;
            font-size: .69rem; font-weight: 500; line-height: 1.2;
            padding: .15rem .1rem; border: 1px solid currentColor; border-radius: .25rem;
            color: var(--history-color); margin-right: .35rem;
        }
        [class*="st-key-history_row_done_"] {--history-status: "完成"; --history-color: var(--st-green-text-color, #239455);}
        [class*="st-key-history_row_limited_"] {--history-status: "受限"; --history-color: var(--st-yellow-text-color, #aa7200);}
        [class*="st-key-history_row_stopped_"] {--history-status: "中止"; --history-color: var(--st-red-text-color, #ff5555);}
        [class*="st-key-history_row_failed_"] {--history-status: "失败"; --history-color: var(--st-red-text-color, #ff5555);}
        [class*="st-key-history_row_running_"] {--history-status: "执行中"; --history-color: var(--st-blue-text-color, #438bdb);}
        [class*="st-key-history_row_waiting_"] {--history-status: "待补充"; --history-color: var(--st-yellow-text-color, #aa7200);}
        [class*="st-key-history_row_"] [data-testid="stHorizontalBlock"] {gap: .45rem; align-items: center;}
        [class*="st-key-history_row_"] [data-testid="stColumn"]:has([class*="st-key-open_"]) {
            flex: 1 1 0; width: auto; min-width: 0;
        }
        [class*="st-key-history_row_"] [data-testid="stColumn"]:has([class*="st-key-remove_"]) {
            flex: 0 0 2.125rem; width: 2.125rem; min-width: 2.125rem;
        }
        [class*="st-key-history_row_"] [class*="st-key-remove_"] button {
            display: flex; align-items: center; justify-content: center;
            width: 2.125rem; min-width: 2.125rem; height: 2.125rem; min-height: 2.125rem;
            padding: 0; border: 0; border-radius: .45rem;
            background: transparent; opacity: .65;
        }
        [class*="st-key-remove_"] button p {font-size: 1.2rem; line-height: 1; margin: 0;}
        [class*="st-key-remove_"] button:enabled:is(:hover, :focus-visible) {
            opacity: 1; color: #ff5f63;
            background: color-mix(in srgb, currentColor 12%, transparent);
        }
        [role="dialog"] .delete-task-copy {display: grid; gap: 1rem;}
        [role="dialog"] .delete-task-question {
            font-weight: 600; line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere;
            padding: .65rem .75rem; border-radius: .5rem;
            background: color-mix(in srgb, currentColor 6%, transparent);
        }
        [role="dialog"] .delete-task-description {
            margin: 0; font-size: .875rem; line-height: 1.55;
            color: color-mix(in srgb, currentColor 76%, transparent);
        }
        .st-key-delete_task_actions {
            border-top: 1px solid color-mix(in srgb, currentColor 15%, transparent);
            padding-top: .75rem; margin-top: .25rem;
        }
        [class*="st-key-open_"] .stTooltipHoverTarget[aria-describedby] {
            anchor-name: --history-tooltip-row;
        }
        @supports (left: anchor(right)) {
          @media (min-width: 640px) {
            body:has([class*="st-key-open_"] .stTooltipHoverTarget[aria-describedby])
            [role="tooltip"]:has([data-testid="stTooltipContent"]) {
                position-anchor: --history-tooltip-row;
                left: calc(anchor(right) + 8px) !important;
                top: clamp(8px, anchor(top), calc(100dvh - 160px)) !important;
                transform: none !important;
            }
          }
        }
        body:has([class*="st-key-open_"] .stTooltipHoverTarget[aria-describedby]) [data-testid="stTooltipContent"] {
            width: min(300px, calc(100vw - 24px));
            max-width: calc(100vw - 24px);
            padding: .65rem .8rem;
            border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
            border-radius: .75rem;
            box-shadow: 0 6px 18px #0002;
            pointer-events: none;
            white-space: normal;
            overflow-wrap: anywhere;
            animation: none;
        }
        body:has([class*="st-key-open_"] .stTooltipHoverTarget[aria-describedby]) [data-testid="stTooltipContent"] p {
            white-space: normal;
            overflow-wrap: anywhere;
        }
        </style>""", unsafe_allow_html=True)
    if "api_client" not in st.session_state:
        try:
            st.session_state["api_client"] = ResearchClient()
        except ApiError as error:
            st.error(str(error))
            return
    client = st.session_state["api_client"]
    service_ok = False
    runs = []
    try:
        client.health()
        history_offset = st.session_state.get("history_offset", 0)
        runs = client.list_runs(offset=history_offset)
        service_ok = True
    except ApiError as error:
        st.warning(str(error))
        st.code(".venv\\Scripts\\python -m atmospheric_research_agent.service", language="powershell")
    task = st.session_state.get("research_task")
    if task is not None and service_ok:
        try:
            task.refresh()
        except ApiError as error:
            st.warning(str(error))
            if error.code == "run_not_found":
                st.session_state.pop("research_task", None)
                task = None
    with st.sidebar:
        with st.container(key="sidebar_brand"):
            st.markdown(
                '<div class="sidebar-brand-frame"><div class="sidebar-brand">🌤️ 大气研究工作台</div></div>',
                unsafe_allow_html=True,
            )
        st.button("＋ 新建研究", type="primary", use_container_width=True,
                  key="sidebar_new",
                  disabled=task is not None and task.status == "running",
                  on_click=request_new_task)
        if st.session_state.get("confirm_new_task"):
            st.warning("确认后将放弃等待中的任务；已用 API 费用不会退回。")
            confirm, cancel = st.columns(2)
            confirm.button("确认新建", on_click=abandon_and_reset, use_container_width=True)
            cancel.button("继续当前任务", on_click=lambda: st.session_state.pop("confirm_new_task", None),
                          use_container_width=True)
        with st.container(key="history_heading"):
            st.caption("历史任务")
        if not runs and service_ok:
            st.caption("暂无研究记录")
        for view in runs:
            question = " ".join(view["question"].split())
            label = re.sub(r"([\\`*_{}\[\]()<>#!|~])", r"\\\1", question)
            status = {
                "completed": ("done", "完成"), "running": ("running", "执行中"),
                "waiting_user": ("waiting", "待补充"), "interrupted": ("stopped", "中止"),
                "suspended": ("stopped", "中止"), "abandoned": ("stopped", "中止"),
                "failed": ("failed", "失败"),
            }.get(view["status"], ("stopped", "中止"))
            if view["status"] == "completed":
                status = {
                    "limited_report": ("limited", "受限"), "failure_notice": ("failed", "失败"),
                }.get(view.get("delivery_type"), status)
            selected = task is not None and task.state["run_id"] == view["run_id"]
            with st.container(key=f"history_row_{status[0]}_{view['run_id']}"):
                item, actions = st.columns([8, 1], vertical_alignment="center")
                opened = item.button(label, key=f"open_{view['run_id']}",
                                     help=f"{label}\n\n状态：{status[1]}", use_container_width=True,
                                     type="primary" if selected else "secondary")
                if actions.button("×", key=f"remove_{view['run_id']}", help="删除任务",
                                  disabled=view["status"] in {"running", "waiting_user"}):
                    st.session_state["pending_delete"] = view
            if opened:
                st.session_state["research_task"] = RemoteResearchTask(client, view)
                st.session_state.pop("confirm_new_task", None)
                st.rerun()
        if pending_delete := st.session_state.get("pending_delete"):
            confirm_delete_task(client, pending_delete, last_on_page=len(runs) == 1)
        if st.session_state.get('history_offset', 0) > 0 or len(runs) == 20:
            previous, following = st.columns(2)
            if previous.button('上一页', disabled=not service_ok or st.session_state.get('history_offset', 0) == 0):
                st.session_state['history_offset'] = max(0, st.session_state.get('history_offset', 0) - 20)
                st.rerun()
            if following.button('下一页', disabled=not service_ok or len(runs) < 20):
                st.session_state['history_offset'] = st.session_state.get('history_offset', 0) + 20
                st.rerun()
        st.divider()
        st.caption("● 服务已连接" if service_ok else "○ 服务未连接")
        with st.expander("服务与预算"):
            st.caption(client.base_url)
            st.caption(f"研究 {MAX_RESEARCH_ROUNDS} 轮 · 工具 {MAX_TOOL_CALLS} 次 · 追问 {MAX_CLARIFICATIONS} 次")
            st.button("刷新服务状态", on_click=lambda: None)
    if st.session_state.get("api_error"):
        st.error(st.session_state.pop("api_error"))
    if task is None:
        with st.container(key="research_home"):
            st.markdown('<h1 class="research-home-title">今天想研究什么？</h1>', unsafe_allow_html=True)
            with st.form("research_question"):
                question = st.text_area("研究问题", height=100, key="question_draft", max_chars=10000,
                                        label_visibility="collapsed", placeholder="例如：为什么副热带高压主体常晴热少雨，边缘也一样吗？")
                with st.expander("添加研究材料（可选）"):
                    generation = st.session_state.get('material_upload_generation', 0)
                    uploaded = st.file_uploader('添加研究材料（可选）', type=['txt', 'md', 'pdf'],
                        accept_multiple_files=True, key=f'material_uploads_{generation}',
                        help=f'最多 {MATERIAL_MAX_FILES} 份，每份不超过 {MATERIAL_UPLOAD_MAX_BYTES // (1024 * 1024)}MB。TXT/MD 请使用 UTF-8；PDF 需有文字层，不做 OCR。')
                    role = st.selectbox('材料用途', options=list(MATERIAL_ROLE_LABELS),
                        format_func=MATERIAL_ROLE_LABELS.get, key='material_role_choice')
                submitted = st.form_submit_button("开始研究", type="primary", disabled=not service_ok)
            st.caption("研究会调用已配置的模型和工具，可能产生 API 费用。上传材料保存在本地。")
        if submitted:
            if not question.strip():
                st.warning("请填写研究问题。")
            elif len(uploaded) > MATERIAL_MAX_FILES:
                st.warning(f'每个任务最多上传 {MATERIAL_MAX_FILES} 份材料。')
            elif any(item.size == 0 or item.size > MATERIAL_UPLOAD_MAX_BYTES for item in uploaded):
                st.warning('材料不能为空，且每份不能超过上传大小上限。')
            else:
                files = [(item.name, item.getvalue()) for item in uploaded]
                signature = (role, tuple((name, sha256(content).hexdigest()) for name, content in files)) if files else None
                pending = st.session_state.get("create_request")
                if not pending or pending["question"] != question.strip() or pending.get('material_signature') != signature:
                    pending = {"question":question.strip(), "key":str(uuid4()), 'material_signature': signature}
                    st.session_state["create_request"] = pending
                try:
                    options = {'materials': files, 'material_role': role} if files else {}
                    view = client.create(pending["question"], pending["key"], **options)
                    st.session_state["research_task"] = RemoteResearchTask(client, view)
                    st.rerun()
                except ApiError as error:
                    st.error(str(error))
        return
    with st.container(key="research_heading"):
        title_question = " ".join(task.state["question"].split())
        heading = escape(f"当前问题：{title_question}")
        st.markdown(f'<h1 class="research-task-title" title="{heading}">{heading}</h1>',
                    unsafe_allow_html=True)
    if task.view.get('material_files'):
        st.write('本任务材料：', '、'.join(task.view['material_files']))
        st.caption('追问后继续使用同一批材料；如需更换材料，请新建研究。')
    if task.status in {"interrupted", "suspended"}:
        can_resume = task.view.get("can_resume", False)
        with st.container(key="resume_research"):
            st.markdown("""<style>
                .st-key-resume_research button[kind="primary"]:enabled {
                    background-color: #15803d; border-color: #15803d; color: white;
                }
                </style>""", unsafe_allow_html=True)
            action = st.empty()
            if action.button("继续研究", type="primary", disabled=not service_ok or not can_resume):
                action.button("正在继续研究…", type="primary", disabled=True)
                try:
                    with st.spinner("正在读取执行检查点…"):
                        view = client.resume(task.state["run_id"])
                    st.session_state["research_task"] = RemoteResearchTask(client, view)
                    st.rerun()
                except ApiError as error:
                    st.error(str(error))
                    action.empty()
                    st.button("刷新任务状态", key="refresh_resume")
            if not can_resume:
                st.caption("这项任务没有可用的执行检查点，可以查看历史记录。")
    if task.status == "running":
        st.info("服务端正在研究。")
        render_progress(task.state, st.empty(), st.empty())
        poll_task(task)
    if task.status == "waiting":
        st.warning("研究已暂停，等待你补充信息。已有证据和已用预算保留。")
        st.write(task.question)
        question_id = task.view["clarification"]["question_id"]
        with st.form(f"clarify_{question_id}"):
            answer = st.text_area("你的补充", key=f"clarification_answer_{question_id}", max_chars=10000,
                                  placeholder="请回答上面的问题，例如地点、日期或研究对象。")
            resume = st.form_submit_button("提交回答，继续原任务", type="primary", disabled=not service_ok)
        if resume:
            if not answer.strip():
                st.warning("回答不能为空；原任务仍在等待。")
            else:
                try:
                    task.answer(answer)
                    st.rerun()
                except ApiError as error:
                    st.error(str(error))
    if task.status == "abandoned":
        st.info("该任务已放弃，可新建研究。")
    render_results(task)
    if task.status == 'completed':
        st.divider()
        st.subheader('继续提问')
        st.caption('复用本会话材料与有限历史背景；每次提问重新取证、单独计算预算，可能产生 API 费用。')
        with st.form(f"follow_up_{task.state['run_id']}"):
            follow_question = st.text_area('接下来想研究什么？', max_chars=10000)
            follow_role = st.selectbox('本轮材料用途', options=list(MATERIAL_ROLE_LABELS),
                                      format_func=MATERIAL_ROLE_LABELS.get)
            follow_submit = st.form_submit_button('继续研究', disabled=not service_ok)
        if follow_submit:
            if not follow_question.strip():
                st.warning('请填写下一轮研究问题。')
            else:
                signature = (task.state['run_id'], follow_question.strip(), follow_role)
                pending = st.session_state.get('follow_up_request')
                if not pending or pending['signature'] != signature:
                    pending = {'signature': signature, 'key': str(uuid4())}
                    st.session_state['follow_up_request'] = pending
                try:
                    view = client.follow_up(task.state['run_id'], follow_question.strip(), pending['key'], follow_role)
                    st.session_state['research_task'] = RemoteResearchTask(client, view)
                    st.rerun()
                except ApiError as error:
                    st.error(str(error))
        st.caption('会话编号：' + task.view.get('conversation_id', task.state['run_id']))


if __name__ == "__main__":
    main()
