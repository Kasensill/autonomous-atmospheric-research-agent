"""启动：python -m streamlit run streamlit_app.py --server.address 127.0.0.1"""

from uuid import uuid4
from hashlib import sha256

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
    if task is not None and task.status not in {"completed", "failed", "abandoned"}:
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
        runs = client.list_runs()
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
        st.markdown("### 🌤️ 大气研究工作台")
        st.caption("证据驱动 · 有限研究 · 可追溯交付")
        st.button("＋ 新建研究", type="primary", use_container_width=True,
                  disabled=task is not None and task.status == "running",
                  on_click=request_new_task)
        if st.session_state.get("confirm_new_task"):
            st.warning("确认后将放弃等待中的任务；已用 API 费用不会退回。")
            confirm, cancel = st.columns(2)
            confirm.button("确认新建", on_click=abandon_and_reset, use_container_width=True)
            cancel.button("继续当前任务", on_click=lambda: st.session_state.pop("confirm_new_task", None),
                          use_container_width=True)
        st.divider()
        st.caption("研究服务：" + ("已连接" if service_ok else "未连接"))
        st.caption(client.base_url)
        st.button("刷新服务状态", on_click=lambda: None)
        st.markdown("**服务端任务**")
        for view in runs:
            label = " ".join(view["question"].split())[:25]
            if st.button(f"{view['status']} · {label}", key=f"open_{view['run_id']}"):
                st.session_state["research_task"] = RemoteResearchTask(client, view)
                st.session_state.pop("confirm_new_task", None)
                st.rerun()
        st.divider()
        st.markdown("**程序预算**")
        st.caption(f"研究 {MAX_RESEARCH_ROUNDS} 轮 · 工具 {MAX_TOOL_CALLS} 次 · 追问 {MAX_CLARIFICATIONS} 次")
        st.caption("由现有研究图控制，界面不修改预算。凭据在服务端本地 .env 配置。")
        st.caption("本地单人服务，一次一个未结束任务。页面可重新选择任务；服务重启后内存任务丢失。仅保留有限历史，不支持运行中强制取消。")
    if st.session_state.get("api_error"):
        st.error(st.session_state.pop("api_error"))
    if task is None:
        st.title("把问题，交给证据。")
    else:
        title_question = " ".join(task.state["question"].split())
        title_question = title_question[:90] + "…" if len(title_question) > 90 else title_question
        st.title(f"当前问题：{title_question}")
    st.write("提出大气科学问题，查看取证过程，获得带来源的研究报告。")
    st.caption("缺少关键条件时暂停追问，证据不足时受限交付或说明失败。")
    if task is None:
        with st.form("research_question"):
            question = st.text_area("研究问题", height=135, key="question_draft", max_chars=10000,
                                    placeholder="例如：为什么副热带高压主体常晴热少雨，边缘也一样吗？")
            generation = st.session_state.get('material_upload_generation', 0)
            uploaded = st.file_uploader('添加研究材料（可选）', type=['txt', 'md', 'pdf'],
                accept_multiple_files=True, key=f'material_uploads_{generation}',
                help=f'最多 {MATERIAL_MAX_FILES} 份，每份不超过 {MATERIAL_UPLOAD_MAX_BYTES // (1024 * 1024)}MB。TXT/MD 请使用 UTF-8；PDF 需有文字层，不做 OCR。')
            role = st.selectbox('材料用途', options=list(MATERIAL_ROLE_LABELS),
                format_func=MATERIAL_ROLE_LABELS.get, key='material_role_choice')
            submitted = st.form_submit_button("开始研究", type="primary", disabled=not service_ok)
            st.caption("开始后服务端会调用已配置的模型和工具，可能产生 API 费用。材料保存在服务端本地，不入仓库；首版不自动清理文件，仅限本地可信使用。")
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
        st.divider()
        st.caption("先取证 → 评估 → 起草 → 核验；报告、证据和运行记录在同一任务内查看。")
        return
    st.text_area("研究问题", value=task.state["question"], height=135,
                 key=f"active_question_{task.state['run_id']}", disabled=True)
    if task.view.get('material_files'):
        st.write('本任务材料：', '、'.join(task.view['material_files']))
        st.caption('追问后继续使用同一批材料；如需更换材料，请新建研究。')
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



if __name__ == "__main__":
    main()
