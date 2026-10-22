"""事件取证、追问边界及来源冲突提示的纯输入构造回归，不调用模型。"""

import pytest

from atmospheric_research_agent.agent import nodes
from atmospheric_research_agent.agent.state import create_initial_state
from atmospheric_research_agent.agent.tools.clarification import ASK_USER_TOOL


CLARIFICATION_BOUNDARY = (
    "只能询问用户才能提供的研究条件（地点、时间、对象、关心的指标）；"
    "不得询问使用哪种资料来源、是否仅用内部知识库或是否联网检索，这些由你自行决定。"
)
SOURCE_NATURE_RULE = (
    "引用证据时，在正文中说明其性质：预报、实况或事后分析、一般机制背景。"
    "预报不得写成实况；邻近地区的资料不得写成目标地点的情况。"
)
CONFLICT_RULE = (
    "同一事实（时间、数值、名称）在不同证据中不一致时，必须在报告中写明分歧并标出各自的 [E#]，"
    "不得择一不提，也不得自行断定哪个正确；无法判断时列入局限性。"
    "例如一条证据称某极值出现在 17 时，另一条称 4 至 5 点。"
)


def prompt_state():
    state = create_initial_state("分析指定时间、地点的天气事件")
    state.update(
        evidence=[{"evidence_id": "one", "content": "一般机制背景", "source_location": "theory.md"}],
        draft_report="待修订草稿 [E1]。",
        report_review={"verdict": "revise", "required_changes": ["说明来源性质"]},
    )
    return state


def test_agent_event_fact_rules_follow_unchanged_current_weather_rules():
    prompt = nodes.AGENT_SYSTEM_PROMPT
    weather_rules = (
        '- 如果问题要求查询“当前天气”而没有明确地点且还有澄清额度，必须先调用 ask_user 询问地点；\n'
        '  不得先调用本地知识库或其他工具来替代该地点信息。\n'
        '- 用户要求简要当前天气分析时，先取得天气，再按实际要素定向补充必要机制；\n'
        '  已有资料足以回答就停止，不为扩大背景而穷举高压、寒潮、锋面等系统。'
    )
    assert weather_rules in prompt
    event_rule = "当问题指向特定日期和地点的具体天气事件（过去的或正在发生的）时，先核实事件事实，再用本地知识库解释机制"
    assert prompt.index(event_rule) > prompt.index(weather_rules)
    for phrase in (
        "事件事实（时间、地点、实测量级、影响范围）优先取自实况观测、官方调查或发布、同行评审论文或权威媒体",
        "预报（事前发布）、实况或事后分析、一般机制背景。预报不得当作实况使用",
        "本地知识库只用于解释机制，不能作为事件事实的依据",
        "只找到预报，或只找到邻近地区的资料（例如问济宁却只有济南的解读）时，如实说明，不得写成目标地点的实况",
    ):
        assert phrase in prompt


@pytest.mark.parametrize("location", ["tool", "agent", "assessment"])
def test_all_clarification_entrances_exclude_source_selection(location):
    prompts = {
        "tool": ASK_USER_TOOL["function"]["description"],
        "agent": nodes.AGENT_SYSTEM_PROMPT,
        "assessment": nodes._assessment_messages(prompt_state())[0].content,
    }
    assert CLARIFICATION_BOUNDARY in prompts[location]


def test_web_source_selection_is_agent_decision_without_removing_existing_search_rule():
    prompt = nodes.AGENT_SYSTEM_PROMPT
    assert "用户要求网页资料时，必须先调用 search_web，再对选定候选调用 read_source" in prompt
    assert "是否使用网页资料由你自行判断，不是需要询问用户的事项。" in prompt


def test_assessment_requires_event_facts_not_only_mechanisms_or_forecasts():
    prompt = nodes._assessment_messages(prompt_state())[0].content
    for phrase in (
        "对指向具体日期和地点事件的问题",
        "只有一般机制背景、缺少事件事实（时间、地点、实测或官方资料），sufficient 应为 false",
        '在 missing_points 里写明"缺少事件事实"',
        "不得因机制背景充足而判为充分",
        "证据只有预报而没有实况时，同样在 missing_points 里说明",
    ):
        assert phrase in prompt


@pytest.mark.parametrize("stage", ["full", "limited", "revision"])
def test_writing_and_revision_preserve_source_nature_and_disclose_conflicts(stage):
    state = prompt_state()
    messages = nodes._revision_messages(state) if stage == "revision" else nodes._report_messages(state, stage)
    prompt = messages[0].content
    assert SOURCE_NATURE_RULE in prompt
    assert CONFLICT_RULE in prompt
    # 新增写作和修订规则不绑定某个城市或专属天气案例。
    assert "济宁" not in prompt and "济南" not in prompt and "西安" not in prompt
    if stage == "revision":
        assert '只执行 required_changes；unsupported_or_overstated_claims 中以"[建议]"开头的项不要修改正文。' in prompt
        assert "先在全文定位所有位置，一并改成一致" in prompt
        assert "Open-Meteo 只能称“模型当前条件”，不称实况或站点实测。" in prompt
        assert "缺少气候基准不写偏冷/偏暖" in prompt
    else:
        assert "Open-Meteo 模型当前条件只能称为“模型当前条件”，不能称“实况”“观测事实”或“站点实测”。" in prompt
        assert "没有同期气候基准或明确比较对象时，不写“偏冷”“偏暖”等比较结论。" in prompt


@pytest.mark.parametrize("stage", ["full", "limited", "revision", "verification"])
def test_user_statements_are_not_evidence_in_report_prompts(stage):
    state = prompt_state()
    if stage == "revision":
        messages = nodes._revision_messages(state)
    elif stage == "verification":
        messages = nodes._verification_messages(state)
    else:
        messages = nodes._report_messages(state, stage)
    prompt = messages[0].content
    for phrase in (
        "用户补充记录是“用户陈述”，不是证据",
        "可作为问题的已知条件，但不得标注 E 编号",
        "不得写成已核实的事实",
        "不得与证据并列称为“证据间分歧”",
        "与证据不一致时写“用户陈述与证据不一致，无法核实”，并列入局限性",
    ):
        assert phrase in prompt
    if stage == "verification":
        assert "把用户陈述与证据的差异归为“建议”" in prompt
        assert "以“[建议]”开头写入 unsupported_or_overstated_claims，不要作为必须修改项" in prompt


def test_agent_prioritizes_reading_event_facts_over_general_mechanisms():
    prompt = nodes.AGENT_SYSTEM_PROMPT
    for phrase in (
        "search_web 只产生候选来源，不产生证据，取得网页证据至少需要搜索和读取两个步骤",
        "事件事实（尤其实况）尚未取得时，不要把剩余研究轮次用于一般机制检索",
        "先读取其他候选来源或换检索词寻找实况",
        "确认确实找不到实况后，才转向本地知识库解释机制",
        "读取候选时优先官方气象机构发布、实况通报、同行评审论文",
        "只有导航或栏目内容的页面应视为无效，换一个来源",
    ):
        assert phrase in prompt
    assert prompt.index("search_web 只产生候选来源") > prompt.index("只找到邻近地区的资料")


def test_assessment_reason_does_not_explain_program_budgets():
    prompt = nodes._assessment_messages(prompt_state())[0].content
    assert "reason 里不得描述预算余量或剩余额度" in prompt
    assert "研究轮次、工具调用、澄清次数均由程序记录，以程序给出的数字为准" in prompt
    assert "需要继续检索时，只在 missing_points 里写明还缺什么" in prompt
