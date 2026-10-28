"""覆盖摘要不改底层定位；比较口径规则为提示词契约，非真实模型效果证明。"""
from copy import deepcopy

import pytest

from atmospheric_research_agent.agent import materials, nodes
from atmospheric_research_agent.agent.state import create_initial_state


def coverage_record(**kwargs):
    return {'display_name': '原创.txt', 'parsed_characters': 20060, 'sent_characters': 20000,
            'parsed_pages': 0, 'document_pages': 0, 'sent_pages': [], 'parse_note': '',
            'truncated': True,
            'sent_ranges': [f'第 {i} 段，字符 1–32' for i in range(1, 354)], **kwargs}


def test_long_text_coverage_is_compact_and_preserves_all_ranges():
    record = coverage_record()
    before = deepcopy(record)
    summary = materials.coverage_text([record])
    assert len(summary) < 200 and '\n' not in summary
    assert '已解析 20060 字符' in summary and '送入模型 20000 字符' in summary
    assert '涉及第 1–353 段' in summary and '已截断' in summary
    assert '0 页' not in summary and '[]' not in summary and '字符 1–32' not in summary
    assert record == before and len(record['sent_ranges']) == 353


def test_pdf_coverage_does_not_claim_missing_pages_were_sent():
    record = coverage_record(display_name='原创.pdf', parsed_pages=5, document_pages=12,
                             sent_pages=[1, 3, 4], parse_note='仅读取了前 5 页（共 12 页），已截断。')
    summary = materials.coverage_text([record])
    assert '已解析 5 页／20060 字符' in summary
    assert '涉及第 1、3–4 页' in summary and '第 1–4 页' not in summary
    assert '共 12 页' in summary and '[1, 3, 4]' not in summary


def test_zero_delivery_and_split_paragraph_have_honest_scope():
    zero = coverage_record(sent_characters=0, sent_ranges=[])
    assert '未送入正文' in materials.coverage_text([zero])
    assert '[]' not in materials.coverage_text([zero])
    split = coverage_record(sent_ranges=['第 1 段，字符 1–12000', '第 1 段，字符 12001–20000'])
    summary = materials.coverage_text([split])
    assert '涉及第 1 段' in summary and '完整' not in summary


def test_unknown_legacy_locator_is_not_fabricated_or_dumped():
    record = coverage_record(sent_ranges=['旧版精确定位'] * 353)
    summary = materials.coverage_text([record])
    assert '精确定位见材料记录' in summary and len(summary) < 200


@pytest.mark.parametrize('builder', ['assessment', 'draft', 'review', 'revision'])
def test_scope_rule_reaches_all_evidence_comparison_stages(builder):
    state = create_initial_state('核对两个来源的数据，不同时段能否比较？')
    state.update(evidence=[{'evidence_id': 'one', 'source_type': 'web',
        'content': '目标时段11日11时至12日2时；现有来源只覆盖10日17时至11日6时。'}],
        draft_report='两者量级不符，但时段不同不能直接比较。[E1]',
        report_review={'verdict': 'revise', 'required_changes': ['统一口径表述。']})
    before = deepcopy(state)
    builders = {'assessment': nodes._assessment_messages,
                'draft': lambda s: nodes._report_messages(s, 'limited'),
                'review': nodes._verification_messages, 'revision': nodes._revision_messages}
    messages = builders[builder](state)
    rules = '\n'.join(m.content for m in messages if m.type == 'system')
    assert '先核对统计时段、地点与空间尺度' in rules
    assert '未建立可比关系时，不得仅凭数值大小认定' in rules
    assert '现有来源未覆盖目标时段或口径，无法直接比较或核实' in rules
    assert '明确的统计包含或累计关系' in rules
    assert state == before
    if builder == 'review':
        assert '必须写入 required_changes，不降为措辞建议' in rules
        assert any('量级不符，但时段不同不能直接比较' in m.content for m in messages)


def test_streamlit_shows_summary_and_collapses_precise_ranges():
    from asgi_helpers import require_loopback
    require_loopback()
    AppTest = pytest.importorskip('streamlit.testing.v1').AppTest
    script = '''
from types import SimpleNamespace
from streamlit_app import render_results
record = {'display_name':'原创.txt','parsed_characters':20060,'sent_characters':20000,
    'parsed_pages':0,'document_pages':0,'sent_pages':[],'truncated':True,'parse_note':'',
    'sent_ranges':[f'第 {i} 段，字符 1–32' for i in range(1,354)]}
task = SimpleNamespace(status='completed', ending='完整报告', elapsed_ms=1,
    view={'clarification_count':0}, export_json=lambda _: '{}',
    state={'run_id':'ui-offline','question':'总结材料','final_report':'原文分析',
           'material_files':['原创.txt'],'material_role':'analysis','materials':[record]})
render_results(task)
'''
    app = AppTest.from_string(script, default_timeout=10).run()
    assert not app.exception
    summary = next(c.value for c in app.caption if '材料覆盖：' in c.value)
    assert '第 1–353 段' in summary and '0 页' not in summary and '[]' not in summary
    detail = next(e for e in app.expander if e.label == '查看精确范围 · 原创.txt')
    assert detail.proto.expanded is False
    assert any('第 353 段，字符 1–32' in t.value for t in detail.text)
