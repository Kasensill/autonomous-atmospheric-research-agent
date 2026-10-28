"""固定语义样例的输入契约；具体模型判断由本地真实核验对照另行记录。"""
from atmospheric_research_agent.agent import nodes


def test_comparison_rule_preserves_real_subset_arithmetic_conflicts():
    rule = nodes._comparison_scope_rule(review=True)
    assert '未建立可比关系时' in rule
    assert '若存在明确的统计包含或累计关系，可以写出关系后推导' in rule
    assert '不能凭口径不同掩盖真正的矛盾' in rule
