from atmospheric_research_agent.agent.state import create_initial_state


def test_create_initial_state_sets_explicit_defaults() -> None:
    state = create_initial_state("  什么条件有利于台风形成？  ")

    assert state["question"] == "什么条件有利于台风形成？"
    assert state["messages"] == []
    assert state["evidence"] == []
    assert state["research_round"] == 0
    assert state["tool_call_count"] == 0
    assert state["clarification_question"] is None
    assert state["final_report"] is None
    assert state["trace"] == []
