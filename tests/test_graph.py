"""Graph-level behaviour: routing, the resolution cycle, and HITL."""

from invoice_agent.graph import build_graph
from invoice_agent.nodes import MAX_RESOLUTION_ROUNDS
from langgraph.types import Command


def run(doc_id: str):
    graph = build_graph()
    config = {"configurable": {"thread_id": doc_id}}
    result = graph.invoke({"document_uri": doc_id, "tenant_id": "test"}, config=config)
    return graph, config, result


def test_clean_invoice_posts_without_human():
    graph, config, result = run("clean-001")
    assert result["outcome"] == "posted"
    assert graph.get_state(config).next == ()


def test_arithmetic_mismatch_is_never_auto_resolved():
    """Broken arithmetic must always reach a human. If this test ever goes
    green by auto-posting, the agent has started inventing numbers."""
    graph, config, _ = run("totals-broken-004")
    assert graph.get_state(config).next == ("human_review",)


def test_duplicate_is_never_auto_resolved():
    graph, config, _ = run("duplicate-005")
    assert graph.get_state(config).next == ("human_review",)


def test_vendor_resolution_cycles_back_through_validation():
    """The invoice has two errors. Round 1 fixes the vendor, re-validation
    then surfaces the GL gap — proving the cycle works and that a partially
    fixed invoice never posts."""
    _, _, result = run("unknown-vendor-003")
    strategies = [r.strategy for r in result["resolutions"]]
    assert "vendor_lookup" in strategies
    assert result["iteration"] >= 1


def test_human_rejection_escalates():
    graph, config, _ = run("totals-broken-004")
    result = graph.invoke(Command(resume={"approved": False}), config=config)
    assert result["outcome"] == "escalated"


def test_human_approval_posts():
    graph, config, _ = run("totals-broken-004")
    result = graph.invoke(Command(resume={"approved": True}), config=config)
    assert result["outcome"] == "posted"


def test_interrupt_payload_explains_itself():
    """Anti-approval-fatigue contract: the payload must say why it stopped."""
    graph, config, _ = run("totals-broken-004")
    payload = graph.get_state(config).tasks[0].interrupts[0].value
    assert payload["why_escalated"] == "arithmetic_mismatch"
    assert payload["findings"]
    assert "confidence" in payload


def test_resolution_rounds_are_bounded():
    _, _, result = run("unknown-vendor-003")
    assert result["iteration"] <= MAX_RESOLUTION_ROUNDS


def test_state_survives_process_boundary():
    """Checkpointer contract: a fresh handle on the same thread_id resumes."""
    graph, config, _ = run("missing-gl-002")
    assert graph.get_state(config).next == ("human_review",)
    resumed = graph.invoke(Command(resume={"approved": True}), config=config)
    assert resumed["outcome"] == "posted"
