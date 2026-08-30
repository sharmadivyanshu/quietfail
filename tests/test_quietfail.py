"""Tests for the detection layer itself."""

import pytest

from quietfail import Budget, BudgetExceeded, CircuitBreaker, CircuitOpen, Store
from quietfail.baseline import MIN_WINDOW_RUNS, build_profile, evaluate_run, evaluate_window
from quietfail.report import render_html


def make_run(**overrides) -> dict:
    base = {
        "run_id": "r1",
        "agent": "a",
        "status": "ok",
        "outcome": "posted",
        "node_path": ["ingest", "validate", "post"],
        "output_keys": ["extracted", "outcome"],
        "step_count": 3,
        "duration_ms": 10,
        "tokens": 0,
        "usd": 0.0,
        "tool_calls": 0,
        "started_at": "2026-01-01T00:00:00+00:00",
        "tool_events": [],
    }
    base.update(overrides)
    base["step_count"] = len(base["node_path"])
    return base


def healthy_runs(n: int = 20) -> list[dict]:
    runs = []
    for i in range(n):
        outcome = "posted" if i % 5 == 0 else "awaiting_human"
        path = (
            ["ingest", "validate", "post"]
            if outcome == "posted"
            else ["ingest", "validate", "resolve", "human_review"]
        )
        runs.append(
            make_run(
                run_id=f"r{i}",
                outcome=outcome,
                node_path=path,
                output_keys=["extracted", "outcome"],
                tool_events=[{"tool": "lookup", "empty": 0, "errored": 0, "latency_ms": 1.0}],
            )
        )
    return runs


@pytest.fixture
def profile() -> dict:
    return build_profile(healthy_runs())


# --- baseline ---------------------------------------------------------------


def test_profile_captures_paths_and_outcomes(profile):
    assert len(profile["node_paths"]) == 2
    assert set(profile["outcomes"]) == {"posted", "awaiting_human"}
    assert profile["outcome_rates"]["posted"] == pytest.approx(0.2)


def test_empty_baseline_is_refused():
    with pytest.raises(ValueError):
        build_profile([])


# --- per-run signals --------------------------------------------------------


def test_healthy_run_raises_nothing(profile):
    assert evaluate_run(healthy_runs(1)[0], profile) == []


def test_unseen_path_is_flagged(profile):
    run = make_run(node_path=["ingest", "validate", "sneaky", "post"])
    signals = {a.signal for a in evaluate_run(run, profile)}
    assert "trajectory.unseen_path" in signals


def test_skipping_a_universal_node_is_critical(profile):
    """`validate` appears on every baseline path — skipping it is the
    canonical silent failure."""
    run = make_run(node_path=["ingest", "post"])
    alerts = [a for a in evaluate_run(run, profile) if a.signal == "trajectory.skipped_node"]
    assert alerts and alerts[0].severity == "critical"


def test_missing_output_key_is_flagged(profile):
    run = make_run(output_keys=["extracted"])
    signals = {a.signal for a in evaluate_run(run, profile)}
    assert "shape.missing_key" in signals


def test_run_without_outcome_is_flagged(profile):
    run = make_run(outcome=None)
    signals = {a.signal for a in evaluate_run(run, profile)}
    assert "outcome.missing" in signals


def test_step_explosion_is_flagged(profile):
    run = make_run(node_path=["ingest"] + ["loop"] * 40)
    signals = {a.signal for a in evaluate_run(run, profile)}
    assert "trajectory.step_explosion" in signals


# --- aggregate signals ------------------------------------------------------


def test_small_windows_are_never_evaluated(profile):
    """The bug that made every first-run-after-deploy alert."""
    window = [make_run(outcome="posted")] * (MIN_WINDOW_RUNS - 1)
    assert evaluate_window(window, profile) == []


def test_healthy_window_is_quiet(profile):
    assert evaluate_window(healthy_runs(20), profile) == []


def test_tool_empty_spike_is_flagged(profile):
    window = [
        make_run(
            run_id=f"x{i}",
            tool_events=[{"tool": "lookup", "empty": 1, "errored": 0, "latency_ms": 1.0}],
        )
        for i in range(MIN_WINDOW_RUNS)
    ]
    signals = {a.signal for a in evaluate_window(window, profile)}
    assert "tool.empty_rate_spike" in signals


def test_escalation_collapse_is_flagged(profile):
    """Every run posts, none escalate — the gate stopped firing."""
    window = [make_run(run_id=f"y{i}", outcome="posted") for i in range(MIN_WINDOW_RUNS)]
    signals = {a.signal for a in evaluate_window(window, profile)}
    assert "outcome.rate_collapse" in signals


# --- budgets ----------------------------------------------------------------


def test_budget_raises_on_breach():
    budget = Budget(max_usd=1.0)
    budget.charge_tokens(1000, usd=0.5)
    with pytest.raises(BudgetExceeded):
        budget.charge_tokens(1000, usd=0.75)


def test_budget_caps_steps():
    budget = Budget(max_steps=2)
    budget.charge_step()
    budget.charge_step()
    with pytest.raises(BudgetExceeded):
        budget.charge_step()


def test_circuit_breaker_trips_on_repeated_failure():
    breaker = CircuitBreaker(threshold=3)
    for _ in range(2):
        breaker.record("api", failed=True)
    with pytest.raises(CircuitOpen):
        breaker.record("api", failed=True)


def test_circuit_breaker_resets_on_success():
    breaker = CircuitBreaker(threshold=2)
    breaker.record("api", failed=True)
    breaker.record("api", failed=False)
    breaker.record("api", failed=True)  # would trip if the reset were missing


# --- store and report -------------------------------------------------------


def test_store_roundtrip(tmp_path):
    from quietfail.store import RunRecord, ToolEvent

    store = Store(tmp_path / "t.sqlite")
    record = RunRecord(agent="a", node_path=["ingest", "post"], outcome="posted")
    record.tool_events.append(ToolEvent("lookup", empty=False, errored=False, latency_ms=2.0))
    store.save_run(record)

    runs = store.runs("a")
    assert len(runs) == 1
    assert runs[0]["node_path"] == ["ingest", "post"]
    assert runs[0]["tool_events"][0]["tool"] == "lookup"


def test_baseline_roundtrip_stamps_built_at(tmp_path, profile):
    store = Store(tmp_path / "t.sqlite")
    store.save_baseline("a", 20, profile)
    assert "_built_at" in store.load_baseline("a")


def test_report_renders_without_a_baseline(tmp_path):
    html = render_html("a", healthy_runs(3), [], None)
    assert "quietfail" in html and "<table" in html
