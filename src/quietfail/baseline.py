"""Baseline profiles and drift evaluation.

Deliberately statistical, not ML: frequency tables, percentile bands, and
observed-path sets. Every alert must be explainable in one sentence, because
an alert nobody understands is an alert nobody acts on.
"""

from __future__ import annotations

from collections import Counter

from .drift import Embedder, build_content_profile
from .store import Alert

# A key present in at least this share of baseline runs is treated as expected.
EXPECTED_KEY_RATE = 0.95
# Tool empty/error rates above baseline + this margin are worth a shout.
TOOL_RATE_MARGIN = 0.20
# Aggregate signals need a real sample. Below this, any distribution looks
# "shifted" purely by chance — evaluating a window of one run made the very
# first run after a deploy alert every time. False positives are how a
# monitor gets muted, and a muted monitor is worse than none.
MIN_WINDOW_RUNS = 10


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round((pct / 100.0) * (len(ordered) - 1))), len(ordered) - 1)
    return float(ordered[index])


def build_profile(runs: list[dict], embedder: Embedder | None = None) -> dict:
    """Summarise what normal looks like for this agent."""
    if not runs:
        raise ValueError("cannot build a baseline from zero runs")

    key_counter: Counter[str] = Counter()
    for run in runs:
        key_counter.update(set(run["output_keys"]))

    tool_totals: Counter[str] = Counter()
    tool_empty: Counter[str] = Counter()
    tool_error: Counter[str] = Counter()
    for run in runs:
        for event in run["tool_events"]:
            tool_totals[event["tool"]] += 1
            tool_empty[event["tool"]] += int(event["empty"])
            tool_error[event["tool"]] += int(event["errored"])

    step_counts = [float(r["step_count"]) for r in runs]
    n = len(runs)
    content = build_content_profile([r.get("output_text") or "" for r in runs], embedder)

    return {
        "content": content,
        "run_count": n,
        "node_paths": sorted({" > ".join(r["node_path"]) for r in runs}),
        "nodes_seen": sorted({node for r in runs for node in r["node_path"]}),
        "outcomes": dict(Counter(r["outcome"] for r in runs if r["outcome"])),
        "outcome_rates": {
            outcome: count / n
            for outcome, count in Counter(r["outcome"] for r in runs if r["outcome"]).items()
        },
        "output_key_rates": {key: count / n for key, count in key_counter.items()},
        "step_count_p50": _percentile(step_counts, 50),
        "step_count_p99": _percentile(step_counts, 99),
        "tool_rates": {
            tool: {
                "calls": tool_totals[tool],
                "empty_rate": tool_empty[tool] / tool_totals[tool],
                "error_rate": tool_error[tool] / tool_totals[tool],
            }
            for tool in tool_totals
        },
    }


def evaluate_run(run: dict, profile: dict) -> list[Alert]:
    """Signals 1, 2 and 4 — shape, trajectory, terminal state. One run at a time."""
    alerts: list[Alert] = []
    run_id = run["run_id"]

    # --- signal 2: trajectory ---------------------------------------------
    path = " > ".join(run["node_path"])
    if path not in profile["node_paths"]:
        unseen = [n for n in run["node_path"] if n not in profile["nodes_seen"]]
        detail = f"path: {path}"
        if unseen:
            detail += f" (contains never-before-seen node(s): {unseen})"
        alerts.append(
            Alert(
                signal="trajectory.unseen_path",
                severity="critical" if unseen else "warn",
                summary="agent took a route it has never taken before",
                detail=detail,
                scope="run",
                run_id=run_id,
            )
        )

    missing_nodes = [
        node
        for node in profile["nodes_seen"]
        if node in _always_visited(profile) and node not in run["node_path"]
    ]
    if missing_nodes:
        alerts.append(
            Alert(
                signal="trajectory.skipped_node",
                severity="critical",
                summary=f"skipped node(s) that run on every baseline path: {missing_nodes}",
                detail=f"path: {path}",
                scope="run",
                run_id=run_id,
            )
        )

    if run["step_count"] > profile["step_count_p99"] * 1.5:
        alerts.append(
            Alert(
                signal="trajectory.step_explosion",
                severity="warn",
                summary=f"{run['step_count']} steps vs baseline p99 of "
                f"{profile['step_count_p99']:.0f}",
                detail="possible loop or replanning spiral",
                scope="run",
                run_id=run_id,
            )
        )

    # --- signal 1: output shape -------------------------------------------
    present = set(run["output_keys"])
    for key, rate in profile["output_key_rates"].items():
        if rate >= EXPECTED_KEY_RATE and key not in present:
            alerts.append(
                Alert(
                    signal="shape.missing_key",
                    severity="critical",
                    summary=f"output missing {key!r}, present in {rate:.0%} of baseline runs",
                    detail=f"keys present: {sorted(present)}",
                    scope="run",
                    run_id=run_id,
                )
            )

    # --- signal 4: terminal state -----------------------------------------
    outcome = run["outcome"]
    if outcome and outcome not in profile["outcomes"]:
        alerts.append(
            Alert(
                signal="outcome.unseen",
                severity="warn",
                summary=f"terminal outcome {outcome!r} never observed in baseline",
                detail=f"baseline outcomes: {sorted(profile['outcomes'])}",
                scope="run",
                run_id=run_id,
            )
        )
    if outcome is None and run["status"] == "ok":
        alerts.append(
            Alert(
                signal="outcome.missing",
                severity="critical",
                summary="run completed without reaching any terminal outcome",
                detail=f"path: {path}",
                scope="run",
                run_id=run_id,
            )
        )

    return alerts


def _always_visited(profile: dict) -> set[str]:
    """Nodes that appear in every baseline path — skipping one is a real signal."""
    paths = [p.split(" > ") for p in profile["node_paths"]]
    if not paths:
        return set()
    common = set(paths[0])
    for path in paths[1:]:
        common &= set(path)
    return common


def evaluate_window(runs: list[dict], profile: dict) -> list[Alert]:
    """Signals 3 and 4 in aggregate.

    Single-run noise is not news; a shifted distribution is. This is what
    catches a tool that started returning [] instead of raising.
    """
    alerts: list[Alert] = []
    if len(runs) < MIN_WINDOW_RUNS:
        return alerts
    n = len(runs)

    # --- signal 3: tool health --------------------------------------------
    totals: Counter[str] = Counter()
    empties: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for run in runs:
        for event in run["tool_events"]:
            totals[event["tool"]] += 1
            empties[event["tool"]] += int(event["empty"])
            errors[event["tool"]] += int(event["errored"])

    for tool, calls in totals.items():
        base = profile["tool_rates"].get(tool)
        if not base or calls < 3:
            continue
        for kind, counter in (("empty", empties), ("error", errors)):
            observed = counter[tool] / calls
            expected = base[f"{kind}_rate"]
            if observed > expected + TOOL_RATE_MARGIN:
                alerts.append(
                    Alert(
                        signal=f"tool.{kind}_rate_spike",
                        severity="critical",
                        # Keep the volatile numbers out of the summary: the summary
                        # is the de-duplication key, so a count that ticks up every
                        # run would defeat suppression and spam the operator.
                        summary=f"{tool} {kind} rate {observed:.0%} vs baseline {expected:.0%}",
                        detail=f"{counter[tool]}/{calls} calls over the last {n} runs",
                        scope="aggregate",
                    )
                )

    # --- signal 4: outcome mix --------------------------------------------
    observed_rates = {
        outcome: count / n
        for outcome, count in Counter(r["outcome"] for r in runs if r["outcome"]).items()
    }
    for outcome, expected in profile["outcome_rates"].items():
        observed = observed_rates.get(outcome, 0.0)
        if expected >= 0.10 and observed < expected / 3:
            alerts.append(
                Alert(
                    signal="outcome.rate_collapse",
                    severity="critical",
                    summary=f"outcome {outcome!r} collapsed vs baseline {expected:.0%}",
                    detail=f"now {observed:.0%} over the last {n} runs — "
                    "check whether a gate stopped firing",
                    scope="aggregate",
                )
            )

    return alerts
