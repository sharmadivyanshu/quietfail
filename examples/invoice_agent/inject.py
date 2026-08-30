"""Failure injection — the proof that quietfail works.

    poetry run python -m invoice_agent.inject

Phase 1 records normal runs and builds a baseline.
Phase 2 breaks the agent in five realistic ways, none of which raise an
exception or turn a dashboard red, and checks that each one is caught. It also
runs a healthy control first — proving the monitor stays quiet matters as much
as proving it catches things.

Exits non-zero if any injection is missed or the control raises anything, so
CI can gate on it.

Every scenario here is modelled on a documented production failure, not an
invented one.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path

from invoice_agent import nodes, tools
from invoice_agent.graph import build_graph
from quietfail import Budget, Store, Watcher, build_profile

DB = Path("quietfail-demo.sqlite")
AGENT = "invoice_agent"
FIXTURES = [
    "clean-001",
    "missing-gl-002",
    "unknown-vendor-003",
    "totals-broken-004",
    "duplicate-005",
]

GREEN, RED, DIM, BOLD, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def output_text(values: dict) -> str:
    """The prose quietfail watches for wording drift.

    Built from fields that exist on every run — including paused ones — so the
    content baseline isn't limited to invoices that ran to completion.
    """
    parts: list[str] = []
    invoice = values.get("extracted")
    if invoice is not None:
        parts.append(invoice.vendor_name)
        parts.extend(item.description for item in invoice.line_items)
    parts.extend(finding.detail for finding in values.get("findings") or [])
    return " | ".join(parts)


def record(watcher: Watcher, tag: str, *, evaluate: bool, cycles: int = 1) -> list:
    """Run every fixture `cycles` times. Returns any alerts raised."""
    alerts = []
    for cycle in range(cycles):
        for fixture in FIXTURES:
            graph = build_graph()
            config = {"configurable": {"thread_id": f"{tag}-{cycle}-{fixture}"}}
            _, _, run_alerts = watcher.run(
                graph,
                {"document_uri": fixture, "tenant_id": "demo"},
                config=config,
                budget=Budget(max_steps=40),
                evaluate=evaluate,
            )
            alerts.extend(run_alerts)
    return alerts


# --- the injections ----------------------------------------------------


@contextlib.contextmanager
def tool_returns_empty() -> Iterator[str]:
    """A dependency starts returning nothing instead of raising.

    The single most common silent failure: an upstream API answers [] rather
    than 500, the agent treats it as 'no match', and everything looks fine.
    """
    # Break the DATA, not the function — so the instrumented wrapper still
    # runs, exactly as it would if a real upstream sync came back empty.
    original = dict(tools.VENDOR_ALIASES)
    tools.VENDOR_ALIASES.clear()
    try:
        yield "vendor master sync returned empty — every lookup now misses"
    finally:
        tools.VENDOR_ALIASES.update(original)


@contextlib.contextmanager
def approval_gate_removed() -> Iterator[str]:
    """Someone 'improves throughput' by lowering the confidence threshold.

    Escalations stop. Nothing errors. Bad invoices post themselves. This is
    the failure the escalation-rate signal exists to catch.
    """
    original = nodes.human_review
    nodes.human_review = lambda state: {
        "human_decision": {"approved": True, "reviewer": "auto"},
        "requires_human": False,
    }
    try:
        yield "human_review now auto-approves — the gate is gone, nothing escalates"
    finally:
        nodes.human_review = original


@contextlib.contextmanager
def output_field_dropped() -> Iterator[str]:
    """A refactor stops setting a field the downstream system needs.

    Runs still complete. The field is just quietly absent.
    """
    original = nodes.post
    nodes.post = lambda state: {}
    try:
        yield "post() no longer sets `outcome`"
    finally:
        nodes.post = original


@contextlib.contextmanager
def resolution_never_applies() -> Iterator[str]:
    """A fix that doesn't stick, so the agent loops until the guard stops it.

    This is the retry-spiral shape that burned $4,200 in 63 hours.
    """
    original = nodes.apply_resolution
    nodes.apply_resolution = lambda state: {"iteration": state.get("iteration", 0) + 1}
    try:
        yield "apply_resolution became a no-op — fixes never stick"
    finally:
        nodes.apply_resolution = original


@contextlib.contextmanager
def extraction_degrades() -> Iterator[str]:
    """Upstream OCR degrades: fields are all present and well-typed, values are
    just wrong.

    Every structural signal is blind here — same shape, same path, same outcome
    mix, same tool behaviour. Only content drift sees it.
    """
    from invoice_agent import extract as extract_mod

    original = extract_mod.StubExtractor.extract

    def garbled(self, document_uri: str):
        invoice = original(self, document_uri)
        invoice.vendor_name = _garble(invoice.vendor_name)
        for item in invoice.line_items:
            item.description = _garble(item.description)
        return invoice

    extract_mod.StubExtractor.extract = garbled
    try:
        yield "OCR degraded — every field still present, the text is mush"
    finally:
        extract_mod.StubExtractor.extract = original


def _garble(text: str) -> str:
    """Deterministic character mangling, the shape a bad OCR pass produces."""
    swaps = str.maketrans({"o": "0", "l": "1", "e": "3", "a": "@", "s": "5", "i": "!"})
    return text.translate(swaps).replace(" ", "  ")


SCENARIOS = [
    ("tool goes silently empty", tool_returns_empty, ["tool.empty_rate_spike"]),
    ("approval gate removed", approval_gate_removed, ["outcome.rate_collapse"]),
    ("output field dropped", output_field_dropped, ["outcome.missing", "shape.missing_key"]),
    (
        "resolution loop never converges",
        resolution_never_applies,
        ["trajectory.step_explosion", "trajectory.unseen_path"],
    ),
    ("extraction quietly degrades", extraction_degrades, ["content.drift"]),
]


def main() -> int:
    for stale in Path().glob("quietfail-demo*.sqlite"):
        stale.unlink()

    baseline_store = Store(DB)
    quiet = Watcher(baseline_store, agent=AGENT, output_text=output_text, on_alert=lambda a: None)

    print(f"\n{BOLD}Phase 1 — record normal behaviour{RESET}")
    record(quiet, "baseline", evaluate=False, cycles=4)
    runs = baseline_store.runs(AGENT)
    profile = build_profile(runs)
    baseline_store.save_baseline(AGENT, len(runs), profile)
    print(f"  {len(runs)} runs recorded, baseline built")
    print(f"  {DIM}paths seen     : {len(profile['node_paths'])}{RESET}")
    print(
        f"  {DIM}outcome mix    : "
        f"{ {k: f'{v:.0%}' for k, v in profile['outcome_rates'].items()} }{RESET}"
    )
    print(f"  {DIM}tools tracked  : {sorted(profile['tool_rates'])}{RESET}")

    print(f"\n{BOLD}Phase 2 — break it five ways{RESET}")
    print(f"{DIM}None of these raise. None turn a dashboard red.{RESET}\n")

    # Control: nothing is broken. A monitor that cries wolf here is useless,
    # so this is the first thing that has to pass.
    control_store = Store("quietfail-demo-control.sqlite")
    control_store.save_baseline(AGENT, len(runs), profile)
    control = Watcher(control_store, agent=AGENT, output_text=output_text, on_alert=lambda a: None)
    control_alerts = record(control, "control", evaluate=True, cycles=3)
    clean_control = not control_alerts
    if control_alerts:
        print(
            f"  [{RED}FALSE POSITIVE{RESET}] control run raised "
            f"{len(control_alerts)} alert(s) with nothing broken:"
        )
        for alert in control_alerts:
            print(f"           -> {alert.signal}: {alert.summary}")
    else:
        print(f"  [{GREEN}QUIET{RESET}] control — 15 healthy runs, 0 alerts")
    print()

    passed = 0
    for index, (title, injection, expected) in enumerate(SCENARIOS):
        # Each scenario gets a clean store seeded with the same baseline, so
        # one injection's damage cannot contaminate the next one's window.
        scenario_store = Store(f"quietfail-demo-{index}.sqlite")
        scenario_store.save_baseline(AGENT, len(runs), profile)
        watcher = Watcher(
            scenario_store, agent=AGENT, output_text=output_text, on_alert=lambda a: None
        )

        with injection() as description:
            alerts = record(watcher, f"inject-{index}", evaluate=True, cycles=3)

        signals = {a.signal for a in alerts}
        ok = any(sig in signals for sig in expected)
        passed += ok

        mark = f"{GREEN}CAUGHT{RESET}" if ok else f"{RED}MISSED{RESET}"
        print(f"  [{mark}] {title}")
        print(f"           {DIM}{description}{RESET}")
        for signal in sorted(signals):
            first = next(a for a in alerts if a.signal == signal)
            print(f"           -> {first.severity}: {first.summary}")
        if not ok:
            print(f"           {RED}expected one of {expected}{RESET}")
        print()

    print(f"{BOLD}{passed}/{len(SCENARIOS)} silent failures detected{RESET}")
    if passed < len(SCENARIOS) or not clean_control:
        print(f"{RED}FAILED{RESET} — detection incomplete or the control was noisy\n")
        return 1
    print(
        f"{DIM}Report:  quietfail --db quietfail-demo-0.sqlite report "
        f"--agent {AGENT} -o report.html{RESET}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
