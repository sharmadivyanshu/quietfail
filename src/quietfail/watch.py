"""Collection layer — the bit that actually watches a LangGraph agent.

Uses `graph.stream(stream_mode="updates")` rather than monkeypatching nodes.
That keeps quietfail on a public, supported API surface: if it runs on
LangGraph, it can be watched, with no changes to the graph itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps
from typing import Any

from .baseline import evaluate_run, evaluate_window
from .budget import Budget, BudgetExceeded, CircuitBreaker, CircuitOpen
from .drift import Embedder, evaluate_content
from .store import Alert, RunRecord, Store, ToolEvent

# The run currently being recorded on this thread/task, so instrumented tools
# can attach their events without every call site passing a recorder around.
_ACTIVE: ContextVar[RunRecord | None] = ContextVar("quietfail_active_run", default=None)

PAUSED_OUTCOME = "awaiting_human"


def _default_is_empty(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, list | tuple | set | dict | str):
        return len(result) == 0
    return False


def instrument_tool(
    func: Callable | None = None,
    *,
    name: str | None = None,
    is_empty: Callable[[Any], bool] | None = None,
):
    """Record every call to this tool: latency, errors, and empty results.

    Empty results are the point. A tool that returns [] instead of raising is
    the single most common way an agent fails without anything going red.
    """

    def decorate(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        empty_check = is_empty or _default_is_empty

        @wraps(fn)
        def wrapper(*args, **kwargs):
            run = _ACTIVE.get()
            started = time.perf_counter()
            errored, empty = False, False
            try:
                result = fn(*args, **kwargs)
                empty = bool(empty_check(result))
                return result
            except Exception:
                errored = True
                raise
            finally:
                if run is not None:
                    run.tool_events.append(
                        ToolEvent(
                            tool=tool_name,
                            empty=empty,
                            errored=errored,
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
                    )
                    breaker = _BREAKERS.get(run.run_id)
                    if breaker is not None:
                        breaker.record(tool_name, failed=errored or empty)
                    budget = _BUDGETS.get(run.run_id)
                    if budget is not None:
                        budget.charge_tool_call()

        return wrapper

    return decorate(func) if func else decorate


_BUDGETS: dict[str, Budget] = {}
_BREAKERS: dict[str, CircuitBreaker] = {}


@contextmanager
def _active(
    run: RunRecord, budget: Budget | None, breaker: CircuitBreaker | None
) -> Iterator[None]:
    token = _ACTIVE.set(run)
    if budget:
        _BUDGETS[run.run_id] = budget
    if breaker:
        _BREAKERS[run.run_id] = breaker
    try:
        yield
    finally:
        _ACTIVE.reset(token)
        _BUDGETS.pop(run.run_id, None)
        _BREAKERS.pop(run.run_id, None)


class Watcher:
    """Runs an agent and records what it did.

    Two modes:
      observe  — record only, build up a baseline (the default until you have one)
      watch    — record and evaluate every run against the baseline
    """

    def __init__(
        self,
        store: Store,
        agent: str,
        *,
        outcome_key: str = "outcome",
        window: int = 25,
        suppress_window: int = 50,
        output_text: Callable[[dict], str] | None = None,
        embedder: Embedder | None = None,
        on_alert: Callable[[Alert], None] | None = None,
    ):
        self.store = store
        self.agent = agent
        self.outcome_key = outcome_key
        self.window = window
        self.suppress_window = suppress_window
        # How to turn a final state into the text whose wording we watch.
        # Default: nothing, because only the caller knows which field is the
        # agent's actual answer.
        self.output_text = output_text
        self.embedder = embedder
        self.on_alert = on_alert or (lambda alert: print(f"  !! {alert}"))

    def run(
        self,
        graph,
        payload: Any,
        *,
        config: dict,
        budget: Budget | None = None,
        breaker: CircuitBreaker | None = None,
        evaluate: bool = True,
    ) -> tuple[dict, RunRecord, list[Alert]]:
        record = RunRecord(
            agent=self.agent,
            thread_id=str(config.get("configurable", {}).get("thread_id")),
        )
        started = time.perf_counter()

        with _active(record, budget, breaker):
            try:
                for chunk in graph.stream(payload, config=config):
                    for node_name in chunk:
                        if node_name == "__interrupt__":
                            continue
                        record.node_path.append(node_name)
                        if budget:
                            budget.charge_step()
                record.status = "ok"
            except (BudgetExceeded, CircuitOpen) as exc:
                record.status = "error"
                record.error = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                record.status = "error"
                record.error = f"{type(exc).__name__}: {exc}"
                self._finish(record, graph, config, started, budget)
                raise

        self._finish(record, graph, config, started, budget)
        alerts = self._evaluate(record) if evaluate else []
        final = graph.get_state(config).values
        return final, record, alerts

    def _finish(self, record, graph, config, started, budget) -> None:
        record.duration_ms = int((time.perf_counter() - started) * 1000)
        record.ended_at = datetime.now(UTC).isoformat()
        if budget:
            record.tokens, record.usd = budget.tokens, budget.usd

        state = graph.get_state(config)
        values = state.values or {}
        record.output_keys = [k for k, v in values.items() if v not in (None, [], {})]

        if self.output_text is not None:
            try:
                record.output_text = self.output_text(values)
            except Exception as exc:
                record.output_text = None
                print(f"quietfail: output_text extractor failed: {exc}")

        if state.next:
            # Parked at a human gate. Treating that as a terminal outcome is
            # deliberate — it makes the escalation rate a first-class signal.
            record.outcome = PAUSED_OUTCOME
        else:
            outcome = values.get(self.outcome_key)
            record.outcome = str(outcome) if outcome is not None else None

        self.store.save_run(record)

    def _evaluate(self, record: RunRecord) -> list[Alert]:
        profile = self.store.load_baseline(self.agent)
        if not profile:
            return []

        run = self.store.runs(self.agent)[-1]
        alerts = evaluate_run(run, profile)
        alerts += evaluate_window(
            self.store.recent_runs(self.agent, self.window, since=profile.get("_built_at")),
            profile,
        )
        alerts += evaluate_content(
            run.get("output_text"),
            profile.get("content"),
            embedder=self.embedder,
            run_id=run["run_id"],
        )

        # Suppress anything already raised recently — within this evaluation
        # and across previous runs. An unactioned alert repeated 15 times is
        # how a monitor gets muted.
        seen: set[tuple[str, str]] = self.store.recent_alert_keys(self.suppress_window)
        unique: list[Alert] = []
        for alert in alerts:
            key = (alert.signal, alert.summary)
            if key in seen:
                continue
            seen.add(key)
            unique.append(alert)

        self.store.save_alerts(unique)
        for alert in unique:
            self.on_alert(alert)
        return unique
