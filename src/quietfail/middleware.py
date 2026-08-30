"""Adapter for agents built with `create_agent` rather than a raw StateGraph.

`Watcher` collects by streaming a compiled graph, which works for any
`StateGraph`. `create_agent` hides its graph behind an agent abstraction, so
this middleware plugs into the same lifecycle from the inside:

    from quietfail import Store
    from quietfail.middleware import QuietfailMiddleware

    store = Store("quietfail.sqlite")
    agent = create_agent(model="...", tools=[...], middleware=[
        QuietfailMiddleware(store, agent="support_bot", budget_factory=lambda: Budget(max_usd=1.0)),
    ])

Same store, same baselines, same CLI. What differs is the trajectory: there
are no graph nodes to record, so the path is the model/tool call sequence,
which is the equivalent notion of "what route did this run take".

Requires `langchain>=1.0` — imported lazily so the core stays dependency-free.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .baseline import evaluate_run, evaluate_window
from .budget import Budget, BudgetExceeded, CircuitBreaker, CircuitOpen
from .drift import Embedder, evaluate_content
from .store import Alert, RunRecord, Store, ToolEvent


def _import_base():
    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError as exc:  # pragma: no cover - exercised only without langchain
        raise ImportError(
            "QuietfailMiddleware needs langchain>=1.0. "
            "Install it, or use Watcher with a compiled StateGraph instead."
        ) from exc
    return AgentMiddleware


class QuietfailMiddleware(_import_base()):  # type: ignore[misc]
    """Records a `create_agent` run and evaluates it against the baseline."""

    def __init__(
        self,
        store: Store,
        agent: str,
        *,
        budget_factory: Callable[[], Budget] | None = None,
        breaker_factory: Callable[[], CircuitBreaker] | None = None,
        output_text: Callable[[Any], str] | None = None,
        embedder: Embedder | None = None,
        outcome: Callable[[Any], str | None] | None = None,
        window: int = 25,
        suppress_window: int = 50,
        evaluate: bool = True,
        on_alert: Callable[[Alert], None] | None = None,
    ):
        super().__init__()
        self.store = store
        self.agent_name = agent
        self.budget_factory = budget_factory
        self.breaker_factory = breaker_factory
        self.output_text = output_text
        self.embedder = embedder
        self.outcome = outcome
        self.window = window
        self.suppress_window = suppress_window
        self.evaluate = evaluate
        self.on_alert = on_alert or (lambda alert: print(f"  !! {alert}"))
        # One run in flight per middleware instance.
        #
        # The obvious design — key by id(state) — does not work: LangGraph
        # hands each node a freshly built state dict, so the identity seen in
        # before_agent is gone by after_agent. A ContextVar is no better,
        # because nodes may execute on different threads. So the middleware
        # instance IS the run scope, and concurrent runs need one instance
        # each (see `for_run()`).
        self._current: dict | None = None

    # ---- lifecycle ---------------------------------------------------------

    def for_run(self) -> QuietfailMiddleware:
        """A fresh instance with the same config, for concurrent runs."""
        return QuietfailMiddleware(
            self.store,
            self.agent_name,
            budget_factory=self.budget_factory,
            breaker_factory=self.breaker_factory,
            output_text=self.output_text,
            embedder=self.embedder,
            outcome=self.outcome,
            window=self.window,
            suppress_window=self.suppress_window,
            evaluate=self.evaluate,
            on_alert=self.on_alert,
        )

    def before_agent(self, state, runtime) -> None:
        self._current = {
            "record": RunRecord(agent=self.agent_name),
            "budget": self.budget_factory() if self.budget_factory else None,
            "breaker": self.breaker_factory() if self.breaker_factory else None,
            "started": time.perf_counter(),
        }
        return None

    def wrap_model_call(self, request, handler):
        context = self._context(request)
        if context:
            record, budget = context["record"], context["budget"]
            record.node_path.append("model")
            if budget is not None:
                # Check BEFORE spending, so the breach stops the next call
                # rather than merely reporting the one that blew the budget.
                budget.charge_step()
        response = handler(request)
        if context:
            self._charge_usage(context, response)
        return response

    def wrap_tool_call(self, request, handler):
        context = self._context(request)
        tool_name = self._tool_name(request)
        started = time.perf_counter()
        errored = False
        result = None
        try:
            result = handler(request)
            return result
        except Exception:
            errored = True
            raise
        finally:
            if context:
                record = context["record"]
                record.node_path.append(f"tool:{tool_name}")
                empty = self._is_empty(result)
                record.tool_events.append(
                    ToolEvent(
                        tool=tool_name,
                        empty=empty,
                        errored=errored,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
                )
                if context["breaker"] is not None:
                    context["breaker"].record(tool_name, failed=errored or empty)
                if context["budget"] is not None:
                    context["budget"].charge_tool_call()

    def after_agent(self, state, runtime) -> None:
        context, self._current = self._current, None
        if context is None:
            return None

        record, budget = context["record"], context["budget"]
        record.duration_ms = int((time.perf_counter() - context["started"]) * 1000)
        record.ended_at = datetime.now(UTC).isoformat()
        record.status = "ok"
        if budget is not None:
            record.tokens, record.usd = budget.tokens, budget.usd

        record.output_keys = sorted(self._state_keys(state))
        record.outcome = self._resolve_outcome(state)
        if self.output_text is not None:
            try:
                record.output_text = self.output_text(state)
            except Exception as exc:
                record.output_text = None
                print(f"quietfail: output_text extractor failed: {exc}")

        self.store.save_run(record)
        if self.evaluate:
            self._evaluate(record)
        return None

    # ---- helpers -----------------------------------------------------------

    def _context(self, _request) -> dict | None:
        """The run currently in flight, if any."""
        return self._current

    @staticmethod
    def _tool_name(request) -> str:
        for attr in ("tool_name", "name"):
            value = getattr(request, attr, None)
            if isinstance(value, str):
                return value
        call = getattr(request, "tool_call", None)
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            return call["name"]
        tool = getattr(request, "tool", None)
        return getattr(tool, "name", None) or "unknown_tool"

    @staticmethod
    def _is_empty(result) -> bool:
        if result is None:
            return True
        content = getattr(result, "content", result)
        if isinstance(content, str | list | tuple | dict | set):
            return len(content) == 0
        return False

    @staticmethod
    def _charge_usage(context, response) -> None:
        budget = context["budget"]
        if budget is None:
            return
        usage = None
        for attr in ("usage_metadata", "usage"):
            usage = getattr(response, attr, None) or usage
        message = getattr(response, "result", None)
        if usage is None and message is not None:
            usage = getattr(message, "usage_metadata", None)
        if isinstance(usage, dict):
            budget.charge_tokens(int(usage.get("total_tokens", 0) or 0))

    @staticmethod
    def _state_keys(state) -> set[str]:
        if isinstance(state, dict):
            return {k for k, v in state.items() if v not in (None, [], {})}
        return {k for k in dir(state) if not k.startswith("_")}

    def _resolve_outcome(self, state) -> str | None:
        if self.outcome is not None:
            value = self.outcome(state)
            return str(value) if value is not None else None
        # Default: did the agent produce a final answer or not? That is the
        # create_agent equivalent of a terminal state.
        messages = state.get("messages") if isinstance(state, dict) else None
        if not messages:
            return None
        last = messages[-1]
        content = getattr(last, "content", None)
        return "answered" if content else "empty_answer"

    def _evaluate(self, record: RunRecord) -> list[Alert]:
        profile = self.store.load_baseline(self.agent_name)
        if not profile:
            return []

        run = self.store.runs(self.agent_name)[-1]
        alerts = evaluate_run(run, profile)
        alerts += evaluate_window(
            self.store.recent_runs(self.agent_name, self.window, since=profile.get("_built_at")),
            profile,
        )
        alerts += evaluate_content(
            run.get("output_text"),
            profile.get("content"),
            embedder=self.embedder,
            run_id=run["run_id"],
        )

        seen = self.store.recent_alert_keys(self.suppress_window)
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


__all__ = ["QuietfailMiddleware", "BudgetExceeded", "CircuitOpen"]
