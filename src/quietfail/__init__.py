"""quietfail — catch LLM agents that fail silently.

    from quietfail import Store, Watcher, Budget, instrument_tool

    store = Store("quietfail.sqlite")
    watcher = Watcher(store, agent="my_agent")
    final, record, alerts = watcher.run(graph, payload, config=config)

Build a baseline from runs you trust, then every subsequent run is compared
against it. No labels, no ground truth — just this agent's own history.
"""

from .baseline import build_profile, evaluate_run, evaluate_window
from .budget import Budget, BudgetExceeded, CircuitBreaker, CircuitOpen
from .drift import Embedder, LexicalEmbedder, build_content_profile, evaluate_content
from .report import render_html
from .sinks import fan_out, severity_filter, slack_sink, stdout_sink, webhook_sink
from .store import Alert, RunRecord, Store, ToolEvent
from .watch import Watcher, instrument_tool

__version__ = "0.1.0"

__all__ = [
    "Alert",
    "Budget",
    "BudgetExceeded",
    "CircuitBreaker",
    "CircuitOpen",
    "Embedder",
    "LexicalEmbedder",
    "RunRecord",
    "Store",
    "ToolEvent",
    "Watcher",
    "build_content_profile",
    "build_profile",
    "evaluate_content",
    "fan_out",
    "severity_filter",
    "slack_sink",
    "stdout_sink",
    "webhook_sink",
    "evaluate_run",
    "evaluate_window",
    "instrument_tool",
    "render_html",
]
