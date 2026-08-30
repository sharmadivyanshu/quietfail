"""Runtime budgets and circuit breaking.

The point is enforcement, not reporting. A budget that logs a warning while
the loop keeps spending is not a budget. Breaches raise.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    """Raised the moment a run crosses a limit. Deliberately fatal."""


class CircuitOpen(RuntimeError):
    """Raised when one tool fails repeatedly in a single run.

    This is the guard for the documented failure shape where an agent read
    HTTP 429s as transient and retried ~4,800 times an hour for 63 hours.
    """


@dataclass
class Budget:
    """Per-run limits. Construct one per run, never share across runs."""

    max_tokens: int | None = None
    max_usd: float | None = None
    max_tool_calls: int | None = None
    max_seconds: float | None = None
    max_steps: int | None = None

    tokens: int = 0
    usd: float = 0.0
    tool_calls: int = 0
    steps: int = 0
    started: float = field(default_factory=time.monotonic)

    def charge_tokens(self, tokens: int, usd: float = 0.0) -> None:
        self.tokens += tokens
        self.usd += usd
        self._check()

    def charge_tool_call(self) -> None:
        self.tool_calls += 1
        self._check()

    def charge_step(self) -> None:
        self.steps += 1
        self._check()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def _check(self) -> None:
        for limit, used, label in (
            (self.max_tokens, self.tokens, "tokens"),
            (self.max_usd, self.usd, "usd"),
            (self.max_tool_calls, self.tool_calls, "tool_calls"),
            (self.max_steps, self.steps, "steps"),
            (self.max_seconds, self.elapsed, "seconds"),
        ):
            if limit is not None and used > limit:
                raise BudgetExceeded(
                    f"{label} budget exceeded: {used:.4g} > {limit:.4g}. {self.report()}"
                )

    def report(self) -> str:
        return (
            f"spent tokens={self.tokens} usd={self.usd:.4f} "
            f"tool_calls={self.tool_calls} steps={self.steps} "
            f"elapsed={self.elapsed:.1f}s"
        )


@dataclass
class CircuitBreaker:
    """Trips when the same tool fails or returns empty too many times in a run."""

    threshold: int = 5
    failures: dict[str, int] = field(default_factory=dict)

    def record(self, tool: str, *, failed: bool) -> None:
        if not failed:
            self.failures[tool] = 0
            return
        self.failures[tool] = self.failures.get(tool, 0) + 1
        if self.failures[tool] >= self.threshold:
            raise CircuitOpen(
                f"circuit open for {tool!r}: {self.failures[tool]} consecutive "
                f"failures in a single run"
            )
