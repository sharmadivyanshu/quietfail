"""Where alerts go.

Sinks are plain callables taking one Alert, so anything with the right shape
works. Failures are swallowed on purpose: a broken Slack webhook must not
take down the agent it is watching.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable

from .store import Alert

SEVERITY_ICON = {"critical": "🔴", "warn": "🟠", "info": "🔵"}


def stdout_sink(stream=None) -> Callable[[Alert], None]:
    def sink(alert: Alert) -> None:
        print(f"  !! {alert}", file=stream or sys.stderr)

    return sink


def webhook_sink(url: str, timeout: float = 5.0) -> Callable[[Alert], None]:
    """POST the alert as JSON."""

    def sink(alert: Alert) -> None:
        payload = json.dumps(
            {
                "signal": alert.signal,
                "severity": alert.severity,
                "summary": alert.summary,
                "detail": alert.detail,
                "scope": alert.scope,
                "run_id": alert.run_id,
            }
        ).encode()
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(request, timeout=timeout).close()
        except (urllib.error.URLError, OSError) as exc:
            print(f"quietfail: webhook delivery failed: {exc}", file=sys.stderr)

    return sink


def slack_sink(webhook_url: str, timeout: float = 5.0) -> Callable[[Alert], None]:
    """Slack incoming webhook, formatted for a human on call."""

    def sink(alert: Alert) -> None:
        icon = SEVERITY_ICON.get(alert.severity, "⚪")
        payload = json.dumps(
            {
                "text": f"{icon} *{alert.signal}*\n{alert.summary}\n_{alert.detail}_",
            }
        ).encode()
        request = urllib.request.Request(
            webhook_url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(request, timeout=timeout).close()
        except (urllib.error.URLError, OSError) as exc:
            print(f"quietfail: slack delivery failed: {exc}", file=sys.stderr)

    return sink


def fan_out(*sinks: Callable[[Alert], None]) -> Callable[[Alert], None]:
    """Send every alert to several sinks; one failing never blocks the rest."""

    def sink(alert: Alert) -> None:
        for downstream in sinks:
            try:
                downstream(alert)
            except Exception as exc:
                print(f"quietfail: sink {downstream!r} failed: {exc}", file=sys.stderr)

    return sink


def severity_filter(
    sink: Callable[[Alert], None], minimum: str = "warn"
) -> Callable[[Alert], None]:
    """Only forward alerts at or above `minimum`."""
    order = {"info": 0, "warn": 1, "critical": 2}
    floor = order.get(minimum, 1)

    def filtered(alert: Alert) -> None:
        if order.get(alert.severity, 0) >= floor:
            sink(alert)

    return filtered
