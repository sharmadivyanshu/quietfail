"""quietfail CLI.

quietfail baseline --agent invoice_agent
quietfail alerts   --agent invoice_agent
quietfail report   --agent invoice_agent -o report.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .baseline import build_profile
from .report import render_html, render_json
from .store import Store


def _store(args) -> Store:
    return Store(args.db)


def cmd_baseline(args) -> int:
    store = _store(args)
    runs = store.runs(args.agent)
    if len(runs) < args.min_runs:
        print(
            f"only {len(runs)} runs recorded for {args.agent!r}; "
            f"need at least {args.min_runs}. Run in observe mode for longer.",
            file=sys.stderr,
        )
        return 1
    profile = build_profile(runs)
    store.save_baseline(args.agent, len(runs), profile)
    print(f"baseline built for {args.agent!r} from {len(runs)} runs")
    print(f"  distinct paths : {len(profile['node_paths'])}")
    print(f"  outcomes       : {', '.join(sorted(profile['outcomes'])) or '(none)'}")
    print(f"  tools tracked  : {', '.join(sorted(profile['tool_rates'])) or '(none)'}")
    print(f"  steps p50/p99  : {profile['step_count_p50']:.0f} / {profile['step_count_p99']:.0f}")
    return 0


def cmd_alerts(args) -> int:
    alerts = _store(args).alerts(limit=args.limit)
    if not alerts:
        print("no alerts recorded")
        return 0
    for alert in reversed(alerts):
        print(f"[{alert['severity']:<8}] {alert['signal']:<28} {alert['summary']}")
        if args.verbose:
            print(f"           {alert['detail']}")
    criticals = sum(1 for a in alerts if a["severity"] == "critical")
    print(f"\n{len(alerts)} alert(s), {criticals} critical")
    return 1 if criticals else 0


def cmd_report(args) -> int:
    store = _store(args)
    runs = store.runs(args.agent)
    alerts = store.alerts(limit=500)
    profile = store.load_baseline(args.agent)
    if args.format == "json":
        output = render_json(args.agent, runs, alerts, profile)
    else:
        output = render_html(args.agent, runs, alerts, profile)
    Path(args.output).write_text(output)
    print(f"wrote {args.output} ({len(runs)} runs, {len(alerts)} alerts)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quietfail", description=__doc__)
    parser.add_argument("--db", default="quietfail.sqlite", help="path to the run store")
    sub = parser.add_subparsers(dest="command", required=True)

    p_base = sub.add_parser("baseline", help="build a baseline profile from recorded runs")
    p_base.add_argument("--agent", required=True)
    p_base.add_argument("--min-runs", type=int, default=10)
    p_base.set_defaults(func=cmd_baseline)

    p_alerts = sub.add_parser("alerts", help="list recorded alerts")
    p_alerts.add_argument("--agent", default=None)
    p_alerts.add_argument("--limit", type=int, default=50)
    p_alerts.add_argument("-v", "--verbose", action="store_true")
    p_alerts.set_defaults(func=cmd_alerts)

    p_report = sub.add_parser("report", help="write a standalone report")
    p_report.add_argument("--agent", required=True)
    p_report.add_argument("-o", "--output", default="quietfail-report.html")
    p_report.add_argument("--format", choices=["html", "json"], default="html")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
