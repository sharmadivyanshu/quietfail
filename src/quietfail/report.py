"""Single-file HTML report. No dashboard server, no build step.

The incumbents all own the dashboard. quietfail's job is the mechanism, so
the output is one file you can open, email, or attach to an incident.
"""

from __future__ import annotations

import html
import json
from collections import Counter

SEVERITY_COLOR = {"critical": "#b42318", "warn": "#b54708", "info": "#175cd3"}

TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>quietfail — {agent}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, sans-serif;
         margin: 0; padding: 2.5rem; max-width: 1000px; margin-inline: auto; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
  .sub {{ opacity: .65; margin-bottom: 2rem; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem; margin-bottom: 2.5rem; }}
  .tile {{ border: 1px solid color-mix(in srgb, currentColor 15%, transparent);
           border-radius: 10px; padding: 1rem; }}
  .tile .n {{ font-size: 1.75rem; font-weight: 600; }}
  .tile .l {{ opacity: .65; font-size: .8rem; text-transform: uppercase;
              letter-spacing: .05em; }}
  h2 {{ font-size: 1rem; text-transform: uppercase; letter-spacing: .06em;
        opacity: .65; margin: 2rem 0 .75rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .9rem; }}
  th, td {{ text-align: left; padding: .5rem .6rem;
            border-bottom: 1px solid color-mix(in srgb, currentColor 12%, transparent); }}
  th {{ font-weight: 600; opacity: .7; }}
  code {{ font: .85em ui-monospace, SFMono-Regular, monospace;
          background: color-mix(in srgb, currentColor 8%, transparent);
          padding: .1em .35em; border-radius: 4px; }}
  .sev {{ font-weight: 600; }}
  .empty {{ opacity: .55; font-style: italic; padding: 1rem 0; }}
  .bar {{ height: 8px; border-radius: 4px;
          background: color-mix(in srgb, currentColor 20%, transparent); }}
  .bar > i {{ display: block; height: 100%; border-radius: 4px; background: currentColor; }}
</style></head><body>
<h1>quietfail — {agent}</h1>
<div class="sub">{runs_total} runs recorded · baseline built from {baseline_n} runs</div>
<div class="tiles">{tiles}</div>
<h2>Alerts</h2>{alerts}
<h2>Outcome mix</h2>{outcomes}
<h2>Tool health</h2>{tools}
<h2>Recent runs</h2>{recent}
</body></html>
"""


def _tile(n: object, label: str) -> str:
    return (
        f'<div class="tile"><div class="n">{n}</div><div class="l">{html.escape(label)}</div></div>'
    )


def _table(headers: list[str], rows: list[list[str]], empty_msg: str) -> str:
    if not rows:
        return f'<div class="empty">{html.escape(empty_msg)}</div>'
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(agent: str, runs: list[dict], alerts: list[dict], profile: dict | None) -> str:
    profile = profile or {}
    criticals = [a for a in alerts if a["severity"] == "critical"]

    tiles = "".join(
        [
            _tile(len(runs), "runs"),
            _tile(len(alerts), "alerts"),
            _tile(len(criticals), "critical"),
            _tile(sum(r["tool_calls"] for r in runs), "tool calls"),
            _tile(f"${sum(r['usd'] for r in runs):.2f}", "spend"),
        ]
    )

    alert_rows = [
        [
            f'<span class="sev" style="color:{SEVERITY_COLOR.get(a["severity"], "")}">'
            f"{html.escape(a['severity'])}</span>",
            f"<code>{html.escape(a['signal'])}</code>",
            html.escape(a["summary"]),
            f'<span style="opacity:.7">{html.escape(a["detail"])}</span>',
        ]
        for a in alerts
    ]

    observed = Counter(r["outcome"] for r in runs if r["outcome"])
    total = sum(observed.values()) or 1
    baseline_rates = profile.get("outcome_rates", {})
    outcome_rows = []
    for outcome, count in observed.most_common():
        share = count / total
        base = baseline_rates.get(outcome)
        outcome_rows.append(
            [
                f"<code>{html.escape(str(outcome))}</code>",
                str(count),
                f'<div class="bar"><i style="width:{share * 100:.0f}%"></i></div>',
                f"{share:.0%}",
                f"{base:.0%}" if base is not None else "—",
            ]
        )

    tool_stats: dict[str, dict[str, float]] = {}
    for run in runs:
        for event in run["tool_events"]:
            entry = tool_stats.setdefault(event["tool"], {"calls": 0, "empty": 0, "errored": 0})
            entry["calls"] += 1
            entry["empty"] += event["empty"]
            entry["errored"] += event["errored"]

    tool_rows = []
    for tool, stat in sorted(tool_stats.items()):
        base = profile.get("tool_rates", {}).get(tool, {})
        tool_rows.append(
            [
                f"<code>{html.escape(tool)}</code>",
                f"{stat['calls']:.0f}",
                f"{stat['empty'] / stat['calls']:.0%}",
                f"{base.get('empty_rate', 0):.0%}" if base else "—",
                f"{stat['errored'] / stat['calls']:.0%}",
            ]
        )

    recent_rows = [
        [
            f"<code>{html.escape(r['run_id'][:8])}</code>",
            html.escape(str(r["outcome"])),
            str(r["step_count"]),
            f'<span style="opacity:.7">{html.escape(" > ".join(r["node_path"]))}</span>',
            f"{r['duration_ms']} ms",
        ]
        for r in runs[-15:][::-1]
    ]

    return TEMPLATE.format(
        agent=html.escape(agent),
        runs_total=len(runs),
        baseline_n=profile.get("run_count", 0),
        tiles=tiles,
        alerts=_table(
            ["severity", "signal", "what changed", "detail"],
            alert_rows,
            "No alerts. Either nothing drifted, or no baseline exists yet.",
        ),
        outcomes=_table(
            ["outcome", "n", "", "observed", "baseline"], outcome_rows, "No outcomes recorded."
        ),
        tools=_table(
            ["tool", "calls", "empty", "baseline empty", "errors"],
            tool_rows,
            "No instrumented tool calls. Decorate tools with @instrument_tool.",
        ),
        recent=_table(
            ["run", "outcome", "steps", "path", "duration"], recent_rows, "No runs recorded."
        ),
    )


def render_json(agent: str, runs: list[dict], alerts: list[dict], profile: dict | None) -> str:
    return json.dumps(
        {"agent": agent, "runs": len(runs), "alerts": alerts, "profile": profile}, indent=2
    )
