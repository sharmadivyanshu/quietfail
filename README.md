# quietfail

**Catch LLM agents that fail silently — no ground truth required.**

Agent observability for [LangGraph](https://github.com/langchain-ai/langgraph) and LangChain. Detects silent failures, behavioural drift, and runaway cost in production AI agents by learning what *normal* looks like and alerting when a run falls outside it.

> Your agent didn't crash. It turned an error into fluent, plausible prose, returned 200 OK, and your dashboard stayed green.

![quietfail catching five injected silent failures](docs/demo.gif)

---

## The problem

An 8-week longitudinal study of a production LLM agent runtime ([arXiv 2606.14589](https://arxiv.org/abs/2606.14589v1)) found:

- **~70%** of incidents were discovered by a human happening to notice something off
- **4,286 unit tests and 827 governance checks caught approximately none** of them
- **Silence latency ranged from 13 hours to 60 days**
- The novel failure mode: the model *transforms* an error into plausible narrative content instead of surfacing it

Tracing tools show you what happened *if you go and look*. The failures that hurt are the ones nobody looks at. Evaluation frameworks need labelled ground truth, which most teams never build — evaluation has been the #1 reported challenge in the AI stack for three consecutive years, and the most common method remains the vibe check.

**quietfail takes the third path: no ground truth, no labels. Just the agent's own historical behaviour, and an alert when today deviates from it.**

## How it works

Run in observe mode, build a baseline, then watch.

| Signal | Catches | Needs labels? |
|---|---|---|
| **Output shape** — schema conformance, field presence, length distribution | Model silently dropping fields after a version bump | No |
| **Trajectory** — node sequence and per-run counts | Agent stopped calling `validate` entirely | No |
| **Tool health** — empty-result rate, error rate, latency p95 | An API returning `[]` instead of a 500 | No |
| **Terminal-state mix** — outcome distribution | Escalation rate quietly dropping to zero | No |
| **Content drift** — output distance from a baseline centroid | Upstream extraction decaying while every field stays present | No |
| **Budget breach** — tokens, USD, tool calls, wall-clock per run | The retry loop that burned $4,200 in 63 hours | No |

Detection is deliberately statistical, not ML — frequency tables, percentile bands, observed-path sets. When an alert fires you get a one-sentence reason, not a model score.

## Does it actually work?

`examples/invoice_agent/inject.py` breaks a working agent in five realistic
ways, none of which raise an exception or turn a dashboard red, and checks
that each one is caught. **It exits non-zero if any is missed, so CI gates on
it.**

```
$ poetry run python -m invoice_agent.inject

Phase 1 - record normal behaviour
  20 runs recorded, baseline built
  outcome mix : {'posted': '20%', 'awaiting_human': '80%'}

Phase 2 - break it five ways
  [QUIET]  control - 15 healthy runs, 0 alerts
  [CAUGHT] tool goes silently empty
           -> critical: vendor_lookup empty rate 100% vs baseline 0%
  [CAUGHT] approval gate removed
           -> critical: outcome 'awaiting_human' collapsed vs baseline 80%
  [CAUGHT] output field dropped
           -> critical: run completed without reaching any terminal outcome
  [CAUGHT] resolution loop never converges
           -> warn: 15 steps vs baseline p99 of 9
  [CAUGHT] extraction quietly degrades
           -> warn: output wording diverged from baseline (distance 0.84)

5/5 silent failures detected
```

The control line matters as much as the four catches. A monitor that fires on
healthy traffic gets muted, and a muted monitor is worse than none.

## Quickstart

```python
from quietfail import Store, Watcher, Budget

store   = Store("quietfail.sqlite")
watcher = Watcher(store, agent="my_agent")

final, record, alerts = watcher.run(
    graph, payload, config=config, budget=Budget(max_usd=2.0, max_steps=40)
)
```

Record runs you trust, then build the baseline:

```bash
quietfail baseline --agent my_agent      # needs >= 10 recorded runs
quietfail alerts --agent my_agent -v
quietfail report --agent my_agent -o report.html
```

To track tool health, decorate the tools your agent calls:

```python
from quietfail import instrument_tool

@instrument_tool(is_empty=lambda r: not r.get("vendor_id"))
def vendor_lookup(name: str) -> dict:
    ...
```

Alerts can go anywhere:

```python
from quietfail import Watcher, fan_out, severity_filter, slack_sink, stdout_sink

watcher = Watcher(
    store, agent="my_agent",
    output_text=lambda state: state.get("answer", ""),   # enables content drift
    on_alert=fan_out(stdout_sink(), severity_filter(slack_sink(URL), "critical")),
)
```

## Works with both agent styles

**Raw `StateGraph`** — `Watcher` streams the compiled graph, no graph changes:

```python
final, record, alerts = watcher.run(graph, payload, config=config)
```

**`create_agent`** — middleware plugs into the same lifecycle from the inside:

```python
from quietfail.middleware import QuietfailMiddleware

agent = create_agent(model=..., tools=[...], middleware=[
    QuietfailMiddleware(store, agent="support_bot",
                        budget_factory=lambda: Budget(max_usd=1.0)),
])
```

Same store, same baselines, same CLI. The trajectory differs — `create_agent`
has no graph nodes, so the path is the model/tool call sequence instead.

## Storage

SQLite by default: one file, no server, inspectable with any SQL client. For
several processes recording concurrently:

```python
from quietfail.postgres import PostgresStore
store = PostgresStore("postgresql://user:pass@host/quietfail")
```

Identical interface — everything downstream is unchanged.

## Status

**v0.1.0** — all six signals implemented, both agent styles supported, 85
tests, CI on Python 3.11-3.13 against a live Postgres.

- [x] Run store (SQLite and Postgres), collection via `graph.stream`
- [x] `create_agent` middleware adapter
- [x] Baseline profiles, per-run and aggregate drift evaluation
- [x] Content drift with a pluggable embedder
- [x] Budget enforcement and circuit breaking
- [x] Alert sinks: stdout, webhook, Slack, fan-out, severity filtering
- [x] CLI and standalone HTML report

### Known limits

Stated plainly, because a monitoring tool that oversells itself is worse than
none:

- **The default embedder is lexical, not semantic.** A hashed bag of tokens
  catches garbled or rewritten output; it will not catch a fluent paraphrase
  that means something different in the same vocabulary. Pass any object with
  `.embed(text) -> list[float]` to fix that.
- **Aggregate signals need >= 10 runs** after the baseline before they say
  anything. Low-traffic agents get per-run signals only.
- **A baseline learned from broken behaviour treats broken as normal.** Build
  it from runs you have actually looked at.
- **The LLM extraction path in the example is tested with an injected fake
  model**, not against a live API. Retry and error handling are covered;
  real-model output quality is not.
- **One `QuietfailMiddleware` instance tracks one run at a time.** LangGraph
  rebuilds state between nodes and may run them on different threads, so
  neither state identity nor a ContextVar can correlate a run — the instance
  is the run scope. Call `.for_run()` per concurrent invocation.

## The reference agent

`examples/invoice_agent` is a real AP invoice exception resolver, not a toy. It exists so quietfail has something realistic to watch, and it demonstrates the LangGraph patterns worth copying:

- **Deterministic validation, not LLM validation** — totals arithmetic, duplicate detection and PO existence are exact operations. Models do judgment; code does arithmetic.
- **A genuine cycle** — proposed fixes flow back through validation (`apply_resolution → validate`), so a partially fixed invoice can never post. This is why you reach for LangGraph over a chain.
- **A bounded loop** — resolution rounds are capped. An unbounded retry loop is the documented shape of a $4,200 incident.
- **Confidence-gated human-in-the-loop** — `interrupt()` fires only for low-confidence or never-auto-resolvable classes, and the payload leads with *why this one is unusual*. Approval fatigue is a documented antipattern: a human rubber-stamping 73 actions a day is not a control.
- **Reducers chosen deliberately** — `resolutions` accumulates (it's the audit trail); `findings` is replaced (validation re-runs, so accumulating stale errors prevents convergence).

```bash
poetry install
poetry run python -m invoice_agent
```

```
======================================================================
unknown-vendor-003
----------------------------------------------------------------------
  [error] missing_gl_code: 1 line item(s) missing GL code: ['Subscription, Q3']
  class      : coding_incomplete
  attempt    : vendor_lookup (confidence 1.0) -> {'vendor_id': 'V-1003', ...}
  attempt    : gl_code_history (confidence 0.333) -> {'Subscription, Q3': '6450'}
  rounds     : 1
  --> PAUSED at human_review: coding_incomplete
      resumed -> posted
```

Round 1 resolves the vendor; re-validation then surfaces the GL gap rather than posting a half-fixed invoice.

All fixtures are synthetic. No proprietary data.

## Install

```bash
poetry install                          # runtime + dev
poetry install --with llm,postgres      # model-backed extraction + Postgres store
poetry run pytest
poetry run ruff check .
poetry run python -m invoice_agent.inject    # the acceptance test
```

By default the reference agent runs offline against deterministic JSON fixtures — no API key needed. Set `IEA_EXTRACTOR=llm` (plus `OPENAI_API_KEY`) to run extraction against a real model over the matching `.txt` invoice documents.

Useful while learning the graph:

```bash
poetry run python -m invoice_agent               # all fixtures, summarised
poetry run python -m invoice_agent.trace unknown-vendor-003   # state, node by node
```

## Why not just use LangSmith / Langfuse / Arize?

Use them — quietfail is not a replacement. They answer *"what did this run do?"* extremely well. quietfail answers a different question: *"did anything change, and should someone be woken up?"* — without you having to build a labelled eval set first. They are complementary: trace there, alert here.

## Design notes

Design decisions, the bugs found while building, and the questions this
codebase should let you answer are in `DESIGN_NOTES.md`.

## License

MIT

## Contributing

Issues and PRs welcome, especially failure modes you've hit in production that the signal set above would miss.
