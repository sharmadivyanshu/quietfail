# Design notes

The build in the order it happened, with the reasoning and the things that
went wrong. Read this top to bottom and the codebase stops being a maze.

If you only read one section, read **Bugs found and what they taught** — those
are the parts worth talking about out loud.

---

## Phase 1 — the reference agent

**Goal:** something realistic for the detector to watch. A toy agent would
make the detector look good against failures that don't happen in production.

**Files:** `state.py`, `rules.py`, `store.py`, `tools.py`, `extract.py`,
`nodes.py`, `graph.py`

### Decisions

**Validation is deterministic code, not an LLM call.** Totals arithmetic,
duplicate detection and PO existence are exact operations. A model would be
slower, costlier, non-reproducible, and untestable. The rule: *models do
judgment, code does arithmetic.*

**Exception classification is separate from resolution.** Classifying first
means the resolver sees 2–3 tools instead of 15. Reducing available tools
measurably improves function-calling accuracy (arXiv 2411.15399).

**Resolution flows back through validation.** `resolve → apply_resolution →
validate` is a cycle, not a DAG. Fixing one error often reveals the next, and
without the cycle a partially fixed invoice posts.

**The loop is bounded.** `MAX_RESOLUTION_ROUNDS = 3`. An unbounded retry loop
is the documented shape of an agent that burned $4,200 in 63 hours.

**HITL is confidence-gated, and the payload explains itself.** A human
approving 73 actions a day stops reading them — approval fatigue is a
documented antipattern, and an approval gate nobody reads is not a control.
The interrupt payload leads with *why this one is unusual*.

---

## Phase 2 — the detection layer

**Goal:** catch failures without labelled data, because teams don't have it.

**Files:** `store.py`, `budget.py`, `baseline.py`, `watch.py`, `report.py`, `cli.py`

### Decisions

**Collection via `graph.stream()`, not monkeypatching.** Public, supported API
surface. Anything that runs on LangGraph can be watched with no graph changes.

**Statistics, not ML.** Frequency tables, percentile bands, observed-path sets.
Every alert must be explainable in one sentence. A drift score of 0.87 tells an
on-call engineer nothing.

**A pause is a terminal outcome.** A run parked at `human_review` records
`awaiting_human`. That single choice turns "escalation rate" into a
first-class signal — which is how the removed-approval-gate injection gets
caught.

**Budgets raise, they don't warn.** A budget that logs while the loop keeps
spending is not a budget.

**Empty tool results are tracked separately from errors.** A tool returning
`[]` instead of raising is the most common silent failure, and it is invisible
to error-rate monitoring.

**The default embedder is lexical, and the README says so.** A hashed bag of
tokens keeps the core offline and dependency-free, and catches the common case
(garbled or rewritten output). It cannot catch a fluent paraphrase that means
something different. Calling it "semantic drift" would have been the easy lie;
the interface takes any `.embed(text) -> list[float]` so a real embedder drops
straight in.

**Distance needs a floor, not just a sigma threshold.** An agent whose outputs
are near-identical has a standard deviation near zero, which makes trivial
wording noise look like a twelve-sigma event. `MIN_DISTANCE` is what stops the
content signal screaming on a well-behaved agent.

---

## Phase 3 — reach

**Goal:** work with the agent style most people actually start from, and with
more than one process.

**Files:** `middleware.py`, `postgres.py`

### Decisions

**The middleware records the same shape as the graph collector.** A
`create_agent` run has no graph nodes, so the trajectory becomes the
model/tool call sequence (`model`, `tool:lookup`, `model`). Different content,
same *meaning* — "what route did this run take" — which is why the identical
evaluators, baselines and CLI work across both.

**Postgres mirrors the SQLite interface exactly, including the awkward bits.**
`started_at` comes back as an ISO **string**, not a `datetime`, because
`recent_runs` compares it to the baseline's `_built_at` string. Returning the
"better" type would have broken drift evaluation silently — a nice example of
an interface contract living in a field's *type*, not just its name.

**`save_run` deletes tool events before reinserting.** Re-saving a run must
not double its tool history; idempotency is part of the contract.

**Both are lazy imports behind optional dependency groups.** The core installs
with no langchain and no psycopg.

---

## Bugs found and what they taught

These are real, in build order. Each one is a better interview answer than any
feature.

### 1. The reducer was wrong once validation re-ran

`findings` was `Annotated[list[ValidationFinding], add]`. Once `validate`
started running twice, fixed errors accumulated forever and the graph never
converged.

**Fix:** `findings` is replaced; `resolutions` still accumulates because that
one *is* an audit trail.

**Lesson:** a reducer encodes merge semantics, not convenience. Ask "should
this replace or accumulate?" per field — and the answer changes the moment a
node can run more than once.

### 2. A missing edge that only fired on non-convergence

`route_after_validation` could return `"human_review"` when the round cap was
hit, but the conditional-edge map only listed `classify_exception` and `post`.
`KeyError: 'human_review'` — on a path the unit tests never took.

**Fix:** add `human_review` to the map.

**Lesson:** conditional edges have two places to keep in sync — the routing
function and the destination map — and only integration paths catch the drift.
**19 unit tests did not find this. The failure-injection harness found it in
one run.** That is the project's own thesis proving itself.

### 3. Aggregate signals fired on a window of one

The first run after a baseline compared a 1-run window to the baseline
distribution and alerted every time. `awaiting_human` was "0% vs 80%" —
because there had been exactly one run and it happened to post.

**Fix:** `MIN_WINDOW_RUNS = 10`; below that, aggregate signals stay silent.

**Lesson:** false positives are how a monitor gets muted, and a muted monitor
is worse than none. This is why `inject.py` has a control scenario — proving
it stays quiet matters as much as proving it catches things.

### 4. The aggregate window was diluted by baseline runs

`recent_runs` returned the last N runs including the ones the baseline had
already absorbed, so a genuine shift was averaged away by its own baseline.

**Fix:** stamp `_built_at` into the profile; evaluate only runs after it.

### 5. The same alert repeated on every run

An aggregate condition stays true while the window is dirty, so the same
finding was re-raised 15 times — with a run count in the summary that made it
look different each time and defeated de-duplication.

**Fix:** volatile numbers moved out of the summary (the dedup key) into the
detail, plus cross-run suppression via `recent_alert_keys`.

**Lesson:** an alerting tool that spams is self-defeating, and this one is
*about* alert fatigue.

### 6. The LLM extraction path was documented but impossible

`LLMExtractor` read `.txt` documents. Only `.json` fixtures existed. The README
told you to set `IEA_EXTRACTOR=llm`, and doing so raised `FileNotFoundError`
immediately — a claim that had never once been executed.

**Fix:** added the `.txt` invoice documents, made the model injectable so the
retry logic is testable without an API key, and added
`test_every_json_fixture_has_document_text` so the gap cannot reopen.

**Lesson:** a README is a promise. Anything in it that no test exercises is a
guess. The regression test here asserts a *relationship between files*, which
is the kind of test people skip and then regret.

### 7. An injection that tested the wrong thing

"Approval gate removed" originally lowered `CONFIDENCE_THRESHOLD` to 0.0 and
expected escalations to collapse. They didn't — the round cap still forced
unresolvable invoices to a human. The detector was right; the expectation was
wrong.

**Fix:** the injection now patches `human_review` to auto-approve, which is
what "the gate is gone" actually means.

**Lesson:** when a test fails, confirm which side is wrong before fixing.

### 8. Nothing could correlate a `create_agent` run

The obvious design keyed the in-flight run by `id(state)` in `before_agent`
and looked it up again in `after_agent`. Every run recorded zero steps and no
run was ever saved: **LangGraph hands each node a freshly built state dict**,
so the identity is gone by the next hook.

A `ContextVar` — the fix used in `watch.py` for tools — is no better here,
because nodes may execute on different threads and would each see a fresh
context.

**Fix:** the middleware instance *is* the run scope, with `.for_run()` to make
a fresh one per concurrent invocation. The limitation is documented rather
than hidden.

**Lesson:** correlating work across an async/threaded framework needs a scope
the framework guarantees. Object identity and thread identity are both
assumptions, and both were wrong here. Worth knowing which one your framework
actually preserves before designing around it.

---

## Questions to be able to answer

If you can answer these without notes, you own this code.
Try answering each out loud.

**On the agent**
1. Why is validation not an LLM call? What would break if it were?
2. When is `Annotated[list, add]` the wrong reducer? (Bug 1.)
3. What makes this a graph rather than a chain? Point at the cycle.
4. What happens if the process dies between `resolve` and `apply_resolution`?
5. Why is `arithmetic_mismatch` never auto-resolved, even at high confidence?
6. What is `thread_id` for, and what breaks if two invoices share one?

**On the detector**
7. How do you detect a wrong answer without knowing the right one?
8. Why is a paused run recorded as an outcome rather than an incomplete run?
9. Why `MIN_WINDOW_RUNS`? What happened without it? (Bug 3.)
10. Why exclude baseline runs from the drift window? (Bug 4.)
11. Where would this miss a failure? (Honest answer: a fluent paraphrase that
    means something different in the same vocabulary. Structure is normal, and
    a lexical embedder scores it as normal too. That needs a real embedder —
    the interface accepts one, the default isn't one.)
12. Why statistics rather than an anomaly-detection model?
13. Why does the content signal need a distance *floor* as well as a sigma
    threshold? (What happens when std is 0?)
14. A baseline built from already-broken runs learns broken as normal. How
    would you detect that?

**On the whole thing**
15. What does this catch that LangSmith doesn't, and vice versa?
16. What would you build next, and why that rather than something else?

Question 11 is the one to answer honestly. Knowing where your own tool is
blind is a stronger signal than claiming it catches everything.

---

## What's deliberately not built

- **A bundled semantic embedder** — the interface accepts one; shipping a
  default would drag in a model dependency and a network call for every run.
- **A hosted dashboard** — every incumbent owns the dashboard. The mechanism
  is the differentiator; the output is one HTML file.
- **Async hooks** (`abefore_agent` and friends) — the sync path is covered;
  async is a straight port and an honest "not yet".
- **Sampling** — every run is recorded. At high volume you would want to
  record a fraction and keep aggregates exact, which changes the store schema.
