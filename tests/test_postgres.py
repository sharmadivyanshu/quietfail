"""PostgresStore against a real server.

Skipped unless QUIETFAIL_TEST_DSN points at a reachable Postgres. CI sets it
via a service container; locally, export it or these are skipped — never
silently "passed".
"""

import os

import pytest

pytest.importorskip("psycopg", reason="PostgresStore needs psycopg>=3")

DSN = os.getenv("QUIETFAIL_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set QUIETFAIL_TEST_DSN to run these")

from quietfail.baseline import build_profile, evaluate_run  # noqa: E402
from quietfail.postgres import PostgresStore  # noqa: E402
from quietfail.store import Alert, RunRecord, ToolEvent  # noqa: E402


@pytest.fixture
def store():
    import psycopg

    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS tool_events, alerts, runs, baselines CASCADE")
    return PostgresStore(DSN)


def make_record(run_id: str = "r1", **kw) -> RunRecord:
    record = RunRecord(
        agent="pg_agent",
        run_id=run_id,
        node_path=kw.pop("node_path", ["ingest", "validate", "post"]),
        output_keys=kw.pop("output_keys", ["extracted", "outcome"]),
        outcome=kw.pop("outcome", "posted"),
        status="ok",
        duration_ms=12,
        **kw,
    )
    record.tool_events.append(ToolEvent("lookup", empty=False, errored=False, latency_ms=1.5))
    return record


def test_run_roundtrip(store):
    store.save_run(make_record())
    runs = store.runs("pg_agent")
    assert len(runs) == 1
    assert runs[0]["node_path"] == ["ingest", "validate", "post"]
    assert runs[0]["outcome"] == "posted"
    assert runs[0]["tool_events"][0]["tool"] == "lookup"


def test_save_run_is_idempotent(store):
    """Re-saving must not duplicate the tool history."""
    record = make_record()
    store.save_run(record)
    store.save_run(record)
    runs = store.runs("pg_agent")
    assert len(runs) == 1
    assert len(runs[0]["tool_events"]) == 1


def test_timestamps_come_back_as_iso_strings(store):
    """The rest of quietfail compares started_at as a string, so the Postgres
    store must not leak datetimes through the same field."""
    store.save_run(make_record())
    started = store.runs("pg_agent")[0]["started_at"]
    assert isinstance(started, str)
    assert started > "2020-01-01"


def test_baseline_roundtrip(store):
    for i in range(12):
        store.save_run(make_record(run_id=f"r{i}", output_text="acme paper toner"))
    profile = build_profile(store.runs("pg_agent"))
    store.save_baseline("pg_agent", 12, profile)

    loaded = store.load_baseline("pg_agent")
    assert loaded["run_count"] == 12
    assert "_built_at" in loaded
    assert loaded["node_paths"] == profile["node_paths"]


def test_baseline_upsert_replaces(store):
    store.save_run(make_record())
    profile = build_profile(store.runs("pg_agent"))
    store.save_baseline("pg_agent", 1, profile)
    store.save_baseline("pg_agent", 99, profile)
    assert store.load_baseline("pg_agent")["run_count"] == 1  # profile is the source


def test_alerts_roundtrip_and_dedup_keys(store):
    store.save_alerts(
        [
            Alert(signal="s1", severity="critical", summary="sum1", detail="d", scope="run"),
            Alert(signal="s2", severity="warn", summary="sum2", detail="d", scope="aggregate"),
        ]
    )
    assert len(store.alerts()) == 2
    assert ("s1", "sum1") in store.recent_alert_keys()


def test_recent_runs_filters_by_since(store):
    for i in range(5):
        store.save_run(make_record(run_id=f"r{i}"))
    all_runs = store.runs("pg_agent")
    cutoff = all_runs[2]["started_at"]
    assert len(store.recent_runs("pg_agent", 10, since=cutoff)) == 2


def test_run_count(store):
    for i in range(3):
        store.save_run(make_record(run_id=f"r{i}"))
    assert store.run_count("pg_agent") == 3
    assert store.run_count("other") == 0


def test_evaluation_works_off_the_postgres_store(store):
    """The real contract: rows from Postgres must be shaped exactly like rows
    from SQLite, because the evaluators do not know which one they came from."""
    for i in range(12):
        store.save_run(make_record(run_id=f"r{i}"))
    profile = build_profile(store.runs("pg_agent"))
    store.save_baseline("pg_agent", 12, profile)

    store.save_run(make_record(run_id="odd", node_path=["ingest", "surprise", "post"]))
    odd = store.runs("pg_agent")[-1]
    signals = {a.signal for a in evaluate_run(odd, store.load_baseline("pg_agent"))}
    assert "trajectory.unseen_path" in signals


def test_watcher_accepts_the_postgres_store(store):
    """Duck-typing check: Watcher only needs the Store interface."""
    from quietfail import Watcher

    watcher = Watcher(store, agent="pg_agent")
    assert watcher.store is store
