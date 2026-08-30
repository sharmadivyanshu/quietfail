"""Postgres-backed run store, for when SQLite's single writer stops being enough.

Same interface as `Store`, so nothing else changes:

    from quietfail import Watcher
    from quietfail.postgres import PostgresStore

    store = PostgresStore("postgresql://user:pass@host/quietfail")
    watcher = Watcher(store, agent="my_agent")

Reach for this when several processes record runs concurrently. SQLite is the
right answer until then — one file, no server, and you can inspect it with any
SQL client.

Requires `psycopg>=3` — imported lazily so the core stays dependency-free.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from .store import Alert, RunRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    agent       TEXT NOT NULL,
    thread_id   TEXT,
    started_at  TIMESTAMPTZ NOT NULL,
    ended_at    TIMESTAMPTZ,
    status      TEXT NOT NULL,
    outcome     TEXT,
    node_path   JSONB NOT NULL,
    output_keys JSONB NOT NULL,
    step_count  INTEGER NOT NULL,
    duration_ms INTEGER,
    tokens      INTEGER NOT NULL DEFAULT 0,
    usd         DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    tool_calls  INTEGER NOT NULL DEFAULT 0,
    output_text TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS tool_events (
    id         BIGSERIAL PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    tool       TEXT NOT NULL,
    empty      BOOLEAN NOT NULL,
    errored    BOOLEAN NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id        BIGSERIAL PRIMARY KEY,
    run_id    TEXT,
    raised_at TIMESTAMPTZ NOT NULL,
    signal    TEXT NOT NULL,
    severity  TEXT NOT NULL,
    summary   TEXT NOT NULL,
    detail    TEXT NOT NULL,
    scope     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS baselines (
    agent      TEXT PRIMARY KEY,
    built_at   TIMESTAMPTZ NOT NULL,
    run_count  INTEGER NOT NULL,
    profile    JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent, started_at);
CREATE INDEX IF NOT EXISTS idx_tool_events_run ON tool_events(run_id);
"""


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else value


class PostgresStore:
    """Drop-in replacement for `Store`, backed by Postgres."""

    def __init__(self, dsn: str, *, schema: bool = True):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - only without psycopg
            raise ImportError(
                "PostgresStore needs psycopg>=3 — pip install 'psycopg[binary]'"
            ) from exc
        self._psycopg = psycopg
        self.dsn = dsn
        if schema:
            with self._conn() as conn:
                conn.execute(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator:
        conn = self._psycopg.connect(self.dsn, autocommit=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- writes ------------------------------------------------------------

    def save_run(self, run: RunRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO runs
                   (run_id, agent, thread_id, started_at, ended_at, status, outcome,
                    node_path, output_keys, step_count, duration_ms, tokens, usd,
                    tool_calls, output_text, error)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (run_id) DO UPDATE SET
                     ended_at = EXCLUDED.ended_at, status = EXCLUDED.status,
                     outcome = EXCLUDED.outcome, node_path = EXCLUDED.node_path,
                     output_keys = EXCLUDED.output_keys, step_count = EXCLUDED.step_count,
                     duration_ms = EXCLUDED.duration_ms, tokens = EXCLUDED.tokens,
                     usd = EXCLUDED.usd, tool_calls = EXCLUDED.tool_calls,
                     output_text = EXCLUDED.output_text, error = EXCLUDED.error""",
                (
                    run.run_id,
                    run.agent,
                    run.thread_id,
                    run.started_at,
                    run.ended_at,
                    run.status,
                    run.outcome,
                    json.dumps(run.node_path),
                    json.dumps(sorted(run.output_keys)),
                    run.step_count,
                    run.duration_ms,
                    run.tokens,
                    run.usd,
                    len(run.tool_events),
                    run.output_text,
                    run.error,
                ),
            )
            # Replace rather than append, so a re-saved run does not double its
            # tool history — save_run is idempotent by contract.
            conn.execute("DELETE FROM tool_events WHERE run_id = %s", (run.run_id,))
            for event in run.tool_events:
                conn.execute(
                    "INSERT INTO tool_events (run_id, tool, empty, errored, latency_ms)"
                    " VALUES (%s,%s,%s,%s,%s)",
                    (run.run_id, event.tool, event.empty, event.errored, event.latency_ms),
                )

    def save_alerts(self, alerts: list[Alert]) -> None:
        if not alerts:
            return
        with self._conn() as conn:
            for alert in alerts:
                conn.execute(
                    "INSERT INTO alerts (run_id, raised_at, signal, severity, summary,"
                    " detail, scope) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (
                        alert.run_id,
                        _now(),
                        alert.signal,
                        alert.severity,
                        alert.summary,
                        alert.detail,
                        alert.scope,
                    ),
                )

    def save_baseline(self, agent: str, run_count: int, profile: dict) -> None:
        profile = {**profile, "_built_at": _now().isoformat()}
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO baselines (agent, built_at, run_count, profile)
                   VALUES (%s,%s,%s,%s)
                   ON CONFLICT (agent) DO UPDATE SET
                     built_at = EXCLUDED.built_at, run_count = EXCLUDED.run_count,
                     profile = EXCLUDED.profile""",
                (agent, _now(), run_count, json.dumps(profile)),
            )

    # ---- reads -------------------------------------------------------------

    def runs(self, agent: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM runs WHERE agent = %s ORDER BY started_at"
        params: tuple = (agent,)
        if limit:
            sql += " LIMIT %s"
            params = (agent, limit)
        with self._conn() as conn:
            cur = conn.execute(sql, params)
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, r, strict=True)) for r in cur.fetchall()]
            for row in rows:
                row["started_at"] = _iso(row["started_at"])
                row["ended_at"] = _iso(row["ended_at"])
                row["tool_events"] = [
                    {"tool": t, "empty": int(e), "errored": int(x), "latency_ms": ms}
                    for t, e, x, ms in conn.execute(
                        "SELECT tool, empty, errored, latency_ms FROM tool_events"
                        " WHERE run_id = %s ORDER BY id",
                        (row["run_id"],),
                    ).fetchall()
                ]
        return rows

    def recent_runs(self, agent: str, window: int, since: str | None = None) -> list[dict]:
        runs = self.runs(agent)
        if since:
            runs = [r for r in runs if r["started_at"] > since]
        return runs[-window:]

    def load_baseline(self, agent: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT profile FROM baselines WHERE agent = %s", (agent,)
            ).fetchone()
        return row[0] if row else None

    def alerts(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT %s", (limit,))
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, r, strict=True)) for r in cur.fetchall()]
        for row in rows:
            row["raised_at"] = _iso(row["raised_at"])
        return rows

    def recent_alert_keys(self, limit: int = 50) -> set[tuple[str, str]]:
        with self._conn() as conn:
            return {
                (signal, summary)
                for signal, summary in conn.execute(
                    "SELECT signal, summary FROM alerts ORDER BY id DESC LIMIT %s", (limit,)
                ).fetchall()
            }

    def run_count(self, agent: str) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM runs WHERE agent = %s", (agent,)).fetchone()[
                0
            ]


__all__ = ["PostgresStore"]
