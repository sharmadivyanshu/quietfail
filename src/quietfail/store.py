"""Run store. SQLite by default — zero config, no server, easy to inspect.

Schema is deliberately flat and readable. You should be able to answer
"what did my agent do yesterday?" with a hand-written SQL query.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    agent       TEXT NOT NULL,
    thread_id   TEXT,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    status      TEXT NOT NULL,
    outcome     TEXT,
    node_path   TEXT NOT NULL,
    output_keys TEXT NOT NULL,
    step_count  INTEGER NOT NULL,
    duration_ms INTEGER,
    tokens      INTEGER NOT NULL DEFAULT 0,
    usd         REAL    NOT NULL DEFAULT 0.0,
    tool_calls  INTEGER NOT NULL DEFAULT 0,
    output_text TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS tool_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id     TEXT NOT NULL REFERENCES runs(run_id),
    tool       TEXT NOT NULL,
    empty      INTEGER NOT NULL,
    errored    INTEGER NOT NULL,
    latency_ms REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT,
    raised_at TEXT NOT NULL,
    signal    TEXT NOT NULL,
    severity  TEXT NOT NULL,
    summary   TEXT NOT NULL,
    detail    TEXT NOT NULL,
    scope     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS baselines (
    agent      TEXT PRIMARY KEY,
    built_at   TEXT NOT NULL,
    run_count  INTEGER NOT NULL,
    profile    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent, started_at);
CREATE INDEX IF NOT EXISTS idx_tool_events_run ON tool_events(run_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class ToolEvent:
    tool: str
    empty: bool
    errored: bool
    latency_ms: float


@dataclass
class RunRecord:
    """Everything quietfail knows about one agent run."""

    agent: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    thread_id: str | None = None
    started_at: str = field(default_factory=_now)
    ended_at: str | None = None
    status: str = "running"  # running | ok | error
    outcome: str | None = None  # domain terminal state, e.g. "posted"
    node_path: list[str] = field(default_factory=list)
    output_keys: list[str] = field(default_factory=list)
    duration_ms: int | None = None
    tokens: int = 0
    usd: float = 0.0
    tool_events: list[ToolEvent] = field(default_factory=list)
    output_text: str | None = None
    error: str | None = None

    @property
    def step_count(self) -> int:
        return len(self.node_path)

    def node_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.node_path:
            counts[node] = counts.get(node, 0) + 1
        return counts

    def as_dict(self) -> dict:
        data = asdict(self)
        data["step_count"] = self.step_count
        return data


@dataclass
class Alert:
    signal: str
    severity: str  # info | warn | critical
    summary: str
    detail: str
    scope: str  # run | aggregate
    run_id: str | None = None

    def __str__(self) -> str:
        return f"[{self.severity}] {self.signal}: {self.summary}"


class Store:
    def __init__(self, path: str | Path = "quietfail.sqlite"):
        self.path = str(path)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """Additive migrations for stores created by an earlier version.

        CREATE TABLE IF NOT EXISTS silently does nothing on an existing table,
        so new columns have to be added explicitly.
        """
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(runs)")}
        for column, ddl in (("output_text", "TEXT"),):
            if column not in existing:
                conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {ddl}")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- writes ------------------------------------------------------------

    def save_run(self, run: RunRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO runs
                   (run_id, agent, thread_id, started_at, ended_at, status, outcome,
                    node_path, output_keys, step_count, duration_ms, tokens, usd,
                    tool_calls, output_text, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
            conn.executemany(
                "INSERT INTO tool_events (run_id, tool, empty, errored, latency_ms)"
                " VALUES (?,?,?,?,?)",
                [
                    (run.run_id, e.tool, int(e.empty), int(e.errored), e.latency_ms)
                    for e in run.tool_events
                ],
            )

    def save_alerts(self, alerts: list[Alert]) -> None:
        if not alerts:
            return
        with self._conn() as conn:
            conn.executemany(
                "INSERT INTO alerts (run_id, raised_at, signal, severity, summary, detail, scope)"
                " VALUES (?,?,?,?,?,?,?)",
                [
                    (a.run_id, _now(), a.signal, a.severity, a.summary, a.detail, a.scope)
                    for a in alerts
                ],
            )

    def save_baseline(self, agent: str, run_count: int, profile: dict) -> None:
        # Stamp the profile so evaluation can exclude runs already absorbed.
        profile = {**profile, "_built_at": _now()}
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO baselines (agent, built_at, run_count, profile)"
                " VALUES (?,?,?,?)",
                (agent, _now(), run_count, json.dumps(profile)),
            )

    # ---- reads -------------------------------------------------------------

    def runs(self, agent: str, limit: int | None = None) -> list[dict]:
        sql = "SELECT * FROM runs WHERE agent = ? ORDER BY started_at"
        params: tuple = (agent,)
        if limit:
            sql += " LIMIT ?"
            params = (agent, limit)
        with self._conn() as conn:
            rows = [dict(r) for r in conn.execute(sql, params)]
        for row in rows:
            row["node_path"] = json.loads(row["node_path"])
            row["output_keys"] = json.loads(row["output_keys"])
            row["tool_events"] = self._tool_events(row["run_id"])
        return rows

    def recent_runs(self, agent: str, window: int, since: str | None = None) -> list[dict]:
        """Last `window` runs, optionally only those after `since`.

        Aggregate drift must be measured on runs the baseline has NOT already
        absorbed — otherwise the baseline dilutes the very signal you are
        looking for.
        """
        runs = self.runs(agent)
        if since:
            runs = [r for r in runs if r["started_at"] > since]
        return runs[-window:]

    def _tool_events(self, run_id: str) -> list[dict]:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT tool, empty, errored, latency_ms FROM tool_events WHERE run_id = ?",
                    (run_id,),
                )
            ]

    def load_baseline(self, agent: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT profile FROM baselines WHERE agent = ?", (agent,)).fetchone()
        return json.loads(row["profile"]) if row else None

    def alerts(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            return [
                dict(r)
                for r in conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,))
            ]

    def recent_alert_keys(self, limit: int = 50) -> set[tuple[str, str]]:
        """(signal, summary) pairs raised recently — used to suppress repeats.

        An aggregate condition stays true for as long as the window is dirty,
        so without this the same finding is re-raised on every subsequent run.
        """
        with self._conn() as conn:
            return {
                (r["signal"], r["summary"])
                for r in conn.execute(
                    "SELECT signal, summary FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
                )
            }

    def run_count(self, agent: str) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) c FROM runs WHERE agent = ?", (agent,)).fetchone()[
                "c"
            ]
