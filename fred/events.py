"""The event table: the structured output that plain-language questions compile into.

This is the part the provided stack does not give us. The index stores one free-text
caption per 5-second segment; we store one row per *verified football event*, with the
evidence that produced it. A question like "crosses that led to a goal" becomes a filter
over this table, not a similarity search.

Primary store is VastDB, so the rows live next to the vectors on the sponsor platform.
SQLite mirrors every write so the demo, the eval and the UI keep working if the cluster
is unreachable - which on a shared hackathon cluster is a when, not an if.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field

from . import config

LOCAL_DB = os.environ.get("FRED_DB", "data/out/fred_events.sqlite")
VASTDB_TABLE = os.environ.get("FRED_VASTDB_TABLE", "fred-events")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    match           TEXT NOT NULL,
    source_video    TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    t_anchor        REAL NOT NULL,
    window_start    REAL NOT NULL,
    window_end      REAL NOT NULL,
    goal_confirmed  INTEGER DEFAULT 1,
    assist_type     TEXT,
    confidence      REAL,
    anchor_sources  TEXT,
    anchor_score    REAL,
    cosmos_letter   TEXT,
    cosmos_reasoning TEXT,
    cosmos_model    TEXT,
    cosmos_ok       INTEGER,
    cosmos_error    TEXT,
    clip_path       TEXT,
    clip_s3_uri     TEXT,
    asr_text        TEXT,
    detections      TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_assist ON events(assist_type);
CREATE INDEX IF NOT EXISTS idx_events_confirmed ON events(goal_confirmed);
CREATE INDEX IF NOT EXISTS idx_events_match ON events(match);
"""

COLUMNS = [
    "event_id", "match", "source_video", "event_type", "t_anchor",
    "window_start", "window_end", "goal_confirmed", "assist_type", "confidence",
    "anchor_sources", "anchor_score", "cosmos_letter", "cosmos_reasoning",
    "cosmos_model", "cosmos_ok", "cosmos_error", "clip_path", "clip_s3_uri",
    "asr_text", "detections",
]


@dataclass
class Event:
    event_id: str
    match: str
    source_video: str
    event_type: str = "goal"
    t_anchor: float = 0.0
    window_start: float = 0.0
    window_end: float = 0.0
    goal_confirmed: int = 1
    assist_type: str = ""
    confidence: float = 0.0
    anchor_sources: str = ""
    anchor_score: float = 0.0
    cosmos_letter: str = ""
    cosmos_reasoning: str = ""
    cosmos_model: str = ""
    cosmos_ok: int = 1
    cosmos_error: str = ""
    clip_path: str = ""
    clip_s3_uri: str = ""
    asr_text: str = ""
    detections: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def clock(self) -> str:
        minutes, seconds = divmod(int(self.t_anchor), 60)
        return f"{minutes:02d}:{seconds:02d}"


def connect(path: str = None) -> sqlite3.Connection:
    path = path or LOCAL_DB
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def upsert(events, path: str = None) -> int:
    """Write rows locally. Idempotent on event_id, so re-running a match is safe."""
    if isinstance(events, Event):
        events = [events]
    conn = connect(path)
    placeholders = ",".join("?" for _ in COLUMNS)
    with conn:
        for event in events:
            data = event.as_dict() if isinstance(event, Event) else dict(event)
            conn.execute(
                f"INSERT OR REPLACE INTO events ({','.join(COLUMNS)}) VALUES ({placeholders})",
                [data.get(column) for column in COLUMNS],
            )
    count = len(events)
    conn.close()
    return count


def query(where: str = "", params=(), order: str = "match, t_anchor",
          limit: int = 200, path: str = None) -> list:
    conn = connect(path)
    sql = "SELECT * FROM events"
    if where:
        sql += f" WHERE {where}"
    sql += f" ORDER BY {order} LIMIT {int(limit)}"
    rows = [dict(row) for row in conn.execute(sql, params)]
    conn.close()
    return rows


def stats(path: str = None) -> dict:
    conn = connect(path)
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    confirmed = conn.execute(
        "SELECT COUNT(*) FROM events WHERE goal_confirmed = 1").fetchone()[0]
    rejected = total - confirmed
    by_assist = {
        row[0] or "unknown": row[1]
        for row in conn.execute(
            "SELECT assist_type, COUNT(*) FROM events WHERE goal_confirmed = 1 "
            "GROUP BY assist_type ORDER BY 2 DESC"
        )
    }
    by_match = {
        row[0]: row[1]
        for row in conn.execute("SELECT match, COUNT(*) FROM events GROUP BY match")
    }
    conn.close()
    return {"total_events": total, "confirmed_goals": confirmed,
            "rejected_by_cosmos": rejected,
            "by_assist_type": by_assist, "by_match": by_match}


# --------------------------------------------------------------------------- VastDB

def _vastdb_session():
    import vastdb  # imported lazily: absent off-VM

    endpoint = config.get("VDB_ENDPOINT") or config.require("S3_ENDPOINT")
    return vastdb.connect(
        endpoint=endpoint,
        access=config.require("ACCESS_KEY"),
        secret=config.require("SECRET_KEY"),
        ssl_verify=False,
    )


def push_to_vastdb(events, table_name: str = None) -> dict:
    """Write the event table into VastDB alongside the pipeline's own collection.

    Deliberately contains no vector columns: selecting fixed-size list columns needs the
    pipeline's vastdb patch, and our rows are filtered, not searched by similarity.
    """
    table_name = table_name or VASTDB_TABLE
    try:
        import pyarrow as pa

        rows = [e.as_dict() if isinstance(e, Event) else dict(e) for e in events]
        if not rows:
            return {"ok": True, "written": 0, "note": "nothing to write"}

        schema = pa.schema([
            pa.field("event_id", pa.string()), pa.field("match", pa.string()),
            pa.field("source_video", pa.string()), pa.field("event_type", pa.string()),
            pa.field("t_anchor", pa.float64()), pa.field("window_start", pa.float64()),
            pa.field("window_end", pa.float64()), pa.field("goal_confirmed", pa.int32()),
            pa.field("assist_type", pa.string()),
            pa.field("confidence", pa.float64()), pa.field("anchor_sources", pa.string()),
            pa.field("anchor_score", pa.float64()), pa.field("cosmos_letter", pa.string()),
            pa.field("cosmos_reasoning", pa.string()), pa.field("cosmos_model", pa.string()),
            pa.field("cosmos_ok", pa.int32()), pa.field("cosmos_error", pa.string()),
            pa.field("clip_path", pa.string()), pa.field("clip_s3_uri", pa.string()),
            pa.field("asr_text", pa.string()), pa.field("detections", pa.string()),
        ])
        batch = pa.RecordBatch.from_pylist(
            [{f.name: row.get(f.name) for f in schema} for row in rows], schema=schema
        )

        bucket_name = config.require("VASTDB_BUCKET")
        schema_name = config.get("VDB_SCHEMA") or "vss-schema"
        session = _vastdb_session()
        with session.transaction() as tx:
            bucket = tx.bucket(bucket_name)
            try:
                db_schema = bucket.schema(schema_name)
            except Exception:
                db_schema = bucket.create_schema(schema_name)
            try:
                table = db_schema.table(table_name)
            except Exception:
                table = db_schema.create_table(table_name, schema)
            table.insert(batch)
        return {"ok": True, "written": len(rows),
                "table": f"{bucket_name}/{schema_name}/{table_name}"}
    except Exception as exc:  # noqa: BLE001 - VastDB is best-effort; SQLite is the truth
        return {"ok": False, "written": 0, "error": f"{type(exc).__name__}: {exc}"}


def export_jsonl(path: str = "data/out/events.jsonl", db_path: str = None) -> str:
    rows = query(limit=10000, path=db_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path
