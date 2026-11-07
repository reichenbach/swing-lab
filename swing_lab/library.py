"""Per-player swing library (SQLite) — the basis for progress tracking.

Every analyzed swing can be recorded against a player so you can compare a kid
to their own baseline over a season ("separation is up 8 degrees since April").
This is the local-first seed of the team app: one file, no server. The same
schema ports straight to Postgres when this becomes multi-tenant.

DB defaults to output/swing_lab.db (gitignored). Stdlib sqlite3 only.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

DEFAULT_DB = "output/swing_lab.db"

# Scalar metrics worth trending over time. Speeds are the scale/fps-invariant
# *_norm versions (shoulder-widths/sec); raw px speeds are kept in the report
# JSON but not trended, because pixels aren't comparable across sessions.
SCALARS = [
    "contact_frame", "attack_angle_deg", "max_separation", "sequence_ok",
    "head_drift_ratio", "shoulder_tilt_deg", "bat_tracked", "ball_tracked",
    "bat_angle_at_contact_deg", "bat_speed_peak", "bat_speed_peak_norm",
    "hand_speed_peak_norm", "on_plane_ratio", "body_scale_px",
]


def scalars_from_metrics(m) -> dict:
    """Pull the trendable scalar fields out of a SwingMetrics."""
    out = {}
    for k in SCALARS:
        v = getattr(m, k)
        out[k] = int(v) if isinstance(v, bool) else v
    return out


def _connect(db: str) -> sqlite3.Connection:
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS swings (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            player     TEXT NOT NULL,
            video      TEXT,
            created_at REAL NOT NULL,
            metrics    TEXT NOT NULL,
            report     TEXT,
            out_dir    TEXT
        )
        """
    )
    # Migrate older DBs that predate the report/out_dir columns.
    cols = {r["name"] for r in con.execute("PRAGMA table_info(swings)")}
    for col in ("report", "out_dir"):
        if col not in cols:
            con.execute(f"ALTER TABLE swings ADD COLUMN {col} TEXT")
    return con


def record(player: str, video: str, metrics: dict, db: str = DEFAULT_DB,
           created_at: float | None = None, report: dict | None = None,
           out_dir: str | None = None) -> int:
    con = _connect(db)
    with con:
        cur = con.execute(
            "INSERT INTO swings (player, video, created_at, metrics, report, out_dir) "
            "VALUES (?,?,?,?,?,?)",
            (player, video, created_at or time.time(), json.dumps(metrics),
             json.dumps(report) if report is not None else None, out_dir),
        )
    return int(cur.lastrowid)


def _parse(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else {}
    if d.get("report"):
        d["report"] = json.loads(d["report"])
    return d


def get(swing_id: int, db: str = DEFAULT_DB) -> dict | None:
    con = _connect(db)
    row = con.execute("SELECT * FROM swings WHERE id = ?", (swing_id,)).fetchone()
    return _parse(row) if row else None


def recent(limit: int = 20, db: str = DEFAULT_DB) -> list[dict]:
    con = _connect(db)
    rows = con.execute(
        "SELECT * FROM swings ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [_parse(r) for r in rows]


def players(db: str = DEFAULT_DB) -> list[dict]:
    con = _connect(db)
    rows = con.execute(
        "SELECT player, COUNT(*) n, MAX(created_at) last "
        "FROM swings GROUP BY player ORDER BY player"
    ).fetchall()
    return [dict(r) for r in rows]


def _rows(player: str, db: str) -> list[dict]:
    con = _connect(db)
    rows = con.execute(
        "SELECT * FROM swings WHERE player = ? ORDER BY created_at ASC", (player,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = json.loads(d["metrics"])
        out.append(d)
    return out


def history(player: str, db: str = DEFAULT_DB) -> list[dict]:
    return _rows(player, db)


def baseline(player: str, db: str = DEFAULT_DB) -> dict | None:
    rows = _rows(player, db)
    return rows[0]["metrics"] if rows else None


def latest(player: str, db: str = DEFAULT_DB) -> dict | None:
    rows = _rows(player, db)
    return rows[-1]["metrics"] if rows else None


def count(player: str, db: str = DEFAULT_DB) -> int:
    return len(_rows(player, db))
