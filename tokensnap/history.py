"""Persistent request history for the web dashboard's charts.

The live stats file (``stats.py``) keeps running totals plus the last 50
requests - perfect for ``tokensnap status`` / ``monitor``, but it can't answer
"how many tokens did I save last Tuesday?". This module adds a small SQLite
database at ``~/.tokensnap/history.db`` that records one row per optimized
request, so the dashboard can draw per-day / per-week / per-month charts.

Design notes:
  * Writes are **fire-and-forget** and never raise - the proxy calls
    :func:`log_request` on its hot path (via ``stats.record_request``), so a
    locked or missing database must degrade to "no history", never a failed
    request.
  * The proxy process writes while a *separate* dashboard process reads, so we
    open a short-lived connection per call and enable WAL mode for concurrent
    cross-process access.
  * Uses only the stdlib (``sqlite3``) - Tokensnap gains no new dependency.
"""

import logging
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from tokensnap import config as config_mod

log = logging.getLogger("tokensnap.history")

# Kept as a module attribute so tests can redirect it (monkeypatch history.DB_FILE).
DB_FILE = config_mod.CONFIG_DIR / "history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp        REAL    NOT NULL,
    model            TEXT,
    est_tokens_in    INTEGER NOT NULL DEFAULT 0,
    real_tokens_in   INTEGER NOT NULL DEFAULT 0,
    real_tokens_out  INTEGER NOT NULL DEFAULT 0,
    cache_read       INTEGER NOT NULL DEFAULT 0,
    cache_write      INTEGER NOT NULL DEFAULT 0,
    saved            INTEGER NOT NULL DEFAULT 0,
    http_status      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests (timestamp);

CREATE TABLE IF NOT EXISTS daily_summary (
    date             TEXT    PRIMARY KEY,
    total_requests   INTEGER NOT NULL DEFAULT 0,
    total_est_in     INTEGER NOT NULL DEFAULT 0,
    total_real_in    INTEGER NOT NULL DEFAULT 0,
    total_real_out   INTEGER NOT NULL DEFAULT 0,
    total_saved      INTEGER NOT NULL DEFAULT 0
);
"""


def _connect() -> sqlite3.Connection:
    """Open a short-lived connection to the history DB, creating the file/dir.

    WAL + a small busy timeout let the proxy process write while a dashboard
    process reads the same file without either one erroring on a lock.
    """
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    """Create the tables if they don't exist yet. Safe to call repeatedly."""
    conn = _connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def log_request(
    model: Optional[str],
    est_tokens_in: int,
    real_tokens_in: int,
    real_tokens_out: int,
    cache_read: int,
    cache_write: int,
    saved: int,
    http_status: int,
    ts: Optional[float] = None,
) -> None:
    """Record one optimized request. Best-effort: never raises.

    Also folds the row into ``daily_summary`` for today so month/all-time
    aggregates stay cheap without a nightly job.
    """
    ts = time.time() if ts is None else ts
    day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)  # cheap CREATE IF NOT EXISTS; self-heals a fresh DB
            conn.execute(
                "INSERT INTO requests (timestamp, model, est_tokens_in, "
                "real_tokens_in, real_tokens_out, cache_read, cache_write, "
                "saved, http_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (ts, model or "?", int(est_tokens_in), int(real_tokens_in),
                 int(real_tokens_out), int(cache_read), int(cache_write),
                 int(saved), int(http_status)),
            )
            conn.execute(
                "INSERT INTO daily_summary (date, total_requests, total_est_in, "
                "total_real_in, total_real_out, total_saved) "
                "VALUES (?, 1, ?, ?, ?, ?) "
                "ON CONFLICT(date) DO UPDATE SET "
                "total_requests = total_requests + 1, "
                "total_est_in   = total_est_in + excluded.total_est_in, "
                "total_real_in  = total_real_in + excluded.total_real_in, "
                "total_real_out = total_real_out + excluded.total_real_out, "
                "total_saved    = total_saved + excluded.total_saved",
                (day, int(est_tokens_in), int(real_tokens_in),
                 int(real_tokens_out), int(saved)),
            )
            conn.commit()
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        # The proxy must survive a bad DB (locked, unwritable dir, disk full) -
        # degrade to "no history" rather than break the request.
        log.debug("history.log_request skipped (%s)", exc)


def _bucket_rows(period: str) -> List[Dict[str, Any]]:
    """Build the ordered, zero-filled bucket skeleton for a chart period.

    Returns dicts with a ``label`` and the SQLite ``strftime`` key that groups
    request rows into that bucket, so gaps (days with no traffic) still show.
    """
    today = date.today()
    if period == "week":
        # Last 8 ISO weeks, oldest first.
        buckets = []
        start = today - timedelta(days=today.weekday())  # Monday of this week
        for i in range(7, -1, -1):
            monday = start - timedelta(weeks=i)
            buckets.append({
                "label": monday.strftime("%b %d"),
                "key": monday.strftime("%Y-%W"),
            })
        return buckets
    if period == "month":
        # Last 6 calendar months, oldest first.
        buckets = []
        y, m = today.year, today.month
        seq = []
        for _ in range(6):
            seq.append((y, m))
            m -= 1
            if m == 0:
                m = 12
                y -= 1
        for yy, mm in reversed(seq):
            buckets.append({
                "label": date(yy, mm, 1).strftime("%b %Y"),
                "key": "%04d-%02d" % (yy, mm),
            })
        return buckets
    # Default: last 7 days, oldest first.
    buckets = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        buckets.append({"label": day.strftime("%a %d"), "key": day.strftime("%Y-%m-%d")})
    return buckets


def _strftime_key(period: str) -> str:
    """The SQLite strftime format that groups rows into this period's buckets."""
    if period == "week":
        return "%Y-%W"
    if period == "month":
        return "%Y-%m"
    return "%Y-%m-%d"


def chart_data(period: str = "day") -> Dict[str, Any]:
    """Return aggregated saved/real-token data for the requested period.

    ``period`` is one of ``day`` (last 7 days), ``week`` (last 8 weeks), or
    ``month`` (last 6 months). Always returns a full, zero-filled series so the
    chart shows a continuous axis even when traffic is sparse. Never raises -
    an unreadable DB yields an all-zero series with ``has_data`` False.
    """
    period = period if period in ("day", "week", "month") else "day"
    skeleton = _bucket_rows(period)
    fmt = _strftime_key(period)
    by_key: Dict[str, Dict[str, int]] = {}
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            rows = conn.execute(
                "SELECT strftime(?, timestamp, 'unixepoch', 'localtime') AS k, "
                "COUNT(*) AS requests, "
                "COALESCE(SUM(saved), 0) AS saved, "
                "COALESCE(SUM(est_tokens_in), 0) AS est_in, "
                "COALESCE(SUM(real_tokens_in), 0) AS real_in, "
                "COALESCE(SUM(real_tokens_out), 0) AS real_out "
                "FROM requests GROUP BY k",
                (fmt,),
            ).fetchall()
            for r in rows:
                by_key[r["k"]] = {
                    "requests": r["requests"], "saved": r["saved"],
                    "est_in": r["est_in"], "real_in": r["real_in"],
                    "real_out": r["real_out"],
                }
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as exc:
        log.debug("history.chart_data failed (%s)", exc)

    labels, saved, requests, real_in, real_out = [], [], [], [], []
    for bucket in skeleton:
        agg = by_key.get(bucket["key"], {})
        labels.append(bucket["label"])
        saved.append(int(agg.get("saved", 0)))
        requests.append(int(agg.get("requests", 0)))
        real_in.append(int(agg.get("real_in", 0)))
        real_out.append(int(agg.get("real_out", 0)))
    return {
        "period": period,
        "labels": labels,
        "saved": saved,
        "requests": requests,
        "real_in": real_in,
        "real_out": real_out,
        "has_data": any(requests),
    }


def totals() -> Dict[str, int]:
    """All-time aggregates from the history DB (empty zeros on any error)."""
    empty = {"requests": 0, "saved": 0, "est_in": 0, "real_in": 0, "real_out": 0}
    try:
        conn = _connect()
        try:
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT COUNT(*) AS requests, COALESCE(SUM(saved),0) AS saved, "
                "COALESCE(SUM(est_tokens_in),0) AS est_in, "
                "COALESCE(SUM(real_tokens_in),0) AS real_in, "
                "COALESCE(SUM(real_tokens_out),0) AS real_out FROM requests"
            ).fetchone()
            return {
                "requests": row["requests"], "saved": row["saved"],
                "est_in": row["est_in"], "real_in": row["real_in"],
                "real_out": row["real_out"],
            }
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return empty
