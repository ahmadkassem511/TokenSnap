"""Offline tests for tokensnap.history: SQLite logging, aggregates, charts.

Every test redirects the DB to a throwaway file so nothing touches the real
~/.tokensnap/history.db. The module claims log_request "never raises" - a few
tests deliberately point it at a broken path to prove that contract holds.
"""

import time
from datetime import date, datetime, timedelta

import pytest

from tokensnap import history


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point history at a fresh throwaway DB for every test."""
    monkeypatch.setattr(history, "DB_FILE", tmp_path / "history.db")
    yield


def _ts_for(days_ago: int = 0) -> float:
    """A unix timestamp at local noon, N days before today (stable bucket key)."""
    d = date.today() - timedelta(days=days_ago)
    return datetime(d.year, d.month, d.day, 12, 0, 0).timestamp()


def _log(saved=100, ts=None, **kw):
    args = dict(
        model="qwen2.5:7b",
        est_tokens_in=1000,
        real_tokens_in=800,
        real_tokens_out=200,
        cache_read=0,
        cache_write=0,
        saved=saved,
        http_status=200,
    )
    args.update(kw)
    history.log_request(ts=ts, **args)


class TestInitAndTotals:
    def test_empty_db_has_zero_totals(self):
        t = history.totals()
        assert t == {"requests": 0, "saved": 0, "est_in": 0, "real_in": 0, "real_out": 0}

    def test_init_db_creates_file(self):
        history.init_db()
        assert history.DB_FILE.exists()

    def test_log_then_totals(self):
        _log(saved=100, est_tokens_in=1000, real_tokens_in=800, real_tokens_out=200)
        _log(saved=50, est_tokens_in=500, real_tokens_in=400, real_tokens_out=100)
        t = history.totals()
        assert t["requests"] == 2
        assert t["saved"] == 150
        assert t["est_in"] == 1500
        assert t["real_in"] == 1200
        assert t["real_out"] == 300

    def test_log_autocreates_db(self):
        # No explicit init_db() - the first log_request must self-heal the schema.
        assert not history.DB_FILE.exists()
        _log()
        assert history.DB_FILE.exists()
        assert history.totals()["requests"] == 1


class TestChartData:
    def test_day_period_has_seven_buckets(self):
        c = history.chart_data("day")
        assert c["period"] == "day"
        assert len(c["labels"]) == 7
        assert len(c["saved"]) == 7
        assert c["has_data"] is False  # empty DB

    def test_week_period_has_eight_buckets(self):
        assert len(history.chart_data("week")["labels"]) == 8

    def test_month_period_has_six_buckets(self):
        assert len(history.chart_data("month")["labels"]) == 6

    def test_unknown_period_falls_back_to_day(self):
        assert history.chart_data("century")["period"] == "day"

    def test_today_traffic_lands_in_last_day_bucket(self):
        _log(saved=100, ts=_ts_for(0))
        c = history.chart_data("day")
        assert c["has_data"] is True
        # Buckets are oldest-first, so today is the final one.
        assert c["saved"][-1] == 100
        assert c["requests"][-1] == 1
        assert sum(c["saved"][:-1]) == 0

    def test_saved_sums_within_a_bucket(self):
        _log(saved=100, ts=_ts_for(0))
        _log(saved=25, ts=_ts_for(0))
        c = history.chart_data("day")
        assert c["saved"][-1] == 125
        assert c["requests"][-1] == 2

    def test_older_traffic_lands_in_earlier_bucket(self):
        _log(saved=30, ts=_ts_for(3))
        c = history.chart_data("day")
        # 7 day buckets, index 6 == today, so 3 days ago == index 3.
        assert c["saved"][3] == 30
        assert c["saved"][-1] == 0


class TestNeverRaises:
    def test_log_request_swallows_bad_db_path(self, monkeypatch, tmp_path):
        # Parent is a regular file, so mkdir() raises OSError, not sqlite3.Error.
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(history, "DB_FILE", blocker / "nested" / "history.db")
        _log()  # must not raise

    def test_chart_data_returns_zeros_on_bad_db(self, monkeypatch, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(history, "DB_FILE", blocker / "nested" / "history.db")
        c = history.chart_data("day")
        assert c["has_data"] is False
        assert c["saved"] == [0] * 7

    def test_totals_returns_zeros_on_bad_db(self, monkeypatch, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(history, "DB_FILE", blocker / "nested" / "history.db")
        assert history.totals()["requests"] == 0


class TestProjectTracking:
    def test_project_defaults_to_unknown(self):
        _log()  # no project passed
        rows = history.project_totals()
        assert rows == [
            {"project": "unknown", "requests": 1, "saved": 100,
             "real_in": 800, "real_out": 200}
        ]

    def test_project_totals_groups_and_orders_by_saved(self):
        _log(saved=100, project="proj-a")
        _log(saved=50, project="proj-a")
        _log(saved=500, project="proj-b")
        rows = history.project_totals()
        # Ordered by total tokens saved, descending.
        assert [r["project"] for r in rows] == ["proj-b", "proj-a"]
        a = next(r for r in rows if r["project"] == "proj-a")
        assert a["requests"] == 2
        assert a["saved"] == 150

    def test_project_totals_empty_on_fresh_db(self):
        assert history.project_totals() == []

    def test_chart_data_filters_by_project(self):
        _log(saved=100, project="proj-a", ts=_ts_for(0))
        _log(saved=999, project="proj-b", ts=_ts_for(0))
        all_saved = sum(history.chart_data("day")["saved"])
        a_saved = sum(history.chart_data("day", project="proj-a")["saved"])
        assert all_saved == 1099
        assert a_saved == 100
        # The filter is echoed back so the frontend can confirm it.
        assert history.chart_data("day", project="proj-a")["project"] == "proj-a"
        assert history.chart_data("day")["project"] is None

    def test_totals_ignore_project_and_stay_global(self):
        _log(saved=100, project="proj-a")
        _log(saved=50, project="proj-b")
        assert history.totals()["saved"] == 150
        assert history.totals()["requests"] == 2

    def test_legacy_db_without_project_is_migrated(self):
        # Simulate a pre-'project' history.db: create the old requests table
        # (no project column) with a row, then log through the new code path
        # and confirm the column is added and both rows are queryable.
        import sqlite3

        conn = sqlite3.connect(str(history.DB_FILE))
        conn.executescript(
            "CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp REAL NOT NULL, model TEXT, est_tokens_in INTEGER DEFAULT 0, "
            "real_tokens_in INTEGER DEFAULT 0, real_tokens_out INTEGER DEFAULT 0, "
            "cache_read INTEGER DEFAULT 0, cache_write INTEGER DEFAULT 0, "
            "saved INTEGER DEFAULT 0, http_status INTEGER DEFAULT 0);"
        )
        conn.execute("INSERT INTO requests (timestamp, model, saved) VALUES (1, 'old', 42)")
        conn.commit()
        conn.close()

        _log(saved=10, project="proj-new")  # triggers migration + insert

        rows = {r["project"]: r for r in history.project_totals()}
        assert rows["unknown"]["saved"] == 42       # legacy row backfilled
        assert rows["proj-new"]["saved"] == 10       # new tagged row
        assert history.totals()["requests"] == 2

    def test_project_totals_never_raises_on_bad_db(self, monkeypatch, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(history, "DB_FILE", blocker / "nested" / "history.db")
        assert history.project_totals() == []
