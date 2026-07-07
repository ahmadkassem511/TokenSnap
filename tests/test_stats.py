"""Offline tests for tokensnap.stats: pid liveness, stop_proxy, mark_stopped.

Uses a real (but harmless) sleeper subprocess to exercise the actual
termination path safely - it must never target the pytest process itself.
"""

import os
import socket
import subprocess
import sys
import time

import pytest

from tokensnap import stats


@pytest.fixture(autouse=True)
def isolated_stats_file(tmp_path, monkeypatch):
    """Redirect the stats file to a throwaway location for every test."""
    monkeypatch.setattr(stats, "STATS_DIR", tmp_path)
    monkeypatch.setattr(stats, "STATS_FILE", tmp_path / "stats.json")
    yield


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_sleeper():
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class TestPidAlive:
    def test_current_process_is_alive(self):
        assert stats._pid_alive(os.getpid()) is True

    def test_bogus_pid_is_not_alive(self):
        assert stats._pid_alive(999_999_999) is False

    def test_none_pid_is_not_alive(self):
        assert stats._pid_alive(None) is False

    def test_zero_pid_is_not_alive(self):
        assert stats._pid_alive(0) is False


class TestMarkStopped:
    def test_clears_proxy_info_keeps_totals(self):
        stats.mark_started("127.0.0.1", 8889)
        stats.record_request("/v1/messages", "claude-sonnet-5", 100, 60, 200, 0.1)
        stats.mark_stopped()
        data = stats.load()
        assert data["proxy"] == {}
        assert data["totals"]["requests"] == 1  # totals are not wiped


class TestContextEngineTotals:
    @pytest.fixture(autouse=True)
    def isolated_history(self, tmp_path, monkeypatch):
        # record_request also best-effort-logs to history.db; keep it off the
        # real one so this test stays fully isolated.
        from tokensnap import history

        monkeypatch.setattr(history, "DB_FILE", tmp_path / "history.db")
        yield

    def test_context_totals_only_move_on_context_requests(self):
        stats.mark_started("127.0.0.1", 8889)
        # A classic request must not touch the context subtotals.
        stats.record_request("/v1/messages", "m", 100, 60, 200, 0.1)
        # Two Context Engine requests: saved 40 then 50, fetching 2 then 1 events.
        stats.record_request("/v1/messages", "m", 100, 60, 200, 0.1,
                             context_store=True, events_fetched=2)
        stats.record_request("/v1/messages", "m", 90, 40, 200, 0.1,
                             context_store=True, events_fetched=1)
        totals = stats.load()["totals"]
        assert totals["requests"] == 3
        assert totals["context_requests"] == 2
        assert totals["context_saved"] == 40 + 50
        assert totals["context_events_fetched"] == 3

    def test_old_stats_file_backfills_context_keys(self):
        stats._save({
            "proxy": {},
            "totals": {"requests": 1, "tokens_before": 10, "tokens_after": 5,
                       "tokens_saved": 5},
            "recent": [],
        })
        totals = stats.load()["totals"]
        assert totals["context_requests"] == 0
        assert totals["context_saved"] == 0
        assert totals["context_events_fetched"] == 0


class TestRealUsage:
    def test_record_accumulates_real_usage(self):
        stats.mark_started("127.0.0.1", 8889)
        stats.record_request(
            "/v1/messages", "claude-sonnet-5", 100, 60, 200, 0.1,
            real_input=1000, real_output=200, real_cache_read=8000,
            real_cache_creation=50,
        )
        stats.record_request(
            "/v1/messages", "claude-sonnet-5", 80, 40, 200, 0.1,
            real_input=500, real_output=100, real_cache_read=4000,
            real_cache_creation=0,
        )
        totals = stats.load()["totals"]
        assert totals["real_input"] == 1500
        assert totals["real_output"] == 300
        assert totals["real_cache_read"] == 12000
        assert totals["real_cache_creation"] == 50

        recent = stats.load()["recent"][-1]
        assert recent["real_input"] == 500
        assert recent["real_cache_read"] == 4000

    def test_old_stats_file_backfills_new_keys(self):
        # A stats file written by an older version lacks the real_* totals.
        stats._save({
            "proxy": {},
            "totals": {"requests": 3, "tokens_before": 900, "tokens_after": 300,
                       "tokens_saved": 600},
            "recent": [],
        })
        totals = stats.load()["totals"]
        assert totals["requests"] == 3          # preserved
        assert totals["real_input"] == 0        # backfilled, no KeyError
        assert totals["real_cache_read"] == 0


class TestStopProxy:
    def test_nothing_running_returns_false(self):
        port = _free_port()  # guaranteed nothing listens here
        attempted, pid = stats.stop_proxy("127.0.0.1", port)
        assert attempted is False
        assert pid is None

    def test_terminates_process_recorded_by_pid(self):
        proc = _spawn_sleeper()
        try:
            port = _free_port()
            data = stats.load()
            data["proxy"] = {
                "pid": proc.pid,
                "host": "127.0.0.1",
                "port": port,
                "started_at": time.time(),
            }
            stats._save(data)

            attempted, pid = stats.stop_proxy("127.0.0.1", port, timeout=10)
            assert attempted is True
            assert pid == proc.pid

            proc.wait(timeout=10)
            assert proc.poll() is not None  # process actually exited

            assert stats.load()["proxy"] == {}
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)

    def test_stale_pid_falls_back_to_port_lookup(self):
        # A pid that isn't alive should be ignored in favor of the port
        # based fallback (which, with nothing listening, still finds none).
        port = _free_port()
        data = stats.load()
        data["proxy"] = {
            "pid": 999_999_999,
            "host": "127.0.0.1",
            "port": port,
            "started_at": time.time(),
        }
        stats._save(data)

        attempted, pid = stats.stop_proxy("127.0.0.1", port)
        assert attempted is False
        assert pid is None
