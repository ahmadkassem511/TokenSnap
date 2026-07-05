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
