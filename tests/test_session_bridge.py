"""Offline tests for tokensnap.session_bridge (Project Cortex, Phase 2): saving
session summaries, finding the latest for continuity, importing external
transcripts, rendering the bridge block, pruning, and never-raising.
"""

import time

import pytest

from tokensnap import session_bridge as sb


def _convo():
    return [
        {"role": "user", "content": "build the parser in parser.py"},
        {"role": "assistant", "content": "Decision: use a recursive descent parser"},
        {"role": "user", "content": "error: stack overflow on deep input"},
        {"role": "assistant", "content": "fixed with an explicit stack, works now"},
    ]


class TestSaveAndList:
    def test_save_then_latest_round_trips(self, tmp_path):
        p = sb.save_session(str(tmp_path), "sess-1", _convo())
        assert p is not None and p.exists()
        latest = sb.latest_session(str(tmp_path))
        assert latest["session_id"] == "sess-1"
        assert latest["summary"]["task"] == "build the parser in parser.py"

    def test_summary_captures_decisions_and_fixes(self, tmp_path):
        sb.save_session(str(tmp_path), "sess-1", _convo())
        summ = sb.latest_session(str(tmp_path))["summary"]
        assert any("recursive descent" in d for d in summ["decisions"])
        assert any("stack overflow" in e for e in summ["errors_resolved"])

    def test_tiny_session_is_not_saved(self, tmp_path):
        assert sb.save_session(str(tmp_path), "s", [{"role": "user", "content": "hi"}]) is None
        assert sb.list_sessions(str(tmp_path)) == []

    def test_unknown_dir_returns_none(self, tmp_path):
        assert sb.save_session(str(tmp_path / "nope"), "s", _convo()) is None

    def test_latest_of_several_is_newest(self, tmp_path):
        sb.save_session(str(tmp_path), "old", _convo())
        time.sleep(0.01)
        sb.save_session(str(tmp_path), "new", _convo())
        assert sb.latest_session(str(tmp_path))["session_id"] == "new"

    def test_latest_excludes_current_session(self, tmp_path):
        sb.save_session(str(tmp_path), "only", _convo())
        # A session must never bridge to itself.
        assert sb.latest_session(str(tmp_path), exclude_session_id="only") is None

    def test_pruning_keeps_only_the_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sb, "_MAX_SESSIONS", 3)
        for i in range(6):
            sb.save_session(str(tmp_path), "s%d" % i, _convo())
            time.sleep(0.005)
        assert len(sb.list_sessions(str(tmp_path))) == 3


class TestImportExternal:
    def test_import_paste_becomes_a_bridge(self, tmp_path):
        text = (
            "Decision: migrate the DB to PostgreSQL.\n"
            "error: connection refused\n"
            "fixed the port, works now\n"
        )
        p = sb.import_external(str(tmp_path), text)
        assert p is not None
        latest = sb.latest_session(str(tmp_path))
        assert latest["source"] == "external"
        assert any("PostgreSQL" in d for d in latest["summary"]["decisions"])

    def test_import_empty_text_is_noop(self, tmp_path):
        assert sb.import_external(str(tmp_path), "   ") is None

    def test_imported_bridge_wins_over_older_live_session(self, tmp_path):
        sb.save_session(str(tmp_path), "live", _convo())
        time.sleep(0.01)
        sb.import_external(str(tmp_path), "Decision: switch to Rust")
        assert sb.latest_session(str(tmp_path))["source"] == "external"


class TestFormatBridge:
    def test_bridge_block_has_header_and_facts(self, tmp_path):
        sb.save_session(str(tmp_path), "s", _convo())
        block = sb.format_bridge(sb.latest_session(str(tmp_path)))
        assert "SESSION BRIDGE" in block
        assert "parser.py" in block

    def test_empty_summary_renders_nothing(self):
        assert sb.format_bridge({"summary": {}}) == ""
        assert sb.format_bridge({}) == ""

    def test_external_origin_is_noted(self, tmp_path):
        sb.import_external(str(tmp_path), "Decision: use gRPC")
        block = sb.format_bridge(sb.latest_session(str(tmp_path)))
        assert "imported from external" in block


class TestNeverRaises:
    def test_list_sessions_skips_corrupt_files(self, tmp_path):
        sb.save_session(str(tmp_path), "ok", _convo())
        bad = sb.sessions_dir(str(tmp_path)) / "9999-bad.json"
        bad.write_text("{not json", encoding="utf-8")
        sessions = sb.list_sessions(str(tmp_path))
        assert len(sessions) == 1  # the good one; the corrupt one skipped

    def test_list_sessions_missing_dir_returns_empty(self, tmp_path):
        assert sb.list_sessions(str(tmp_path / "never")) == []
