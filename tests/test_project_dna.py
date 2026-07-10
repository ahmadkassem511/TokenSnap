"""Offline tests for tokensnap.project_dna (Project Cortex, Phase 1): scanning
a temp project into a persistent DNA, the living-memory updates, focus, the
static-refresh throttle, Core Memory rendering, and the never-raises contract.
"""

import time

import pytest

from tokensnap import project_dna as dna


def _make_project(root):
    (root / "pyproject.toml").write_text(
        'dependencies = ["flask", "requests"]\n', encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# Demo\n\nA demo web service.\n", encoding="utf-8"
    )
    (root / "main.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("x = 1\n", encoding="utf-8")
    (root / "node_modules").mkdir()  # ignored


class TestLoadSave:
    def test_load_missing_returns_empty_dna(self, tmp_path):
        d = dna.load_dna(str(tmp_path))
        assert d["static"] == {}
        assert d["focus"] == ""
        assert d["schema"] == dna.SCHEMA_VERSION

    def test_save_then_load_round_trips(self, tmp_path):
        d = dna._empty_dna()
        d["focus"] = "ship it"
        assert dna.save_dna(str(tmp_path), d) is True
        assert dna.dna_path(str(tmp_path)).exists()
        assert dna.load_dna(str(tmp_path))["focus"] == "ship it"

    def test_load_fills_missing_keys(self, tmp_path):
        # A hand-written partial file must still load with the full shape.
        p = dna.dna_path(str(tmp_path))
        p.parent.mkdir(parents=True)
        p.write_text('{"focus": "partial"}', encoding="utf-8")
        d = dna.load_dna(str(tmp_path))
        assert d["focus"] == "partial"
        assert d["decisions"] == []
        assert d["static"] == {}

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        p = dna.dna_path(str(tmp_path))
        p.parent.mkdir(parents=True)
        p.write_text("{not json", encoding="utf-8")
        assert dna.load_dna(str(tmp_path))["focus"] == ""


class TestStaticAnalysis:
    def test_get_static_dna_scans_stack_and_entry_points(self, tmp_path):
        _make_project(tmp_path)
        static = dna.get_static_dna(str(tmp_path))
        assert static["language"] == "Python"
        assert static["framework"] == "Flask"
        assert "flask" in static["key_dependencies"]
        assert "main.py" in static["entry_points"]
        assert "src/app.py" in static["entry_points"]
        assert "node_modules/" not in static["folder_structure"]


class TestEnsureDna:
    def test_ensure_generates_and_persists_static(self, tmp_path):
        _make_project(tmp_path)
        d = dna.ensure_dna(str(tmp_path), {"dna_update_interval": 86400})
        assert d["static"]["language"] == "Python"
        assert d["generated_at"] > 0
        assert dna.dna_path(str(tmp_path)).exists()

    def test_ensure_unknown_dir_returns_empty_and_writes_nothing(self, tmp_path):
        missing = tmp_path / "nope"
        d = dna.ensure_dna(str(missing))
        assert d["static"] == {}
        assert not (missing / dna.DNA_DIRNAME).exists()

    def test_static_refresh_is_throttled_by_interval(self, tmp_path):
        _make_project(tmp_path)
        d1 = dna.ensure_dna(str(tmp_path), {"dna_update_interval": 86400})
        first_gen = d1["generated_at"]
        # A huge interval => the second call must NOT rescan (same timestamp).
        d2 = dna.ensure_dna(str(tmp_path), {"dna_update_interval": 86400})
        assert d2["generated_at"] == first_gen
        # A zero interval => always rescan (timestamp advances).
        time.sleep(0.01)
        d3 = dna.ensure_dna(str(tmp_path), {"dna_update_interval": 0})
        assert d3["generated_at"] > first_gen


class TestFocus:
    def test_set_focus_persists_and_clips(self, tmp_path):
        _make_project(tmp_path)
        dna.set_focus(str(tmp_path), "  Implement Project Cortex  ")
        assert dna.load_dna(str(tmp_path))["focus"] == "Implement Project Cortex"

    def test_focus_survives_a_later_session_update(self, tmp_path):
        _make_project(tmp_path)
        dna.set_focus(str(tmp_path), "the goal")
        dna.update_dna_from_session(
            str(tmp_path), [{"role": "user", "content": "we will use redis"}]
        )
        assert dna.load_dna(str(tmp_path))["focus"] == "the goal"


class TestSessionUpdate:
    def test_extracts_decisions_and_files(self, tmp_path):
        _make_project(tmp_path)
        msgs = [
            {"role": "user", "content": "let's use aiohttp in server.py"},
            {"role": "assistant", "content": "Decision: going with SQLite"},
        ]
        d = dna.update_dna_from_session(str(tmp_path), msgs)
        texts = [x["text"] for x in d["decisions"]]
        assert any("aiohttp" in t for t in texts)
        assert any("SQLite" in t for t in texts)
        assert d["changelog"]  # a changelog entry was recorded

    def test_resolved_issues_paired_across_messages(self, tmp_path):
        _make_project(tmp_path)
        msgs = [
            {"role": "user", "content": "error: ImportError in app.py"},
            {"role": "assistant", "content": "fixed the import, works now"},
        ]
        d = dna.update_dna_from_session(str(tmp_path), msgs)
        assert d["resolved_issues"]
        assert "ImportError" in d["resolved_issues"][0]["text"]

    def test_updates_are_deduplicated_and_capped(self, tmp_path):
        _make_project(tmp_path)
        msg = [{"role": "assistant", "content": "Decision: use SQLite"}]
        dna.update_dna_from_session(str(tmp_path), msg)
        dna.update_dna_from_session(str(tmp_path), msg)  # same decision again
        d = dna.load_dna(str(tmp_path))
        sqlite = [t for t in (x["text"] for x in d["decisions"]) if "SQLite" in t]
        assert len(sqlite) == 1  # not duplicated
        assert len(d["decisions"]) <= dna._MAX_DECISIONS

    def test_session_update_on_unknown_dir_is_noop(self, tmp_path):
        missing = tmp_path / "gone"
        d = dna.update_dna_from_session(str(missing), [{"role": "user", "content": "x"}])
        assert d["decisions"] == []


class TestCoreMemory:
    def test_format_core_memory_includes_key_facts(self, tmp_path):
        _make_project(tmp_path)
        dna.ensure_dna(str(tmp_path), {"dna_update_interval": 86400})
        dna.set_focus(str(tmp_path), "finish the parser")
        d = dna.update_dna_from_session(
            str(tmp_path), [{"role": "assistant", "content": "Decision: use SQLite"}]
        )
        block = dna.format_core_memory(d)
        assert "CORE MEMORY" in block
        assert "finish the parser" in block
        assert "Flask" in block
        assert "SQLite" in block

    def test_empty_dna_renders_nothing(self):
        assert dna.format_core_memory(dna._empty_dna()) == ""

    def test_core_memory_is_bounded(self, tmp_path):
        _make_project(tmp_path)
        d = dna.ensure_dna(str(tmp_path), {"dna_update_interval": 86400})
        d["focus"] = "x" * 5000
        block = dna.format_core_memory(d)
        assert len(block) <= dna._CORE_MEMORY_CHAR_CAP + 400


class TestNeverRaises:
    def test_save_dna_never_raises_on_unwritable_path(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        # Parent is a file => mkdir raises OSError, which must be swallowed.
        assert dna.save_dna(str(blocker / "nested"), dna._empty_dna()) is False

    def test_load_dna_empty_project_dir(self):
        assert dna.load_dna("")["focus"] == ""
