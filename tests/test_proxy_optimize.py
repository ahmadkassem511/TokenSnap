"""Offline tests for tokensnap.proxy.optimize_body: confirms the proxy wires
the configured `keep_messages` value into the compressor (not a hardcoded
constant), that the aggressive-mode threshold is read from config, and that
the Differential Context Engine path activates only when opted in.
"""

import pytest

from tokensnap import config as config_mod
from tokensnap import context_store
from tokensnap.proxy import optimize_body


def _long_history(n_exchanges):
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": "step %d: edit src/app.py" % i})
        msgs.append({"role": "assistant", "content": "did step %d" % i})
    return msgs


def _cfg(**overrides):
    cfg = dict(config_mod.DEFAULTS)
    cfg["compressor_type"] = "regex"  # keep these tests fully offline
    cfg.update(overrides)
    return cfg


class TestKeepMessagesPlumbing:
    def test_default_keeps_ten_exchanges(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg())
        assert meta["compressed"] is True
        assert len(new_body["messages"]) == 10 * 2  # 10 exchanges kept verbatim

    def test_custom_keep_messages_changes_kept_count(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(keep_messages=3))
        assert len(new_body["messages"]) == 3 * 2

    def test_maximum_preset_value_effectively_disables_compression(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(keep_messages=999))
        assert meta["compressed"] is False
        assert len(new_body["messages"]) == len(body["messages"])

    def test_short_history_untouched_regardless_of_keep_messages(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(2)}
        new_body, meta = optimize_body(body, _cfg(keep_messages=1))
        assert meta["compressed"] is False
        assert len(new_body["messages"]) == 4


class TestSelectiveCompressionPlumbing:
    def test_selective_compression_on_by_default_cleans_tool_results(self):
        dump = "\n".join(["log line %d" % i for i in range(200)] + ["error: boom"])
        msgs = _long_history(10)
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": dump}
        ]})
        body = {"model": "claude-sonnet-5", "messages": msgs}
        new_body, _ = optimize_body(body, _cfg())
        kept_tool_result = next(
            m for m in new_body["messages"]
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        text = kept_tool_result["content"][0]["content"]
        assert "tokensnap" in text  # compressed with an omission marker
        assert len(text) < len(dump)

    def test_selective_compression_off_keeps_tool_result_verbatim(self):
        dump = "\n".join(["log line %d" % i for i in range(200)] + ["error: boom"])
        msgs = _long_history(1)
        msgs.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": dump}
        ]})
        body = {"model": "claude-sonnet-5", "messages": msgs}
        new_body, _ = optimize_body(body, _cfg(selective_compression=False))
        kept_tool_result = next(
            m for m in new_body["messages"]
            if isinstance(m.get("content"), list)
            and any(b.get("type") == "tool_result" for b in m["content"])
        )
        assert kept_tool_result["content"][0]["content"] == dump

    def test_compressor_type_off_disables_truncation_entirely(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(compressor_type="off", keep_messages=3))
        assert meta["compressed"] is False
        assert len(new_body["messages"]) == len(body["messages"])


class TestAggressiveThreshold:
    def test_not_aggressive_when_under_threshold(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(3)}
        _, meta = optimize_body(body, _cfg(context_threshold=0.95))
        assert meta["aggressive"] is False

    def test_aggressive_kicks_in_past_threshold(self):
        # A tiny context_threshold guarantees "near the limit" is hit even
        # for a small request, proving the proxy reads context_threshold
        # from config rather than a hardcoded 0.9.
        body = {"model": "claude-sonnet-5", "messages": _long_history(10)}
        _, meta = optimize_body(body, _cfg(context_threshold=0.0001))
        assert meta["aggressive"] is True


class TestContextStoreDisabledByDefault:
    def test_default_path_never_adds_context_tree_or_tool(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg())
        assert "context_store" not in meta  # classic path taken
        assert "tools" not in new_body
        system = new_body.get("system") or ""
        assert "CONTEXT TREE" not in (system if isinstance(system, str) else str(system))


class TestDifferentialContextPath:
    @pytest.fixture(autouse=True)
    def isolated_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(context_store, "DB_FILE", tmp_path / "context_store.db")
        yield

    def test_long_history_rebuilt_as_tree_plus_tail(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        new_body, meta = optimize_body(body, _cfg(context_store_enabled=True))
        assert meta["context_store"] is True
        assert meta["compressed"] is True
        # Only the last 2 exchanges remain verbatim.
        assert len(new_body["messages"]) == 4
        assert meta["events_omitted"] == 40 - 4
        # fetch_context tool merged in.
        assert any(t["name"] == "fetch_context" for t in new_body["tools"])
        # Context Tree injected into system.
        system = new_body["system"]
        text = system if isinstance(system, str) else str(system)
        assert "CONTEXT TREE" in text
        # Fewer estimated tokens than the original.
        assert meta["after"] < meta["before"]

    def test_events_are_persisted_to_the_store(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(20)}
        optimize_body(body, _cfg(context_store_enabled=True))
        # Every non-empty message was mirrored into external memory.
        assert context_store.event_count() == 40

    def test_short_history_not_reconstructed(self):
        body = {"model": "claude-sonnet-5", "messages": _long_history(2)}
        new_body, meta = optimize_body(body, _cfg(context_store_enabled=True))
        assert meta["context_store"] is True
        assert meta["compressed"] is False
        assert "tools" not in new_body  # nothing omitted -> no tool advertised
        assert len(new_body["messages"]) == 4

    def test_existing_tools_are_preserved(self):
        body = {
            "model": "claude-sonnet-5",
            "messages": _long_history(20),
            "tools": [{"name": "Bash"}, {"name": "Read"}],
        }
        new_body, _ = optimize_body(body, _cfg(context_store_enabled=True))
        names = [t["name"] for t in new_body["tools"]]
        assert names == ["Bash", "Read", "fetch_context"]


class TestProjectPrimer:
    """The primer injects a project overview on the first request of a session.
    conftest keeps the project state isolated; here we point it at a temp
    project so the primer has something real to scan. Project Cortex supersedes
    the primer, so these tests disable it (`project_cortex_enabled=False`) to
    exercise the primer path specifically."""

    @pytest.fixture(autouse=True)
    def temp_project(self, tmp_path, monkeypatch):
        from tokensnap import project, project_primer

        monkeypatch.setattr(context_store, "DB_FILE", tmp_path / "context_store.db")
        (tmp_path / "pyproject.toml").write_text(
            'dependencies = ["flask"]\n', encoding="utf-8"
        )
        project.set_current_project(str(tmp_path))
        project_primer.reset_cache()
        yield

    @staticmethod
    def _pcfg(**kw):
        kw.setdefault("project_cortex_enabled", False)
        return _cfg(**kw)

    def _has_primer(self, new_body):
        system = new_body.get("system") or ""
        return "PROJECT PRIMER" in (system if isinstance(system, str) else str(system))

    def test_injected_on_first_request_only(self):
        body = {"model": "claude-sonnet-5",
                "system": "base", "messages": _long_history(1)}
        nb1, m1 = optimize_body(dict(body), self._pcfg())
        assert m1["primed"] is True
        assert self._has_primer(nb1)
        # Same conversation again -> already primed, not re-injected.
        nb2, m2 = optimize_body(dict(body), self._pcfg())
        assert m2["primed"] is False
        assert not self._has_primer(nb2)

    def test_disabled_toggle_skips_injection(self):
        body = {"model": "claude-sonnet-5",
                "system": "base", "messages": _long_history(1)}
        nb, m = optimize_body(body, self._pcfg(project_primer_enabled=False))
        assert m["primed"] is False
        assert not self._has_primer(nb)

    def test_injected_in_differential_path_too(self):
        body = {"model": "claude-sonnet-5",
                "system": "base", "messages": _long_history(20)}
        nb, m = optimize_body(body, self._pcfg(context_store_enabled=True))
        assert m["primed"] is True
        assert self._has_primer(nb)

    def test_unknown_project_is_not_primed(self):
        from tokensnap import project

        project.set_current_project("")  # clears -> get_current_project == 'unknown'
        body = {"model": "claude-sonnet-5",
                "system": "base", "messages": _long_history(1)}
        _, m = optimize_body(body, self._pcfg())
        assert m["primed"] is False


class TestProjectCortex:
    """Core Memory (Project DNA) + Session Bridge injection and session flush."""

    @pytest.fixture(autouse=True)
    def temp_project(self, tmp_path, monkeypatch):
        from tokensnap import project, project_dna, proxy as proxy_mod

        monkeypatch.setattr(context_store, "DB_FILE", tmp_path / "context_store.db")
        proj = tmp_path / "proj"
        proj.mkdir()
        (proj / "pyproject.toml").write_text('dependencies = ["flask"]\n', encoding="utf-8")
        (proj / "main.py").write_text("print(1)\n", encoding="utf-8")
        project.set_current_project(str(proj))
        project_dna.set_focus(str(proj), "Build the login flow")
        proxy_mod.reset_cortex_state()
        self.proj = str(proj)
        yield

    @staticmethod
    def _sys(new_body):
        s = new_body.get("system") or ""
        return s if isinstance(s, str) else str(s)

    def test_core_memory_injected_first_request_only(self):
        body = {"model": "claude-sonnet-5", "system": "base",
                "messages": _long_history(1)}
        nb, m = optimize_body(dict(body), _cfg())
        assert m["primed"] is True
        s = self._sys(nb)
        assert "CORE MEMORY" in s
        assert "Build the login flow" in s   # the focus is present
        assert "PROJECT PRIMER" not in s       # cortex supersedes the primer
        # Not re-injected on the next request of the same session.
        _, m2 = optimize_body(dict(body), _cfg())
        assert m2["primed"] is False

    def test_session_bridge_injected_when_prior_session_exists(self):
        from tokensnap import session_bridge
        session_bridge.save_session(self.proj, "old-sess", [
            {"role": "user", "content": "set up db in db.py"},
            {"role": "assistant", "content": "Decision: use SQLite"},
            {"role": "user", "content": "error: locked db"},
            {"role": "assistant", "content": "fixed with WAL, works now"},
        ])
        body = {"model": "claude-sonnet-5", "system": "base",
                "messages": _long_history(1)}
        nb, _ = optimize_body(dict(body), _cfg())
        s = self._sys(nb)
        assert "SESSION BRIDGE" in s
        assert "SQLite" in s  # resumes the prior decision

    def test_cortex_disabled_falls_back_to_primer(self):
        body = {"model": "claude-sonnet-5", "system": "base",
                "messages": _long_history(1)}
        nb, m = optimize_body(dict(body), _cfg(project_cortex_enabled=False))
        s = self._sys(nb)
        assert "CORE MEMORY" not in s
        assert "PROJECT PRIMER" in s  # primer is the fallback

    def test_session_flush_updates_dna_and_saves_bridge(self):
        from tokensnap import project_dna, session_bridge, proxy as proxy_mod

        convo = _long_history(10)
        convo.append({"role": "assistant", "content": "Decision: adopt pytest fixtures"})
        body = {"model": "claude-sonnet-5", "system": "base", "messages": convo}
        optimize_body(dict(body), _cfg())  # tracks the session
        proxy_mod.flush_all_sessions()
        dna = project_dna.load_dna(self.proj)
        assert dna["changelog"]  # session distilled into the DNA
        assert any("pytest fixtures" in d["text"] for d in dna["decisions"])
        assert session_bridge.latest_session(self.proj) is not None  # bridge saved

    def test_unknown_project_injects_nothing(self):
        from tokensnap import project
        project.set_current_project("")
        body = {"model": "claude-sonnet-5", "system": "base",
                "messages": _long_history(1)}
        _, m = optimize_body(dict(body), _cfg())
        assert m["primed"] is False
