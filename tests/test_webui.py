"""Offline tests for tokensnap.webui pure helpers and app wiring.

These avoid binding a real socket - they exercise log tailing, setup-marker,
route table, and page rendering without a running server. The setup marker
and log path are bound to config_mod.CONFIG_DIR at import time, so tests
monkeypatch the already-resolved module attributes.
"""

import asyncio
import json
import os
import re

import pytest

from tokensnap import webui


class TestTailLog:
    def test_missing_log_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        assert webui.tail_log() == []

    def test_returns_last_n_lines(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        (tmp_path / "proxy.log").write_text(
            "\n".join(f"line{i}" for i in range(10)) + "\n", encoding="utf-8"
        )
        assert webui.tail_log(3) == ["line7", "line8", "line9"]


class TestSetupMarker:
    def test_complete_reflects_marker(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui, "_SETUP_MARKER", tmp_path / ".setup_complete")
        assert webui.setup_is_complete() is False
        webui.mark_setup_complete()
        assert webui.setup_is_complete() is True


class TestBuildApp:
    def test_registers_core_routes(self):
        app = webui.build_app()
        paths = {r.resource.canonical for r in app.router.routes()}
        for expected in ("/", "/setup", "/settings", "/api/stats", "/api/chart", "/api/log"):
            assert expected in paths

    def test_registers_context_route(self):
        app = webui.build_app()
        paths = {r.resource.canonical for r in app.router.routes()}
        assert "/api/context" in paths


class TestContextEngineDashboard:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(webui.stats, "STATS_DIR", tmp_path)
        monkeypatch.setattr(webui.stats, "STATS_FILE", tmp_path / "stats.json")
        monkeypatch.setattr(webui.context_store, "DB_FILE", tmp_path / "context_store.db")
        yield

    def _context_json(self):
        return json.loads(asyncio.run(webui.api_context(None)).text)

    def test_reports_disabled_and_zero_by_default(self):
        data = self._context_json()
        assert data["enabled"] is False
        assert data["tree_size"] == 20
        assert data["events_stored"] == 0
        assert data["tokens_saved"] == 0

    def test_reports_stored_events_and_enabled(self):
        from tokensnap import config as config_mod, context_store

        config_mod.set_value("context_store_enabled", "true")
        context_store.store_message("s", 0, "user", "hi", "hi", "error")
        context_store.store_message("s", 1, "assistant", "yo", "yo", "decision")
        data = self._context_json()
        assert data["enabled"] is True
        assert data["events_stored"] == 2

    def test_public_config_includes_context_keys(self):
        cfg = webui._public_config()
        assert cfg["context_store_enabled"] is False
        assert cfg["context_tree_size"] == 20

    def test_apply_settings_saves_context_keys(self):
        from tokensnap import config as config_mod

        class FakeRequest:
            async def json(self_inner):
                return {"context_store_enabled": "true", "context_tree_size": "35"}

        asyncio.run(webui._apply_settings(FakeRequest()))
        loaded = config_mod.load()
        assert loaded["context_store_enabled"] is True
        assert loaded["context_tree_size"] == 35

    def test_settings_page_has_context_engine_toggle(self):
        html = webui._settings_page()
        assert "id='ctxenabled'" in html
        assert "Differential Context Engine" in html

    def _stats_json(self):
        return json.loads(asyncio.run(webui.api_stats(None)).text)

    def test_api_stats_reports_disabled_by_default(self):
        data = self._stats_json()
        assert data["context_store_enabled"] is False
        assert data["context_tree_size"] == 20
        # Same one-liner wording as `tokensnap status` / `monitor`.
        assert data["context_status"] == "disabled"

    def test_api_stats_reports_enabled_and_custom_tree_size(self):
        from tokensnap import config as config_mod

        config_mod.set_value("context_store_enabled", "true")
        config_mod.set_value("context_tree_size", "42")
        data = self._stats_json()
        assert data["context_store_enabled"] is True
        assert data["context_tree_size"] == 42
        assert data["context_status"] == "enabled (tree size: 42)"

    def test_dashboard_page_has_context_engine_card(self):
        html = webui._dashboard_page()
        assert "id='c_ctx'" in html
        assert "id='c_ctx_sub'" in html
        # Label matches the CLI/monitor wording exactly.
        assert "Differential Context Engine" in html
        # The card is driven by the same /api/stats poll as the other cards.
        assert "context_status" in html

    def test_registers_openrouter_status_route(self):
        app = webui.build_app()
        paths = {r.resource.canonical for r in app.router.routes()}
        assert "/api/openrouter-status" in paths

    def test_registers_launch_route(self):
        app = webui.build_app()
        paths = {r.resource.canonical for r in app.router.routes()}
        assert "/launch" in paths

    def test_ollama_routes_removed(self):
        # The Ollama pull/hardware-detection endpoints no longer exist.
        app = webui.build_app()
        paths = {r.resource.canonical for r in app.router.routes()}
        assert "/setup/hardware" not in paths
        assert "/setup/pull" not in paths


class TestPublicConfig:
    @pytest.fixture(autouse=True)
    def isolated_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        yield

    def test_never_exposes_the_raw_key(self):
        from tokensnap import config as config_mod

        config_mod.set_value("openrouter_api_key", "sk-or-super-secret")
        cfg = webui._public_config()
        assert "openrouter_api_key" not in cfg
        assert cfg["openrouter_api_key_set"] is True
        assert "sk-or-super-secret" not in json.dumps(cfg)

    def test_reports_no_key_set(self):
        cfg = webui._public_config()
        assert cfg["openrouter_api_key_set"] is False

    def test_reports_compressor_settings(self):
        from tokensnap import config as config_mod

        config_mod.set_value("compressor_type", "openrouter")
        config_mod.set_value("selective_compression", "false")
        cfg = webui._public_config()
        assert cfg["compressor_type"] == "openrouter"
        assert cfg["selective_compression"] is False

    def test_reports_fallback_models(self):
        from tokensnap import config as config_mod

        config_mod.set_value("openrouter_fallback_models", "model-a, model-b")
        cfg = webui._public_config()
        assert cfg["openrouter_fallback_models"] == ["model-a", "model-b"]

    def test_default_fallback_models_is_empty_list(self):
        cfg = webui._public_config()
        assert cfg["openrouter_fallback_models"] == []


class TestApiStatsOpenRouterFields:
    """Regression test: the dashboard's Memory Cards pill used to turn green
    whenever compressor_type=="openrouter", even with no API key configured
    (i.e. OpenRouter isn't actually doing anything - it's silently falling
    back to regex). The fix requires api_stats() to expose whether a key is
    actually set, not just which mode is selected."""

    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(webui.stats, "STATS_DIR", tmp_path)
        monkeypatch.setattr(webui.stats, "STATS_FILE", tmp_path / "stats.json")
        yield

    def _stats_json(self):
        response = asyncio.run(webui.api_stats(None))
        return json.loads(response.text)

    def test_reports_key_set_true(self):
        from tokensnap import config as config_mod

        config_mod.set_value("compressor_type", "openrouter")
        config_mod.set_value("openrouter_api_key", "sk-or-test")
        data = self._stats_json()
        assert data["compressor_type"] == "openrouter"
        assert data["openrouter_api_key_set"] is True

    def test_reports_key_set_false_when_openrouter_selected_but_no_key(self):
        from tokensnap import config as config_mod

        config_mod.set_value("compressor_type", "openrouter")
        data = self._stats_json()
        assert data["compressor_type"] == "openrouter"
        assert data["openrouter_api_key_set"] is False

    def test_never_exposes_the_raw_key_value(self):
        from tokensnap import config as config_mod

        config_mod.set_value("openrouter_api_key", "sk-or-super-secret")
        response = asyncio.run(webui.api_stats(None))
        assert "sk-or-super-secret" not in response.text


class TestApplySettingsFallbackModels:
    @pytest.fixture(autouse=True)
    def isolated_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        yield

    def _apply(self, payload):
        class FakeRequest:
            async def json(self_inner):
                return payload

        return asyncio.run(webui._apply_settings(FakeRequest()))

    def test_saves_comma_separated_fallback_models(self):
        from tokensnap import config as config_mod

        self._apply({"openrouter_fallback_models": "a, b, c"})
        assert config_mod.load()["openrouter_fallback_models"] == ["a", "b", "c"]

    def test_explicit_empty_string_clears_fallback_models(self):
        from tokensnap import config as config_mod

        config_mod.set_value("openrouter_fallback_models", "a, b")
        self._apply({"openrouter_fallback_models": ""})
        assert config_mod.load()["openrouter_fallback_models"] == []

    def test_absent_key_leaves_fallback_models_untouched(self):
        from tokensnap import config as config_mod

        config_mod.set_value("openrouter_fallback_models", "a, b")
        self._apply({"keep_messages": 15})
        assert config_mod.load()["openrouter_fallback_models"] == ["a", "b"]


class TestLaunchClaudeTerminal:
    @pytest.fixture(autouse=True)
    def isolate_project_file(self, tmp_path, monkeypatch):
        # Launching now writes the current-project state file; keep it off the
        # real ~/.tokensnap/current_project.
        monkeypatch.setattr(webui.project, "PROJECT_FILE", tmp_path / "current_project")
        yield

    def test_claude_not_installed_returns_install_hint(self, monkeypatch):
        # Claude can't be resolved any way (not on PATH, no npm global, no npx).
        monkeypatch.setattr(webui, "resolve_claude_command", lambda cmd: None)
        ok, message = webui._launch_claude_terminal()
        assert ok is False
        assert "claude.ai/download" in message or "npm install" in message

    def test_starts_proxy_if_not_running(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        # claude resolves (installed, possibly off-PATH -> found via npm/npx).
        monkeypatch.setattr(webui, "resolve_claude_command", lambda cmd: list(cmd))
        # On non-Windows CI runners (no real terminal emulator installed), the
        # Linux terminal-detection loop still needs at least one hit so the
        # test exercises success regardless of which platform branch runs.
        known_terminals = {"x-terminal-emulator", "gnome-terminal", "konsole", "xterm"}
        monkeypatch.setattr(
            webui.shutil, "which",
            lambda name: ("/usr/bin/" + name) if name in known_terminals else None,
        )
        monkeypatch.setattr(webui.stats, "proxy_running", lambda *a, **k: False)
        started = {"called": False}

        def fake_start_proxy_detached():
            started["called"] = True
            return True, tmp_path / "proxy.log"

        monkeypatch.setattr(webui.stats, "start_proxy_detached", fake_start_proxy_detached)
        monkeypatch.setattr(webui.subprocess, "Popen", lambda *a, **k: None)
        ok, message = webui._launch_claude_terminal()
        assert started.get("called") is True
        assert ok is True
        assert "Claude Code launched" in message

    def test_proxy_start_failure_is_reported(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(webui, "resolve_claude_command", lambda cmd: list(cmd))
        monkeypatch.setattr(webui.stats, "proxy_running", lambda *a, **k: False)
        monkeypatch.setattr(
            webui.stats, "start_proxy_detached", lambda: (False, tmp_path / "proxy.log")
        )
        ok, message = webui._launch_claude_terminal()
        assert ok is False
        assert "proxy" in message.lower()


class TestPageRendering:
    """Regression tests for two bugs that made every button on the setup and
    settings pages silently dead:

    1. Page-specific inline <script> blocks were placed inside <head>, which
       is parsed and executed before the <body> elements they look up (by
       id/class) exist. The first DOM lookup returned null, threw, and
       silently aborted the rest of that script - including all the event
       handler assignments later in the same block.
    2. The embedded config JSON was substituted into the template with a
       naive str.replace("__CFG__", json), but "__CFG__" is also a substring
       of the `window.__CFG__` property name in the same line, so both got
       replaced - corrupting `window.__CFG__=<json>;` into
       `window.<json>=<json>;`, a JS syntax error.
    """

    @pytest.fixture(autouse=True)
    def isolated_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        yield

    def _head_and_body(self, html):
        head = html[: html.index("</head>")]
        body = html[html.index("<body>") :]
        return head, body

    @pytest.mark.parametrize(
        "page_fn, script_marker",
        [
            (webui._setup_page, "getElementById('finish')"),
            (webui._settings_page, "getElementById('save')"),
            (webui._dashboard_page, "loadChart"),
        ],
    )
    def test_page_script_is_not_in_head(self, page_fn, script_marker):
        html = page_fn()
        head, _ = self._head_and_body(html)
        assert script_marker not in head, (
            "page-specific script leaked into <head>, where it would execute "
            "before <body> elements exist"
        )

    @pytest.mark.parametrize(
        "page_fn, element_id",
        [(webui._setup_page, "apikey"), (webui._settings_page, "save"),
         (webui._dashboard_page, "chart")],
    )
    def test_referenced_element_appears_before_its_script(self, page_fn, element_id):
        html = page_fn()
        element_pos = html.index("id='%s'" % element_id)
        script_pos = html.rindex("<script>")  # the last inline script block
        assert element_pos < script_pos

    @pytest.mark.parametrize("page_fn", [webui._setup_page, webui._settings_page])
    def test_embedded_config_is_valid_uncorrupted_json(self, page_fn):
        html = page_fn()
        match = re.search(r"window\.__TSNAP_CFG__=(\{.*?\});", html)
        assert match, "window.__TSNAP_CFG__ assignment not found"
        embedded = json.loads(match.group(1))  # raises if corrupted
        assert embedded == webui._public_config()

    def test_setup_page_prefills_saved_values(self):
        from tokensnap import config as config_mod

        config_mod.set_value("keep_messages", "17")
        config_mod.set_value("openrouter_model", "custom-model")
        html = webui._setup_page()
        match = re.search(r"window\.__TSNAP_CFG__=(\{.*?\});", html)
        embedded = json.loads(match.group(1))
        assert embedded["keep_messages"] == 17
        assert embedded["openrouter_model"] == "custom-model"

    def test_settings_page_prefills_saved_values(self):
        from tokensnap import config as config_mod

        config_mod.set_value("keep_messages", "17")
        config_mod.set_value("openrouter_model", "custom-model")
        html = webui._settings_page()
        match = re.search(r"window\.__TSNAP_CFG__=(\{.*?\});", html)
        embedded = json.loads(match.group(1))
        assert embedded["keep_messages"] == 17
        assert embedded["openrouter_model"] == "custom-model"

    def test_setup_page_has_no_ollama_references(self):
        html = webui._setup_page()
        assert "ollama" not in html.lower()

    def test_settings_page_has_no_ollama_references(self):
        html = webui._settings_page()
        assert "ollama" not in html.lower()

    def test_settings_page_never_embeds_raw_api_key(self):
        from tokensnap import config as config_mod

        config_mod.set_value("openrouter_api_key", "sk-or-super-secret-value")
        html = webui._settings_page()
        assert "sk-or-super-secret-value" not in html

    def test_settings_page_offers_smart_preset(self):
        html = webui._settings_page()
        assert "data-preset='smart'" in html

    def test_setup_page_offers_smart_preset(self):
        html = webui._setup_page()
        assert "data-preset='smart'" in html

    def test_setup_page_has_launch_claude_button(self):
        html = webui._setup_page()
        assert "id='launchClaudeBtn'" in html
        assert "Launch Claude Code" in html

    def test_settings_page_has_launch_claude_button(self):
        html = webui._settings_page()
        assert "id='launchClaudeBtn'" in html
        assert "Launch Claude Code with current settings" in html

    def test_settings_page_has_fallback_models_field(self):
        html = webui._settings_page()
        assert "id='fallback'" in html

    def test_launch_buttons_appear_before_their_script(self):
        for page_fn in (webui._setup_page, webui._settings_page):
            html = page_fn()
            btn_pos = html.index("id='launchClaudeBtn'")
            script_pos = html.rindex("<script>")
            assert btn_pos < script_pos


class _FakeJsonRequest:
    """Minimal stand-in for web.Request exposing an async .json() body."""

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class TestProjectDirEndpoints:
    @pytest.fixture(autouse=True)
    def restore_launch_dir(self):
        # The launch dir is module-global state; snapshot and restore it so
        # these tests never leak a (possibly since-deleted) path into others.
        saved = webui.get_launch_dir()
        yield
        webui._launch_dir = saved

    def test_routes_registered(self):
        app = webui.build_app()
        paths = {r.resource.canonical for r in app.router.routes()}
        assert "/browse-folder" in paths
        assert "/get-project-dir" in paths
        assert "/set-project-dir" in paths

    def test_get_project_dir_returns_current(self, tmp_path):
        webui.set_launch_dir(str(tmp_path))
        data = json.loads(asyncio.run(webui.get_project_dir(None)).text)
        assert data["path"] == webui.get_launch_dir()

    def test_set_project_dir_accepts_existing_directory(self, tmp_path):
        resp = asyncio.run(webui.set_project_dir(_FakeJsonRequest({"path": str(tmp_path)})))
        data = json.loads(resp.text)
        assert data["ok"] is True
        assert webui.get_launch_dir() == os.path.abspath(str(tmp_path))

    def test_set_project_dir_rejects_missing_path(self, tmp_path):
        before = webui.get_launch_dir()
        missing = str(tmp_path / "does-not-exist")
        resp = asyncio.run(webui.set_project_dir(_FakeJsonRequest({"path": missing})))
        data = json.loads(resp.text)
        assert data["ok"] is False
        assert "error" in data
        assert webui.get_launch_dir() == before  # unchanged

    def test_set_project_dir_rejects_a_file(self, tmp_path):
        f = tmp_path / "afile.txt"
        f.write_text("x", encoding="utf-8")
        resp = asyncio.run(webui.set_project_dir(_FakeJsonRequest({"path": str(f)})))
        assert json.loads(resp.text)["ok"] is False

    def test_set_project_dir_rejects_empty(self):
        resp = asyncio.run(webui.set_project_dir(_FakeJsonRequest({"path": ""})))
        assert json.loads(resp.text)["ok"] is False

    def test_browse_folder_success_updates_launch_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui, "_pick_folder_native", lambda: (str(tmp_path), None))
        data = json.loads(asyncio.run(webui.browse_folder(None)).text)
        assert data["path"] == str(tmp_path)
        assert webui.get_launch_dir() == os.path.abspath(str(tmp_path))

    def test_browse_folder_cancel_returns_null_path(self, monkeypatch):
        before = webui.get_launch_dir()
        monkeypatch.setattr(webui, "_pick_folder_native", lambda: (None, None))
        data = json.loads(asyncio.run(webui.browse_folder(None)).text)
        assert data["path"] is None
        assert webui.get_launch_dir() == before  # cancel leaves it unchanged

    def test_browse_folder_error_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            webui, "_pick_folder_native", lambda: (None, "no display here")
        )
        data = json.loads(asyncio.run(webui.browse_folder(None)).text)
        assert data["path"] is None
        assert data["error"] == "no display here"

    def test_dashboard_page_has_project_dir_ui(self):
        html = webui._dashboard_page()
        assert "id='projdir'" in html
        assert "id='browsebtn'" in html
        assert "id='setdirbtn'" in html
        assert "Project directory" in html
        # Wired to the new endpoints.
        assert "/browse-folder" in html
        assert "/set-project-dir" in html
        assert "/get-project-dir" in html


class TestLaunchUsesProjectDir:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(webui.project, "PROJECT_FILE", tmp_path / "current_project")
        monkeypatch.setattr(webui, "resolve_claude_command", lambda cmd: list(cmd))
        monkeypatch.setattr(webui.stats, "proxy_running", lambda *a, **k: True)
        # Every terminal branch needs a hit on Linux CI (no real emulator).
        known = {"x-terminal-emulator", "gnome-terminal", "konsole", "xterm"}
        monkeypatch.setattr(
            webui.shutil, "which",
            lambda name: ("/usr/bin/" + name) if name in known else None,
        )
        saved = webui.get_launch_dir()
        yield
        webui._launch_dir = saved

    def test_launch_passes_selected_dir_as_cwd(self, tmp_path, monkeypatch):
        calls = {}
        monkeypatch.setattr(
            webui.subprocess, "Popen",
            lambda *a, **k: calls.update(a=a, k=k) or None,
        )
        proj = tmp_path / "myproject"
        proj.mkdir()
        assert webui.set_launch_dir(str(proj)) is True
        ok, _ = webui._launch_claude_terminal()
        assert ok is True
        # The chosen folder is handed to the terminal as its working directory.
        assert calls["k"].get("cwd") == webui.get_launch_dir()

    def test_launch_falls_back_to_none_cwd_for_stale_dir(self, tmp_path, monkeypatch):
        calls = {}
        monkeypatch.setattr(
            webui.subprocess, "Popen",
            lambda *a, **k: calls.update(k=k) or None,
        )
        # Point at a directory, then delete it so it's stale at launch time.
        gone = tmp_path / "gone"
        gone.mkdir()
        webui.set_launch_dir(str(gone))
        gone.rmdir()
        ok, _ = webui._launch_claude_terminal()
        assert ok is True
        assert calls["k"].get("cwd") is None  # guarded fallback, no crash

    def test_launch_records_current_project(self, tmp_path, monkeypatch):
        monkeypatch.setattr(webui.subprocess, "Popen", lambda *a, **k: None)
        monkeypatch.delenv("TOKENSNAP_PROJECT", raising=False)
        proj = tmp_path / "tagged"
        proj.mkdir()
        webui.set_launch_dir(str(proj))
        webui._launch_claude_terminal()
        # The proxy reads the current-project state file per request to tag
        # its rows; the launch must have written the selected folder there.
        assert webui.project.get_current_project() == webui.get_launch_dir()
        # And the env var is still set as a fallback.
        assert os.environ["TOKENSNAP_PROJECT"] == webui.get_launch_dir()


class TestMultiLevelStatsEndpoints:
    @pytest.fixture(autouse=True)
    def isolated(self, tmp_path, monkeypatch):
        from tokensnap import config as config_mod, history

        monkeypatch.setattr(webui.config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(webui.config_mod, "CONFIG_FILE", tmp_path / "config.json")
        monkeypatch.setattr(webui.stats, "STATS_DIR", tmp_path)
        monkeypatch.setattr(webui.stats, "STATS_FILE", tmp_path / "stats.json")
        monkeypatch.setattr(history, "DB_FILE", tmp_path / "history.db")
        defaults = dict(config_mod.DEFAULTS)
        # A port nothing is listening on -> proxy_running() is False, no I/O.
        defaults["port"] = 6553
        monkeypatch.setattr(config_mod, "DEFAULTS", defaults)
        # Keep proxy_running from actually probing a socket.
        monkeypatch.setattr(webui.stats, "proxy_running", lambda *a, **k: False)
        yield

    def _json(self, coro):
        return json.loads(asyncio.run(coro).text)

    def test_routes_registered(self):
        app = webui.build_app()
        paths = {r.resource.canonical for r in app.router.routes()}
        assert "/api/stats/session" in paths
        assert "/api/stats/alltime" in paths
        assert "/api/stats/projects" in paths

    def test_alltime_reads_history_db(self):
        from tokensnap import history

        history.log_request(
            model="m", est_tokens_in=100, real_tokens_in=80, real_tokens_out=20,
            cache_read=0, cache_write=0, saved=60, http_status=200, project="p1",
        )
        data = self._json(webui.api_stats_alltime(None))
        assert data["total_requests"] == 1
        assert data["total_est_saved"] == 60
        assert data["total_real_in"] == 80
        assert data["total_real_out"] == 20

    def test_projects_endpoint_orders_and_adds_display_name(self):
        from tokensnap import history

        history.log_request(model="m", est_tokens_in=1, real_tokens_in=1,
                            real_tokens_out=1, cache_read=0, cache_write=0,
                            saved=10, http_status=200, project="C:/work/small")
        history.log_request(model="m", est_tokens_in=1, real_tokens_in=1,
                            real_tokens_out=1, cache_read=0, cache_write=0,
                            saved=900, http_status=200, project="/home/me/big-proj")
        data = self._json(webui.api_stats_projects(None))
        projs = data["projects"]
        assert [p["project"] for p in projs] == ["/home/me/big-proj", "C:/work/small"]
        # Display name is the last path segment, for both separators.
        assert projs[0]["name"] == "big-proj"
        assert projs[1]["name"] == "small"

    def test_projects_endpoint_empty_on_fresh_db(self):
        data = self._json(webui.api_stats_projects(None))
        assert data["projects"] == []

    def test_session_endpoint_reads_stats_json(self):
        webui.stats.mark_started("127.0.0.1", 6553)
        webui.stats.record_request("/v1/messages", "m", 100, 40, 200, 0.1)
        data = self._json(webui.api_stats_session(None))
        assert data["requests"] == 1
        assert data["tokens_saved"] == 60
        assert data["running"] is False

    def test_display_name_helper_handles_edge_cases(self):
        assert webui._project_display_name("unknown") == "unknown"
        assert webui._project_display_name("") == "unknown"
        assert webui._project_display_name("C:/a/b/proj") == "proj"
        assert webui._project_display_name("/x/y/z/") == "z"
        assert webui._project_display_name("bare") == "bare"

    def test_dashboard_page_has_three_level_cards_and_projects(self):
        html = webui._dashboard_page()
        # Three-level summary cards.
        for eid in ("s_saved", "a_saved", "p_count"):
            assert "id='%s'" % eid in html
        # Project stats section + filter + friendly empty message.
        assert "id='projects'" in html
        assert "id='projfilter'" in html
        assert "No projects tracked yet" in html
        # Wired to the new endpoints + localStorage persistence.
        assert "/api/stats/alltime" in html
        assert "/api/stats/projects" in html
        assert "tsnap_project_filter" in html
