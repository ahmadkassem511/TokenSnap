"""Offline tests for tokensnap.webui pure helpers and app wiring.

These avoid binding a real socket - they exercise log tailing, setup-marker,
route table, and page rendering without a running server. The setup marker
and log path are bound to config_mod.CONFIG_DIR at import time, so tests
monkeypatch the already-resolved module attributes.
"""

import json
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
