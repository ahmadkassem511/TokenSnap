"""Offline tests for tokensnap.webui pure helpers and app wiring.

These avoid binding a real socket - they exercise the model sizing, input
validation, log tailing, setup-marker, and route table without a running
server. The setup marker and log path are bound to config_mod.CONFIG_DIR at
import time, so tests monkeypatch the already-resolved module attributes.
"""

import pytest

from tokensnap import webui


class TestRecommendModel:
    def test_low_ram_gets_small_model(self):
        assert webui.recommend_model(4) == "qwen2.5:3b"

    def test_mid_ram_gets_7b(self):
        assert webui.recommend_model(8) == "qwen2.5:7b"
        assert webui.recommend_model(16) == "qwen2.5:7b"

    def test_high_ram_gets_14b(self):
        assert webui.recommend_model(32) == "qwen2.5:14b"

    def test_boundary_just_under_8(self):
        assert webui.recommend_model(7.9) == "qwen2.5:3b"


class TestValidModel:
    @pytest.mark.parametrize("m", ["qwen2.5:7b", "llama3.1:8b", "qwen2.5-coder:14b", "a/b:c"])
    def test_accepts_normal_model_tags(self, m):
        assert webui._valid_model(m) is True

    @pytest.mark.parametrize("m", ["", None, "bad model", "x;rm -rf", "a" * 200, "a b"])
    def test_rejects_junk(self, m):
        assert webui._valid_model(m) is False


class TestHardwareInfo:
    def test_reports_expected_keys(self):
        info = webui.hardware_info()
        for key in ("os", "cpu_cores", "ram_gb", "disk_free_gb"):
            assert key in info
        assert info["cpu_cores"] >= 1


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
