"""Offline tests for tokensnap.config: defaults, coercion, and migrations
from pre-0.4 (Ollama-based) and pre-0.3 (keep_last_n) config files.
"""

import json

import pytest

from tokensnap import config as config_mod


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
    yield tmp_path


class TestDefaults:
    def test_keep_messages_default_is_ten(self):
        assert config_mod.DEFAULTS["keep_messages"] == 10

    def test_context_threshold_default_is_95_percent(self):
        assert config_mod.DEFAULTS["context_threshold"] == 0.95

    def test_no_stale_keep_last_n_key(self):
        # keep_last_n was renamed; it must not linger as a live default.
        assert "keep_last_n" not in config_mod.DEFAULTS

    def test_no_ollama_keys(self):
        # Ollama was replaced by OpenRouter; none of its keys should remain.
        for key in ("llm_compressor", "ollama_url", "ollama_model", "ollama_timeout"):
            assert key not in config_mod.DEFAULTS

    def test_openrouter_defaults_present(self):
        assert config_mod.DEFAULTS["compressor_type"] == "regex"
        assert config_mod.DEFAULTS["openrouter_api_key"] == ""
        assert config_mod.DEFAULTS["openrouter_model"] == "meta-llama/llama-3.1-8b-instruct:free"

    def test_selective_compression_defaults_true(self):
        assert config_mod.DEFAULTS["selective_compression"] is True

    def test_load_without_file_returns_defaults(self):
        cfg = config_mod.load()
        assert cfg["keep_messages"] == 10
        assert cfg["compressor_type"] == "regex"
        assert cfg["selective_compression"] is True


class TestResolveKeyAlias:
    def test_current_name_is_identity(self):
        assert config_mod.resolve_key("keep_messages") == "keep_messages"

    def test_legacy_name_resolves(self):
        assert config_mod.resolve_key("keep_last_n") == "keep_messages"

    def test_unrelated_key_untouched(self):
        assert config_mod.resolve_key("openrouter_model") == "openrouter_model"


class TestSetValue:
    def test_set_keep_messages_directly(self):
        assert config_mod.set_value("keep_messages", "15") == 15
        assert config_mod.load()["keep_messages"] == 15

    def test_set_via_legacy_alias(self):
        assert config_mod.set_value("keep_last_n", "20") == 20
        assert config_mod.load()["keep_messages"] == 20
        assert "keep_last_n" not in config_mod.load()

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            config_mod.set_value("not_a_real_key", "1")

    def test_context_threshold_still_coerces_to_float(self):
        assert config_mod.set_value("context_threshold", "0.85") == 0.85

    def test_set_openrouter_api_key(self):
        assert config_mod.set_value("openrouter_api_key", "sk-or-test") == "sk-or-test"
        assert config_mod.load()["openrouter_api_key"] == "sk-or-test"

    def test_set_openrouter_model(self):
        assert config_mod.set_value("openrouter_model", "some/model") == "some/model"

    def test_compressor_type_choices_validated(self):
        assert config_mod.set_value("compressor_type", "OPENROUTER") == "openrouter"
        assert config_mod.load()["compressor_type"] == "openrouter"
        with pytest.raises(ValueError):
            config_mod.set_value("compressor_type", "ollama")

    def test_ollama_key_no_longer_settable(self):
        # Ollama keys were removed outright, not aliased to anything.
        with pytest.raises(KeyError):
            config_mod.set_value("ollama_model", "x")

    def test_compression_level_defaults_to_adaptive(self):
        assert config_mod.load()["compression_level"] == "adaptive"

    def test_compression_level_choices_validated(self):
        assert config_mod.set_value("compression_level", "FULL") == "full"
        assert config_mod.load()["compression_level"] == "full"
        with pytest.raises(ValueError):
            config_mod.set_value("compression_level", "extreme")

    @pytest.mark.parametrize(
        "raw,expected",
        [("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
         ("false", False), ("0", False), ("no", False), ("off", False)],
    )
    def test_selective_compression_bool_coercion(self, raw, expected):
        assert config_mod.set_value("selective_compression", raw) is expected

    def test_selective_compression_rejects_junk(self):
        with pytest.raises(ValueError):
            config_mod.set_value("selective_compression", "maybe")


class TestLegacyFileMigration:
    def test_keep_last_n_migrates_value_on_load(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"keep_last_n": 3, "host": "0.0.0.0"}), encoding="utf-8"
        )
        cfg = config_mod.load()
        assert cfg["keep_messages"] == 3  # migrated, not reset to the new default
        assert "keep_last_n" not in cfg
        assert cfg["host"] == "0.0.0.0"  # untouched keys preserved

    def test_new_key_wins_if_both_present(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"keep_last_n": 3, "keep_messages": 12}), encoding="utf-8"
        )
        cfg = config_mod.load()
        assert cfg["keep_messages"] == 12
        assert "keep_last_n" not in cfg

    def test_migration_does_not_mutate_file_on_disk(self, tmp_path):
        # load() only migrates the in-memory dict; the file itself is
        # rewritten only on the next explicit save() (e.g. `config set`).
        (tmp_path / "config.json").write_text(
            json.dumps({"keep_last_n": 4}), encoding="utf-8"
        )
        config_mod.load()
        on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert on_disk == {"keep_last_n": 4}

    def test_broken_file_falls_back_to_defaults(self, tmp_path):
        (tmp_path / "config.json").write_text("{not valid json", encoding="utf-8")
        cfg = config_mod.load()
        assert cfg["keep_messages"] == 10

    @pytest.mark.parametrize("old_value", ["auto", "ollama", "off"])
    def test_llm_compressor_migrates_to_regex(self, tmp_path, old_value):
        # All three pre-0.4 modes safely become "regex": there's no
        # OpenRouter key to carry over, and "regex" preserves the old
        # truncation behavior exactly (new "off" would disable it).
        (tmp_path / "config.json").write_text(
            json.dumps({"llm_compressor": old_value}), encoding="utf-8"
        )
        cfg = config_mod.load()
        assert cfg["compressor_type"] == "regex"
        assert "llm_compressor" not in cfg

    def test_llm_compressor_does_not_override_explicit_compressor_type(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"llm_compressor": "auto", "compressor_type": "openrouter"}),
            encoding="utf-8",
        )
        cfg = config_mod.load()
        assert cfg["compressor_type"] == "openrouter"

    def test_ollama_keys_dropped_on_load(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "llm_compressor": "auto",
                    "ollama_url": "http://127.0.0.1:11434",
                    "ollama_model": "qwen2.5:7b",
                    "ollama_timeout": 10.0,
                }
            ),
            encoding="utf-8",
        )
        cfg = config_mod.load()
        for key in ("llm_compressor", "ollama_url", "ollama_model", "ollama_timeout"):
            assert key not in cfg
        assert cfg["compressor_type"] == "regex"

    def test_real_world_pre_0_4_file_migrates_cleanly(self, tmp_path):
        # The exact shape this project's own ~/.tokensnap/config.json had
        # before the OpenRouter rewrite.
        (tmp_path / "config.json").write_text(
            json.dumps(
                {
                    "host": "127.0.0.1",
                    "port": 8889,
                    "upstream": "https://api.anthropic.com",
                    "keep_messages": 15,
                    "aggressive_keep_last_n": 2,
                    "context_threshold": 0.9,
                    "min_messages_to_compress": 8,
                    "llm_compressor": "auto",
                    "ollama_url": "http://127.0.0.1:11434",
                    "ollama_model": "qwen2.5:3b",
                    "ollama_timeout": 10.0,
                    "key": "",
                    "log_level": "INFO",
                }
            ),
            encoding="utf-8",
        )
        cfg = config_mod.load()
        assert cfg["keep_messages"] == 15
        assert cfg["context_threshold"] == 0.9
        assert cfg["compressor_type"] == "regex"
        assert cfg["selective_compression"] is True  # not in the file -> default
        assert cfg["openrouter_api_key"] == ""
        for key in ("llm_compressor", "ollama_url", "ollama_model", "ollama_timeout"):
            assert key not in cfg
