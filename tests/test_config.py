"""Offline tests for tokensnap.config: defaults, coercion, and the
keep_last_n -> keep_messages migration/alias.
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

    def test_load_without_file_returns_defaults(self):
        cfg = config_mod.load()
        assert cfg["keep_messages"] == 10


class TestResolveKeyAlias:
    def test_current_name_is_identity(self):
        assert config_mod.resolve_key("keep_messages") == "keep_messages"

    def test_legacy_name_resolves(self):
        assert config_mod.resolve_key("keep_last_n") == "keep_messages"

    def test_unrelated_key_untouched(self):
        assert config_mod.resolve_key("ollama_model") == "ollama_model"


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


class TestLegacyFileMigration:
    def test_old_config_file_migrates_value_on_load(self, tmp_path):
        # Simulates an existing pre-0.3 ~/.tokensnap/config.json (like the
        # one this project's own real installs had) that stores the old key.
        (tmp_path / "config.json").write_text(
            json.dumps({"keep_last_n": 3, "ollama_model": "qwen2.5:7b"}),
            encoding="utf-8",
        )
        cfg = config_mod.load()
        assert cfg["keep_messages"] == 3  # migrated, not reset to the new default
        assert "keep_last_n" not in cfg
        assert cfg["ollama_model"] == "qwen2.5:7b"  # untouched keys preserved

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
