"""Offline tests for tokensnap.ollama and its compressor integration.

No Ollama server is ever contacted: HTTP helpers are monkeypatched.
"""

import json
import urllib.error

import pytest

from tokensnap import compressor, ollama


CFG = {
    "llm_compressor": "auto",
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "testmodel",
    "ollama_timeout": 5.0,
}

GOOD_CARD = {
    "task": "Add JWT authentication to the Flask API",
    "files_modified": ["src/auth.py", "src/app.py"],
    "decisions": ["Use PyJWT instead of authlib (lighter dependency)"],
    "errors_resolved": ["ImportError in tests -> added missing dependency"],
}


@pytest.fixture(autouse=True)
def _fresh_caches():
    ollama.reset_caches()
    yield
    ollama.reset_caches()


def _messages():
    return [
        {"role": "user", "content": "Add JWT auth to my Flask API"},
        {"role": "assistant", "content": "Done, see src/auth.py"},
    ]


def _mock_server(monkeypatch, response_text, fail_ping=False, fail_post=False,
                  tags=None):
    """Patch the HTTP helpers; returns a dict counting calls.

    `tags` overrides the /api/tags model list; by default it reports
    CFG["ollama_model"] ("testmodel") as pulled, so existing behavior
    (server reachable, model present) is the default unless a test asks
    for something else.
    """
    calls = {"ping": 0, "post": 0}
    if tags is None:
        tags = {"models": [{"name": "testmodel"}]}

    def fake_get(url, timeout):
        calls["ping"] += 1
        if fail_ping:
            raise urllib.error.URLError("connection refused")
        return tags

    def fake_post(url, payload, timeout):
        calls["post"] += 1
        calls["last_payload"] = payload
        if fail_post:
            raise urllib.error.URLError("timed out")
        return {"response": response_text}

    monkeypatch.setattr(ollama, "_http_get", fake_get)
    monkeypatch.setattr(ollama, "_http_post_json", fake_post)
    return calls


class TestEnabled:
    def test_modes(self):
        assert ollama.enabled({"llm_compressor": "auto"})
        assert ollama.enabled({"llm_compressor": "ollama"})
        assert ollama.enabled({"llm_compressor": "OLLAMA"})
        assert not ollama.enabled({"llm_compressor": "off"})
        assert not ollama.enabled({})  # missing key = off


class TestAvailability:
    def test_off_mode_never_pings(self, monkeypatch):
        calls = _mock_server(monkeypatch, "{}")
        assert not ollama.is_available({**CFG, "llm_compressor": "off"})
        assert calls["ping"] == 0

    def test_down_server_reported_unavailable(self, monkeypatch):
        calls = _mock_server(monkeypatch, "{}", fail_ping=True)
        assert not ollama.is_available(CFG)
        assert calls["ping"] == 1

    def test_probe_result_is_cached(self, monkeypatch):
        calls = _mock_server(monkeypatch, "{}", fail_ping=True)
        ollama.is_available(CFG)
        ollama.is_available(CFG)
        ollama.is_available(CFG)
        assert calls["ping"] == 1  # cached within the TTL

    def test_url_change_invalidates_cache(self, monkeypatch):
        calls = _mock_server(monkeypatch, "{}")
        assert ollama.is_available(CFG)
        assert ollama.is_available({**CFG, "ollama_url": "http://127.0.0.1:9999"})
        assert calls["ping"] == 2

    def test_server_up_but_model_not_pulled(self, monkeypatch):
        _mock_server(monkeypatch, "{}", tags={"models": [{"name": "other-model"}]})
        assert not ollama.is_available(CFG)

    def test_model_change_invalidates_cache(self, monkeypatch):
        # Same server, different configured model: must re-check, not
        # reuse a cached result for a different model name.
        calls = _mock_server(monkeypatch, "{}", tags={"models": [{"name": "testmodel"}]})
        assert ollama.is_available(CFG)
        assert not ollama.is_available({**CFG, "ollama_model": "other-model"})
        assert calls["ping"] == 2

    def test_model_tag_suffix_matches_base_name(self, monkeypatch):
        # Ollama tags list models with a version suffix (e.g. testmodel:latest);
        # the configured "testmodel" (no suffix) must still match.
        _mock_server(monkeypatch, "{}", tags={"models": [{"name": "testmodel:latest"}]})
        assert ollama.is_available(CFG)

    def test_no_model_configured_is_unavailable(self, monkeypatch):
        _mock_server(monkeypatch, "{}")
        assert not ollama.is_available({**CFG, "ollama_model": ""})

    def test_missing_model_logs_info_in_auto_mode(self, monkeypatch, caplog):
        _mock_server(monkeypatch, "{}", tags={"models": []})
        with caplog.at_level("INFO", logger="tokensnap.ollama"):
            ollama.is_available({**CFG, "llm_compressor": "auto"})
        assert any("not pulled" in r.message for r in caplog.records)

    def test_missing_model_warns_in_explicit_ollama_mode(self, monkeypatch, caplog):
        _mock_server(monkeypatch, "{}", tags={"models": []})
        with caplog.at_level("INFO", logger="tokensnap.ollama"):
            ollama.is_available({**CFG, "llm_compressor": "ollama"})
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("not pulled" in r.message for r in warnings)
        assert any("ollama pull" in r.message for r in warnings)


class TestStatusReason:
    def test_off_mode(self):
        assert ollama.status_reason({"llm_compressor": "off"}) == "regex (llm_compressor=off)"

    def test_available(self, monkeypatch):
        _mock_server(monkeypatch, "{}")
        assert ollama.status_reason(CFG) == "ollama:testmodel"

    def test_server_down(self, monkeypatch):
        _mock_server(monkeypatch, "{}", fail_ping=True)
        assert "no Ollama server answered" in ollama.status_reason(CFG)

    def test_model_missing(self, monkeypatch):
        _mock_server(monkeypatch, "{}", tags={"models": [{"name": "other"}]})
        reason = ollama.status_reason(CFG)
        assert "not pulled" in reason
        assert "testmodel" in reason

    def test_reuses_cached_check_no_duplicate_network_call(self, monkeypatch):
        calls = _mock_server(monkeypatch, "{}")
        ollama.is_available(CFG)
        ollama.status_reason(CFG)
        assert calls["ping"] == 1  # status_reason must not re-probe


class TestGenerateCard:
    def test_success_returns_sanitized_card(self, monkeypatch):
        calls = _mock_server(monkeypatch, json.dumps(GOOD_CARD))
        card = ollama.try_generate_card(_messages(), CFG)
        assert card == GOOD_CARD
        assert calls["post"] == 1
        # Request shape: JSON mode, non-streaming, deterministic
        payload = calls["last_payload"]
        assert payload["model"] == "testmodel"
        assert payload["format"] == "json"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0

    def test_off_mode_returns_none_without_http(self, monkeypatch):
        calls = _mock_server(monkeypatch, json.dumps(GOOD_CARD))
        assert ollama.try_generate_card(
            _messages(), {**CFG, "llm_compressor": "off"}
        ) is None
        assert calls["ping"] == 0 and calls["post"] == 0

    def test_server_down_returns_none(self, monkeypatch):
        calls = _mock_server(monkeypatch, json.dumps(GOOD_CARD), fail_ping=True)
        assert ollama.try_generate_card(_messages(), CFG) is None
        assert calls["post"] == 0

    def test_generation_error_returns_none(self, monkeypatch):
        _mock_server(monkeypatch, "", fail_post=True)
        assert ollama.try_generate_card(_messages(), CFG) is None

    def test_invalid_json_from_model_returns_none(self, monkeypatch):
        _mock_server(monkeypatch, "definitely not json {")
        assert ollama.try_generate_card(_messages(), CFG) is None

    def test_non_dict_json_returns_none(self, monkeypatch):
        _mock_server(monkeypatch, json.dumps(["a", "list"]))
        assert ollama.try_generate_card(_messages(), CFG) is None

    def test_empty_card_returns_none(self, monkeypatch):
        _mock_server(monkeypatch, json.dumps(
            {"task": "", "files_modified": [], "decisions": [], "errors_resolved": []}
        ))
        assert ollama.try_generate_card(_messages(), CFG) is None

    def test_results_are_cached_per_transcript(self, monkeypatch):
        calls = _mock_server(monkeypatch, json.dumps(GOOD_CARD))
        ollama.try_generate_card(_messages(), CFG)
        ollama.try_generate_card(_messages(), CFG)
        assert calls["post"] == 1
        # A different transcript triggers a new generation
        ollama.try_generate_card(
            [{"role": "user", "content": "something else"}], CFG
        )
        assert calls["post"] == 2

    def test_failures_are_cached_too(self, monkeypatch):
        calls = _mock_server(monkeypatch, "not json")
        assert ollama.try_generate_card(_messages(), CFG) is None
        assert ollama.try_generate_card(_messages(), CFG) is None
        assert calls["post"] == 1  # broken model doesn't cost a call per request


class TestSanitize:
    def test_junk_types_dropped(self):
        card = ollama._sanitize(
            {
                "task": 42,  # not a string -> empty
                "files_modified": ["a.py", 7, None, "a.py", "  ", "b.py"],
                "decisions": "not a list",
                "errors_resolved": [{"nested": "dict"}, "real -> fix"],
            }
        )
        assert card["task"] == ""
        assert card["files_modified"] == ["a.py", "b.py"]
        assert card["decisions"] == []
        assert card["errors_resolved"] == ["real -> fix"]

    def test_long_strings_clipped(self):
        card = ollama._sanitize({"task": "x" * 1000, "decisions": ["y" * 1000]})
        assert len(card["task"]) <= 400
        assert len(card["decisions"][0]) <= 200

    def test_item_caps_enforced(self):
        card = ollama._sanitize({"decisions": ["d%d" % i for i in range(50)]})
        assert len(card["decisions"]) == 10

    def test_newlines_collapsed(self):
        card = ollama._sanitize({"task": "line one\nline   two"})
        assert card["task"] == "line one line two"


class TestTranscript:
    def test_roles_and_text_included(self):
        text = ollama.build_transcript(_messages())
        assert "user: Add JWT auth" in text
        assert "assistant: Done" in text

    def test_long_history_keeps_the_tail(self):
        msgs = [
            {"role": "user", "content": "old %d " % i + "x" * 500}
            for i in range(100)
        ]
        msgs.append({"role": "user", "content": "THE-NEWEST-MESSAGE"})
        text = ollama.build_transcript(msgs)
        assert len(text) <= ollama._MAX_TRANSCRIPT_CHARS + 1
        assert "THE-NEWEST-MESSAGE" in text
        assert "old 0" not in text


class TestCompressorIntegration:
    def _history(self):
        msgs = []
        for i in range(7):
            msgs.append({"role": "user", "content": "step %d on main.py" % i})
            msgs.append({"role": "assistant", "content": "did step %d" % i})
        return msgs

    def test_llm_card_merged_over_regex(self, monkeypatch):
        _mock_server(monkeypatch, json.dumps(GOOD_CARD))
        card, out = compressor.compress_messages(
            self._history(), keep_last_n=2, min_messages=8, llm_cfg=CFG
        )
        assert card is not None and len(out) == 4
        payload = json.loads(card[card.index("{"): card.rindex("}") + 1])
        # LLM fields win...
        assert payload["task"] == GOOD_CARD["task"]
        assert payload["decisions"] == GOOD_CARD["decisions"]
        assert payload["generator"] == "ollama:testmodel"
        # ...but regex-found files are kept and merged with the LLM's
        assert "main.py" in payload["files_modified"]
        assert "src/auth.py" in payload["files_modified"]
        # Bookkeeping fields still present
        assert payload["messages_summarized"] == 10
        assert payload["original_tokens"] > 0

    def test_llm_failure_falls_back_to_regex(self, monkeypatch):
        _mock_server(monkeypatch, "garbage")
        card, _ = compressor.compress_messages(
            self._history(), keep_last_n=2, min_messages=8, llm_cfg=CFG
        )
        payload = json.loads(card[card.index("{"): card.rindex("}") + 1])
        assert "generator" not in payload  # pure regex card
        assert payload["task"].startswith("step 0")
        assert "main.py" in payload["files_modified"]

    def test_no_llm_cfg_never_touches_ollama(self, monkeypatch):
        calls = _mock_server(monkeypatch, json.dumps(GOOD_CARD))
        card, _ = compressor.compress_messages(
            self._history(), keep_last_n=2, min_messages=8
        )
        assert card is not None
        assert calls["ping"] == 0 and calls["post"] == 0


class TestConfig:
    def test_new_defaults_present(self):
        from tokensnap import config as config_mod

        for key in ("llm_compressor", "ollama_url", "ollama_model", "ollama_timeout"):
            assert key in config_mod.DEFAULTS
        assert config_mod.DEFAULTS["llm_compressor"] == "auto"

    def test_llm_compressor_choices_validated(self, tmp_path, monkeypatch):
        from tokensnap import config as config_mod

        monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
        assert config_mod.set_value("llm_compressor", "OFF") == "off"
        assert config_mod.load()["llm_compressor"] == "off"
        with pytest.raises(ValueError):
            config_mod.set_value("llm_compressor", "gpt4")

    def test_ollama_timeout_coerced_to_float(self, tmp_path, monkeypatch):
        from tokensnap import config as config_mod

        monkeypatch.setattr(config_mod, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "CONFIG_FILE", tmp_path / "config.json")
        assert config_mod.set_value("ollama_timeout", "2.5") == 2.5
