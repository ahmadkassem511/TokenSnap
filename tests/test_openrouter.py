"""Offline tests for tokensnap.openrouter and its compressor integration.

No OpenRouter server is ever contacted: the HTTP helper is monkeypatched.
"""

import json
import urllib.error

import pytest

from tokensnap import compressor, openrouter


CFG = {
    "compressor_type": "openrouter",
    "openrouter_api_key": "sk-or-test",
    "openrouter_model": "testmodel",
}

# The model is asked for key_files/important_decisions (see the system
# prompt); try_generate_card()/_sanitize() remap these onto the canonical
# files_modified/decisions names the rest of Tokensnap uses.
RAW_MODEL_CARD = {
    "task": "Add JWT authentication to the Flask API",
    "key_files": ["src/auth.py", "src/app.py"],
    "important_decisions": ["Use PyJWT instead of authlib (lighter dependency)"],
    "errors_resolved": ["ImportError in tests -> added missing dependency"],
}

GOOD_CARD = {
    "task": RAW_MODEL_CARD["task"],
    "files_modified": RAW_MODEL_CARD["key_files"],
    "decisions": RAW_MODEL_CARD["important_decisions"],
    "errors_resolved": RAW_MODEL_CARD["errors_resolved"],
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    openrouter.reset_caches()
    yield
    openrouter.reset_caches()


def _messages():
    return [
        {"role": "user", "content": "Add JWT auth to my Flask API"},
        {"role": "assistant", "content": "Done, see src/auth.py"},
    ]


def _mock_api(monkeypatch, model_output_text, fail=False):
    """Patch the HTTP POST helper; returns a dict counting/recording calls."""
    calls = {"post": 0}

    def fake_post(url, payload, headers, timeout):
        calls["post"] += 1
        calls["last_payload"] = payload
        calls["last_headers"] = headers
        if fail:
            raise urllib.error.URLError("timed out")
        return {"choices": [{"message": {"content": model_output_text}}]}

    monkeypatch.setattr(openrouter, "_http_post_json", fake_post)
    return calls


class TestEnabled:
    def test_requires_type_and_key(self):
        assert openrouter.enabled(CFG)
        assert not openrouter.enabled({**CFG, "openrouter_api_key": ""})
        assert not openrouter.enabled({**CFG, "compressor_type": "regex"})
        assert not openrouter.enabled({**CFG, "compressor_type": "off"})
        assert not openrouter.enabled({})


class TestSummarize:
    def test_sends_bearer_key_and_json_mode(self, monkeypatch):
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        text = openrouter.summarize("some transcript", "testmodel", "sk-or-test")
        assert text == json.dumps(RAW_MODEL_CARD)
        assert calls["post"] == 1
        payload = calls["last_payload"]
        assert payload["model"] == "testmodel"
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["temperature"] == 0
        assert calls["last_headers"]["Authorization"] == "Bearer sk-or-test"

    def test_never_sends_anthropic_key(self, monkeypatch):
        # summarize() only ever accepts an OpenRouter key as its explicit
        # api_key argument - there is no code path for an Anthropic key to
        # leak in, but assert the header shape stays exactly that anyway.
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        openrouter.summarize("text", "testmodel", "sk-or-only-this-one")
        assert calls["last_headers"]["Authorization"] == "Bearer sk-or-only-this-one"

    def test_raises_on_no_choices(self, monkeypatch):
        def fake_post(url, payload, headers, timeout):
            return {"choices": []}

        monkeypatch.setattr(openrouter, "_http_post_json", fake_post)
        with pytest.raises(ValueError):
            openrouter.summarize("text", "testmodel", "key")


class TestTryGenerateCard:
    def test_success_returns_remapped_card(self, monkeypatch):
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        card = openrouter.try_generate_card(_messages(), CFG)
        assert card == GOOD_CARD
        assert calls["post"] == 1

    def test_disabled_returns_none_without_http(self, monkeypatch):
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        assert openrouter.try_generate_card(
            _messages(), {**CFG, "compressor_type": "regex"}
        ) is None
        assert calls["post"] == 0

    def test_no_key_returns_none_without_http(self, monkeypatch):
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        assert openrouter.try_generate_card(
            _messages(), {**CFG, "openrouter_api_key": ""}
        ) is None
        assert calls["post"] == 0

    def test_network_failure_returns_none(self, monkeypatch):
        calls = _mock_api(monkeypatch, "", fail=True)
        assert openrouter.try_generate_card(_messages(), CFG) is None
        assert calls["post"] == 1

    def test_invalid_json_from_model_returns_none(self, monkeypatch):
        _mock_api(monkeypatch, "definitely not json {")
        assert openrouter.try_generate_card(_messages(), CFG) is None

    def test_non_dict_json_returns_none(self, monkeypatch):
        _mock_api(monkeypatch, json.dumps(["a", "list"]))
        assert openrouter.try_generate_card(_messages(), CFG) is None

    def test_empty_card_returns_none(self, monkeypatch):
        _mock_api(monkeypatch, json.dumps(
            {"task": "", "key_files": [], "important_decisions": [], "errors_resolved": []}
        ))
        assert openrouter.try_generate_card(_messages(), CFG) is None

    def test_results_are_cached_per_transcript(self, monkeypatch):
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        openrouter.try_generate_card(_messages(), CFG)
        openrouter.try_generate_card(_messages(), CFG)
        assert calls["post"] == 1
        openrouter.try_generate_card(
            [{"role": "user", "content": "something else"}], CFG
        )
        assert calls["post"] == 2

    def test_failures_are_cached_too(self, monkeypatch):
        calls = _mock_api(monkeypatch, "not json")
        assert openrouter.try_generate_card(_messages(), CFG) is None
        assert openrouter.try_generate_card(_messages(), CFG) is None
        assert calls["post"] == 1  # a broken model doesn't cost a call per request


class TestSanitize:
    def test_junk_types_dropped(self):
        card = openrouter._sanitize(
            {
                "task": 42,  # not a string -> empty
                "key_files": ["a.py", 7, None, "a.py", "  ", "b.py"],
                "important_decisions": "not a list",
                "errors_resolved": [{"nested": "dict"}, "real -> fix"],
            }
        )
        assert card["task"] == ""
        assert card["files_modified"] == ["a.py", "b.py"]
        assert card["decisions"] == []
        assert card["errors_resolved"] == ["real -> fix"]

    def test_long_strings_clipped(self):
        card = openrouter._sanitize({"task": "x" * 1000, "important_decisions": ["y" * 1000]})
        assert len(card["task"]) <= 400
        assert len(card["decisions"][0]) <= 200

    def test_item_caps_enforced(self):
        card = openrouter._sanitize({"important_decisions": ["d%d" % i for i in range(50)]})
        assert len(card["decisions"]) == 10

    def test_newlines_collapsed(self):
        card = openrouter._sanitize({"task": "line one\nline   two"})
        assert card["task"] == "line one line two"

    def test_non_dict_returns_none(self):
        assert openrouter._sanitize(["not", "a", "dict"]) is None
        assert openrouter._sanitize(None) is None


class TestTranscript:
    def test_roles_and_text_included(self):
        text = openrouter.build_transcript(_messages())
        assert "user: Add JWT auth" in text
        assert "assistant: Done" in text

    def test_long_history_keeps_the_tail(self):
        msgs = [
            {"role": "user", "content": "old %d " % i + "x" * 500}
            for i in range(100)
        ]
        msgs.append({"role": "user", "content": "THE-NEWEST-MESSAGE"})
        text = openrouter.build_transcript(msgs)
        assert len(text) <= openrouter._MAX_TRANSCRIPT_CHARS + 1
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
        _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        card, out = compressor.compress_messages(
            self._history(), keep_last_n=2, min_messages=8, llm_cfg=CFG
        )
        assert card is not None and len(out) == 4
        payload = json.loads(card[card.index("{"): card.rindex("}") + 1])
        # LLM fields win...
        assert payload["task"] == GOOD_CARD["task"]
        assert payload["decisions"] == GOOD_CARD["decisions"]
        assert payload["generator"] == "openrouter:testmodel"
        # ...but regex-found files are kept and merged with the LLM's
        assert "main.py" in payload["files_modified"]
        assert "src/auth.py" in payload["files_modified"]
        # Bookkeeping fields still present
        assert payload["messages_summarized"] == 10
        assert payload["original_tokens"] > 0

    def test_llm_failure_falls_back_to_regex(self, monkeypatch):
        _mock_api(monkeypatch, "garbage")
        card, _ = compressor.compress_messages(
            self._history(), keep_last_n=2, min_messages=8, llm_cfg=CFG
        )
        payload = json.loads(card[card.index("{"): card.rindex("}") + 1])
        assert "generator" not in payload  # pure regex card
        assert payload["task"].startswith("step 0")
        assert "main.py" in payload["files_modified"]

    def test_no_llm_cfg_never_touches_openrouter(self, monkeypatch):
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        card, _ = compressor.compress_messages(
            self._history(), keep_last_n=2, min_messages=8
        )
        assert card is not None
        assert calls["post"] == 0

    def test_regex_compressor_type_never_touches_openrouter(self, monkeypatch):
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        card, _ = compressor.compress_messages(
            self._history(), keep_last_n=2, min_messages=8,
            llm_cfg={**CFG, "compressor_type": "regex"},
        )
        assert card is not None
        assert calls["post"] == 0


class TestMemoryCardStatus:
    def test_off(self):
        status = compressor.memory_card_status({"compressor_type": "off"})
        assert status.startswith("off")

    def test_regex(self):
        assert compressor.memory_card_status({"compressor_type": "regex"}) == "regex"

    def test_openrouter_no_key(self):
        status = compressor.memory_card_status({"compressor_type": "openrouter"})
        assert status.startswith("regex")
        assert "openrouter_api_key" in status

    def test_openrouter_with_key(self):
        status = compressor.memory_card_status(
            {"compressor_type": "openrouter", "openrouter_api_key": "x",
             "openrouter_model": "m"}
        )
        assert status == "openrouter:m"
