"""Offline tests for tokensnap.openrouter and its compressor integration.

No OpenRouter server is ever contacted: the HTTP helper is monkeypatched.
"""

import json
import time
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


def _mock_api(monkeypatch, model_output_text, fail=False, resp_headers=None):
    """Patch the HTTP POST helper; returns a dict counting/recording calls."""
    calls = {"post": 0}

    def fake_post(url, payload, headers, timeout):
        calls["post"] += 1
        calls["last_payload"] = payload
        calls["last_headers"] = headers
        if fail:
            raise urllib.error.URLError("timed out")
        body = {"choices": [{"message": {"content": model_output_text}}]}
        return body, (resp_headers or {})

    monkeypatch.setattr(openrouter, "_http_post_json", fake_post)
    return calls


def _mock_multi_model_api(monkeypatch, behavior_by_model):
    """Patch the HTTP POST helper with per-model behavior.

    `behavior_by_model[model]` is either a string (successful model output
    text) or an exception instance to raise for that model.
    """
    calls = {"post": 0, "models_tried": []}

    def fake_post(url, payload, headers, timeout):
        calls["post"] += 1
        model = payload["model"]
        calls["models_tried"].append(model)
        behavior = behavior_by_model[model]
        if isinstance(behavior, BaseException):
            raise behavior
        return {"choices": [{"message": {"content": behavior}}]}, {}

    monkeypatch.setattr(openrouter, "_http_post_json", fake_post)
    return calls


def _http_error(code, headers=None):
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/chat/completions", code, "err",
        headers or {}, None,
    )


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


class TestFallbackAndRetry:
    @pytest.fixture(autouse=True)
    def no_real_sleep(self, monkeypatch):
        # Retry delays are real in production; tests must not actually wait.
        self.slept = []
        monkeypatch.setattr(openrouter, "_sleep", lambda s: self.slept.append(s))

    def test_falls_back_to_second_model_on_429(self, monkeypatch):
        calls = _mock_multi_model_api(monkeypatch, {
            "primary": _http_error(429),
            "backup": json.dumps(RAW_MODEL_CARD),
        })
        cfg = {**CFG, "openrouter_model": "primary",
               "openrouter_fallback_models": ["backup"]}
        card = openrouter.try_generate_card(_messages(), cfg)
        assert card == GOOD_CARD
        assert calls["models_tried"] == ["primary", "backup"]
        assert self.slept == [5]  # default openrouter_retry_delay_seconds

    def test_uses_configured_retry_delay(self, monkeypatch):
        _mock_multi_model_api(monkeypatch, {
            "primary": _http_error(503),
            "backup": json.dumps(RAW_MODEL_CARD),
        })
        cfg = {**CFG, "openrouter_model": "primary",
               "openrouter_fallback_models": ["backup"],
               "openrouter_retry_delay_seconds": 2}
        openrouter.try_generate_card(_messages(), cfg)
        assert self.slept == [2]

    def test_non_retryable_error_skips_fallback_entirely(self, monkeypatch):
        calls = _mock_multi_model_api(monkeypatch, {
            "primary": _http_error(401),  # auth error - retrying won't help
            "backup": json.dumps(RAW_MODEL_CARD),
        })
        cfg = {**CFG, "openrouter_model": "primary",
               "openrouter_fallback_models": ["backup"]}
        card = openrouter.try_generate_card(_messages(), cfg)
        assert card is None
        assert calls["models_tried"] == ["primary"]  # never tried backup
        assert self.slept == []

    def test_max_retries_caps_total_attempts(self, monkeypatch):
        calls = _mock_multi_model_api(monkeypatch, {
            "primary": _http_error(429),
            "backup1": _http_error(429),
            "backup2": json.dumps(RAW_MODEL_CARD),
        })
        cfg = {**CFG, "openrouter_model": "primary",
               "openrouter_fallback_models": ["backup1", "backup2"],
               "openrouter_max_retries": 1}  # only 1 retry beyond primary
        card = openrouter.try_generate_card(_messages(), cfg)
        assert card is None  # backup2 never reached
        assert calls["models_tried"] == ["primary", "backup1"]

    def test_empty_fallback_list_behaves_like_before(self, monkeypatch):
        calls = _mock_api(monkeypatch, "", fail=True)
        card = openrouter.try_generate_card(_messages(), CFG)
        assert card is None
        assert calls["post"] == 1

    def test_fallback_active_flag_set_after_using_a_fallback(self, monkeypatch):
        _mock_multi_model_api(monkeypatch, {
            "primary": _http_error(429),
            "backup": json.dumps(RAW_MODEL_CARD),
        })
        cfg = {**CFG, "openrouter_model": "primary",
               "openrouter_fallback_models": ["backup"]}
        assert openrouter.status_snapshot()["fallback_active"] is False
        openrouter.try_generate_card(_messages(), cfg)
        assert openrouter.status_snapshot()["fallback_active"] is True


class TestCooldown:
    @pytest.fixture(autouse=True)
    def no_real_sleep(self, monkeypatch):
        monkeypatch.setattr(openrouter, "_sleep", lambda s: None)

    def test_all_models_failing_enters_cooldown(self, monkeypatch):
        _mock_api(monkeypatch, "", fail=True)
        assert openrouter.try_generate_card(_messages(), CFG) is None
        assert openrouter.in_cooldown() is True

    def test_cooldown_skips_further_calls_without_http(self, monkeypatch):
        calls = _mock_api(monkeypatch, "", fail=True)
        openrouter.try_generate_card(_messages(), CFG)
        assert calls["post"] == 1
        # A different transcript would normally trigger a fresh call, but
        # cooldown must suppress it entirely.
        different = [{"role": "user", "content": "a completely different task"}]
        assert openrouter.try_generate_card(different, CFG) is None
        assert calls["post"] == 1

    def test_success_never_triggers_cooldown(self, monkeypatch):
        _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        openrouter.try_generate_card(_messages(), CFG)
        assert openrouter.in_cooldown() is False

    def test_cached_hit_bypasses_cooldown_check(self, monkeypatch):
        # A transcript that was already successfully cached should still
        # return its cached card even while OpenRouter is in cooldown from
        # unrelated failures.
        calls = _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD))
        card = openrouter.try_generate_card(_messages(), CFG)
        assert card == GOOD_CARD
        # Force a cooldown window without another failed call.
        openrouter._cooldown_until = time.monotonic() + 60
        assert openrouter.try_generate_card(_messages(), CFG) == GOOD_CARD
        assert calls["post"] == 1  # served from cache, no new HTTP call


class TestRateLimitCapture:
    def test_headers_captured_on_success(self, monkeypatch):
        _mock_api(
            monkeypatch, json.dumps(RAW_MODEL_CARD),
            resp_headers={"X-RateLimit-Remaining": "42", "X-RateLimit-Reset": "1700000000"},
        )
        openrouter.try_generate_card(_messages(), CFG)
        snap = openrouter.status_snapshot()
        assert snap["rate_limit_remaining"] == "42"
        assert snap["rate_limit_reset"] == "1700000000"

    def test_headers_captured_on_429_before_reraise(self, monkeypatch):
        def fake_post(url, payload, headers, timeout):
            raise _http_error(429, headers={"X-RateLimit-Remaining": "0"})

        monkeypatch.setattr(openrouter, "_http_post_json", fake_post)
        openrouter.try_generate_card(_messages(), CFG)
        assert openrouter.status_snapshot()["rate_limit_remaining"] == "0"

    def test_missing_headers_is_safe(self, monkeypatch):
        _mock_api(monkeypatch, json.dumps(RAW_MODEL_CARD), resp_headers={})
        openrouter.try_generate_card(_messages(), CFG)
        snap = openrouter.status_snapshot()
        assert snap["rate_limit_remaining"] is None

    def test_persisted_to_stats(self, monkeypatch, tmp_path):
        from tokensnap import stats

        monkeypatch.setattr(stats, "STATS_DIR", tmp_path)
        monkeypatch.setattr(stats, "STATS_FILE", tmp_path / "stats.json")
        _mock_api(
            monkeypatch, json.dumps(RAW_MODEL_CARD),
            resp_headers={"X-RateLimit-Remaining": "7", "X-RateLimit-Reset": "123"},
        )
        openrouter.try_generate_card(_messages(), CFG)
        data = stats.load()
        assert data["openrouter_rate_limit"] == {"remaining": "7", "reset": "123"}


class TestStatusSnapshotAndResetCaches:
    def test_reset_caches_clears_everything(self, monkeypatch):
        _mock_api(monkeypatch, "", fail=True)
        openrouter.try_generate_card(_messages(), CFG)
        assert openrouter.in_cooldown() is True
        openrouter.reset_caches()
        assert openrouter.in_cooldown() is False
        snap = openrouter.status_snapshot()
        assert snap["recent_errors"] == []
        assert snap["fallback_active"] is False


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
