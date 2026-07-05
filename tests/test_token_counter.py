"""Offline tests for tokensnap.token_counter."""

import pytest

from tokensnap import token_counter


@pytest.fixture
def fallback_mode(monkeypatch):
    """Force the chars/4 fallback path (as if tiktoken were unavailable)."""
    monkeypatch.setattr(token_counter, "_encoder", False)


class TestCountTokens:
    def test_empty_is_zero(self):
        assert token_counter.count_tokens("") == 0

    def test_monotonic(self):
        short = token_counter.count_tokens("hello world")
        long = token_counter.count_tokens("hello world " * 200)
        assert 0 < short < long

    def test_fallback_estimate(self, fallback_mode):
        assert token_counter.count_tokens("a" * 400) == 100

    def test_fallback_minimum_one(self, fallback_mode):
        assert token_counter.count_tokens("ab") == 1


class TestCountMessageTokens:
    def test_string_content(self, fallback_mode):
        messages = [{"role": "user", "content": "a" * 400}]
        # 100 for text + 4 framing overhead
        assert token_counter.count_message_tokens(messages) == 104

    def test_block_content_and_system(self, fallback_mode):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a" * 40},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": "b" * 40}],
                    },
                ],
            },
        ]
        total = token_counter.count_message_tokens(messages, system="c" * 40)
        assert total == 10 + 10 + 10 + 4

    def test_tool_use_counts_input(self, fallback_mode):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash",
                     "input": {"command": "x" * 100}},
                ],
            },
        ]
        assert token_counter.count_message_tokens(messages) > 20

    def test_image_flat_cost(self, fallback_mode):
        messages = [
            {"role": "user", "content": [{"type": "image", "source": {}}]},
        ]
        assert token_counter.count_message_tokens(messages) == 1500 + 4

    def test_empty_messages(self):
        assert token_counter.count_message_tokens([]) == 0


class TestContextWindow:
    def test_default_200k(self):
        assert token_counter.context_window_for("claude-sonnet-5") == 200_000

    def test_unknown_model(self):
        assert token_counter.context_window_for(None) == 200_000
        assert token_counter.context_window_for("mystery") == 200_000

    def test_1m_variants(self):
        assert token_counter.context_window_for("claude-sonnet-4[1m]") == 1_000_000
        assert token_counter.context_window_for("claude-context-1m-beta") == 1_000_000


class TestNearLimit:
    def test_below_threshold(self):
        assert not token_counter.near_limit(100_000, "claude-sonnet-5", 0.9)

    def test_at_threshold(self):
        assert token_counter.near_limit(180_000, "claude-sonnet-5", 0.9)

    def test_above_threshold(self):
        assert token_counter.near_limit(199_999, "claude-sonnet-5", 0.9)
