"""Offline tests for tokensnap.compressor."""

import json

from tokensnap import compressor
from tokensnap.utils import is_tool_result_only


def _exchange(user_text, assistant_text):
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _long_history():
    """A synthetic 14-message transcript with extractable facts."""
    msgs = []
    msgs += _exchange(
        "Please add JWT authentication to my Flask API",
        "Sure. Decision: we will use PyJWT instead of using authlib. "
        "I'll modify src/auth.py and src/app.py.",
    )
    msgs += _exchange(
        "The tests are failing",
        "I see an Error: ImportError in tests/test_auth.py.\n"
        "That is now fixed by adding the missing dependency.",
    )
    msgs += _exchange("Also update the README please", "Updated README.md accordingly.")
    msgs += _exchange("Add token refresh", "Done, see src/auth.py.")
    msgs += _exchange("Now add logout", "Added logout endpoint in src/app.py.")
    msgs += _exchange("Write tests for logout", "Added tests/test_logout.py.")
    msgs += _exchange("Run the tests", "All 12 tests are passing now.")
    return msgs  # 14 messages


class TestCompressMessages:
    def test_short_history_untouched(self):
        msgs = _exchange("hi", "hello")
        card, out = compressor.compress_messages(msgs, keep_last_n=3, min_messages=8)
        assert card is None
        assert out == msgs

    def test_empty_history(self):
        card, out = compressor.compress_messages([], keep_last_n=3)
        assert card is None
        assert out == []

    def test_long_history_compressed(self):
        msgs = _long_history()
        card, out = compressor.compress_messages(msgs, keep_last_n=3, min_messages=8)
        assert card is not None
        assert compressor.MEMORY_CARD_HEADER in card
        assert len(out) == 6  # last 3 exchanges kept verbatim
        assert out == msgs[-6:]

    def test_kept_tail_starts_with_clean_user_message(self):
        msgs = _long_history()
        _, out = compressor.compress_messages(msgs, keep_last_n=3, min_messages=8)
        assert out[0]["role"] == "user"
        assert not is_tool_result_only(out[0])

    def test_card_contains_extracted_facts(self):
        msgs = _long_history()
        card, _ = compressor.compress_messages(msgs, keep_last_n=2, min_messages=8)
        payload = json.loads(card[card.index("{") : card.rindex("}") + 1])
        assert "JWT authentication" in payload["task"]
        assert "src/auth.py" in payload["files_modified"]
        assert "src/app.py" in payload["files_modified"]
        assert any("PyJWT" in d for d in payload["decisions"])
        assert any("ImportError" in e for e in payload["errors_resolved"])
        assert payload["messages_summarized"] > 0
        assert payload["original_tokens"] > 0

    def test_never_cuts_into_tool_result(self):
        # History where the natural cut point lands on a tool_result-only
        # user message; the cut must back up to a clean user message.
        msgs = []
        msgs += _exchange("task one", "ok")
        msgs += _exchange("task two", "ok")
        msgs += _exchange("task three", "ok")
        msgs.append({"role": "user", "content": "run the build"})
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "bash", "input": {}}
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "built"}
                ],
            }
        )
        msgs.append({"role": "assistant", "content": "build succeeded"})

        # keep_last_n=1 puts the natural cut exactly on the tool_result
        # message (index 8); the cut must back up to index 6.
        _, out = compressor.compress_messages(msgs, keep_last_n=1, min_messages=4)
        assert len(out) == 4
        assert out[0]["role"] == "user"
        assert not is_tool_result_only(out[0])
        # tool_use/tool_result pairing preserved inside the kept window
        for i, m in enumerate(out):
            if is_tool_result_only(m):
                prev = out[i - 1]["content"]
                assert any(b.get("type") == "tool_use" for b in prev)

    def test_no_safe_cut_returns_unchanged(self):
        # Every user message after index 0 is tool_result-only: no safe cut
        msgs = [{"role": "user", "content": "start"}]
        for i in range(6):
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "t%d" % i, "name": "bash",
                         "input": {}}
                    ],
                }
            )
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t%d" % i,
                         "content": "ok"}
                    ],
                }
            )
        card, out = compressor.compress_messages(msgs, keep_last_n=2, min_messages=4)
        assert card is None
        assert out == msgs

    def test_string_and_block_content_both_handled(self):
        msgs = []
        for i in range(5):
            msgs.append({"role": "user", "content": "question %d about main.py" % i})
            msgs.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "answer %d" % i}],
                }
            )
        card, out = compressor.compress_messages(msgs, keep_last_n=2, min_messages=6)
        assert card is not None
        payload = json.loads(card[card.index("{") : card.rindex("}") + 1])
        assert "main.py" in payload["files_modified"]
        assert len(out) == 4


class TestBuildMemoryCard:
    def test_task_from_first_user_message(self):
        card = compressor.build_memory_card(
            _exchange("Refactor the database layer", "ok")
        )
        assert card["task"].startswith("Refactor the database")

    def test_files_deduplicated(self):
        msgs = _exchange(
            "fix utils.py", "I edited utils.py, then edited utils.py again"
        )
        card = compressor.build_memory_card(msgs)
        assert card["files_modified"].count("utils.py") == 1

    def test_empty_messages(self):
        card = compressor.build_memory_card([])
        assert card["task"] == ""
        assert card["files_modified"] == []
        assert card["messages_summarized"] == 0
