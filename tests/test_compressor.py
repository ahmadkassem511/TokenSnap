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

    def test_cut_on_tool_result_converts_it_to_text(self):
        # The natural cut lands exactly on a tool_result-only user message;
        # its orphaned tool_results must be converted to plain text so the
        # truncated history stays valid for the API.
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

        # keep_last_n=1 puts the cut exactly on the tool_result (index 8)
        card, out = compressor.compress_messages(msgs, keep_last_n=1, min_messages=4)
        assert card is not None
        assert len(out) == 2
        assert out[0]["role"] == "user"
        assert not is_tool_result_only(out[0])
        assert "built" in out[0]["content"]  # tool output preserved as text

    def test_agentic_tool_loop_history_compresses(self):
        # Claude Code style transcript: one real user prompt, then a long
        # tool_use/tool_result loop. Compression must still engage.
        msgs = [{"role": "user", "content": "start the task on main.py"}]
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
                         "content": "output %d" % i}
                    ],
                }
            )
        card, out = compressor.compress_messages(msgs, keep_last_n=2, min_messages=4)
        assert card is not None
        assert len(out) < len(msgs)
        assert out[0]["role"] == "user"
        assert not is_tool_result_only(out[0])
        # every remaining tool_result still has its tool_use right before it
        for i, m in enumerate(out):
            if is_tool_result_only(m):
                prev = out[i - 1]["content"]
                assert any(b.get("type") == "tool_use" for b in prev)

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


class TestOffMode:
    def test_off_disables_truncation_entirely(self):
        msgs = _long_history()
        card, out = compressor.compress_messages(
            msgs, keep_last_n=1, min_messages=1, llm_cfg={"compressor_type": "off"}
        )
        assert card is None
        assert out == msgs  # full history, untouched

    def test_off_beats_a_tiny_keep_last_n(self):
        # Even a keep_last_n that would normally force a cut is ignored.
        msgs = _long_history() * 3
        card, out = compressor.compress_messages(
            msgs, keep_last_n=1, min_messages=1, llm_cfg={"compressor_type": "off"}
        )
        assert card is None
        assert len(out) == len(msgs)

    def test_regex_and_openrouter_still_compress(self):
        msgs = _long_history()
        for compressor_type in ("regex", "openrouter"):
            card, out = compressor.compress_messages(
                msgs, keep_last_n=3, min_messages=8,
                llm_cfg={"compressor_type": compressor_type},
            )
            assert card is not None
            assert len(out) < len(msgs)


class TestSelectiveCompression:
    def test_assistant_messages_always_pass_through(self):
        dump = "\n".join(["error: line %d" % i for i in range(600)])
        msg = {"role": "assistant", "content": dump}
        assert compressor._compress_message_selectively(msg) is msg

    def test_short_user_text_untouched(self):
        msg = {"role": "user", "content": "please fix the bug in main.py"}
        out = compressor._compress_message_selectively(msg)
        assert out["content"] == msg["content"]

    def test_large_user_terminal_dump_is_compressed(self):
        dump = "\n".join(["npm WARN noisy line %d" % i for i in range(300)] + ["error: build failed"])
        msg = {"role": "user", "content": dump}
        out = compressor._compress_message_selectively(msg)
        assert len(out["content"]) < len(dump)
        assert "error: build failed" in out["content"]
        assert "tokensnap" in out["content"]

    def test_user_prose_preamble_and_question_survive_around_dump(self):
        dump = "\n".join(["error: something broke at line %d" % i for i in range(400)])
        text = "Can you check this error?\n\n" + dump + "\n\nWhat should I do?"
        msg = {"role": "user", "content": text}
        out = compressor._compress_message_selectively(msg)
        assert "Can you check this error?" in out["content"]
        assert "What should I do?" in out["content"]
        assert len(out["content"]) < len(text)

    def test_tool_result_compressed_to_signal_lines(self):
        dump = "\n".join(
            ["npm WARN deprecated pkg@1.0.0"] * 5
            + ["line %d" % i for i in range(150)]
            + ["error: build failed", "Build finished with 1 error"]
        )
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": dump}],
        }
        out = compressor._compress_message_selectively(msg)
        text = out["content"][0]["content"]
        assert "error: build failed" in text
        assert "Build finished with 1 error" in text
        assert len(text) < len(dump)
        assert "1 error" in text or "1 warning" in text  # omission summary present

    def test_small_tool_result_left_alone(self):
        msg = {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "2 tests passed"}],
        }
        out = compressor._compress_message_selectively(msg)
        assert out["content"][0]["content"] == "2 tests passed"

    def test_pathological_dense_errors_still_capped(self):
        # Every line matches the error pattern (distinct line numbers, so
        # dedup can't collapse them) - the cap must still kick in.
        dump = "\n".join("ERROR: broke at line %d" % i for i in range(200))
        text = compressor._extract_signal_lines(dump)
        # first+last half of the cap, plus the omission summary line
        assert text.count("\n") + 1 <= compressor._MAX_SIGNAL_LINES + 1
        assert "200 errors" in text

    def test_tool_use_blocks_never_touched(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "x"}}],
        }
        out = compressor._compress_message_selectively(msg)
        assert out == msg


class TestBuildCompressedContext:
    def _agentic_history(self):
        msgs = [{"role": "user", "content": "Fix the failing tests in src/app.py"}]
        for i in range(8):
            msgs.append({
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t%d" % i, "name": "bash", "input": {}}],
            })
            dump = "\n".join(["retrying..."] * 3 + ["error: test %d failed" % i])
            msgs.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t%d" % i, "content": dump}],
            })
        return msgs

    def test_compresses_and_cleans_tool_results(self):
        msgs = self._agentic_history()
        card, out = compressor.build_compressed_context(
            msgs, keep_messages=2, cfg={"compressor_type": "regex"}, min_messages=4
        )
        assert card is not None
        assert len(out) < len(msgs)
        # Every tool_result surviving in the tail is still paired with its tool_use
        for i, m in enumerate(out):
            if is_tool_result_only(m):
                prev = out[i - 1]["content"]
                assert any(b.get("type") == "tool_use" for b in prev)

    def test_off_keeps_everything_but_still_cleans_messages(self):
        msgs = self._agentic_history()
        card, out = compressor.build_compressed_context(
            msgs, keep_messages=2, cfg={"compressor_type": "off"}, min_messages=4
        )
        assert card is None
        assert len(out) == len(msgs)

    def test_default_cfg_is_safe(self):
        # cfg=None must not crash - callers may omit it.
        msgs = self._agentic_history()
        card, out = compressor.build_compressed_context(msgs, keep_messages=2, min_messages=4)
        assert isinstance(out, list)


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


class TestMessageWeight:
    def test_assistant_outweighs_tool_result(self):
        assistant = {"role": "assistant", "content": "here is my reasoning"}
        tool = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t", "content": "ok"}]}
        assert compressor.message_weight(assistant) > compressor.message_weight(tool)

    def test_user_between_assistant_and_tool(self):
        a = compressor.message_weight({"role": "assistant", "content": "x"})
        u = compressor.message_weight({"role": "user", "content": "x"})
        t = compressor.message_weight(
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t", "content": "x"}]}
        )
        assert a > u > t

    def test_decision_and_important_raise_weight(self):
        plain = {"role": "user", "content": "just some text"}
        decision = {"role": "user", "content": "we will use Postgres"}
        important = {"role": "user", "content": "IMPORTANT: never delete prod"}
        assert compressor.message_weight(decision) > compressor.message_weight(plain)
        assert compressor.message_weight(important) > compressor.message_weight(plain)

    def test_recency_increases_weight(self):
        m = {"role": "user", "content": "same text"}
        old = compressor.message_weight(m, index=0, total=10)
        new = compressor.message_weight(m, index=9, total=10)
        assert new > old

    def test_marked_important_holds_a_floor_despite_age(self):
        note = {"role": "user", "content": "NOTE: keep this credential rotation logic"}
        # Even as the oldest message, an explicit marker stays high.
        assert compressor.message_weight(note, index=0, total=100) >= 4.0

    def test_log_dump_lowers_weight(self):
        dump = "\n".join("INFO line %d" % i for i in range(400)) + "\nerror: boom"
        noisy = {"role": "user", "content": dump}
        prose = {"role": "user", "content": "a short normal message"}
        assert compressor.message_weight(noisy) < compressor.message_weight(prose)

    def test_high_weight_decisions_survive_the_cap(self):
        # More than _MAX_ITEMS decisions: low-weight (tool-result) ones should be
        # dropped in favour of high-weight (assistant) ones.
        msgs = []
        for i in range(compressor._MAX_ITEMS):
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t%d" % i,
                 "content": "we will use lowprio-%d" % i}]})
        for i in range(5):
            msgs.append({"role": "assistant", "content": "Decision: we will use HIGHPRIO-%d" % i})
        card = compressor.build_memory_card(msgs)
        assert len(card["decisions"]) == compressor._MAX_ITEMS
        highprio = [d for d in card["decisions"] if "HIGHPRIO" in d]
        assert len(highprio) == 5  # all high-weight decisions kept
