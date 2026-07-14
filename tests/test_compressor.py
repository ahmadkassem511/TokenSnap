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


def _read_tool_convo(tool_name, tool_input, file_content, second_call_name="Bash",
                      second_input=None, second_content=None):
    """A [tool_use, tool_result, tool_use, tool_result] conversation: the
    first call is the one under test, the second is a plain Bash call whose
    result should always still be compressed normally."""
    second_input = second_input or {"command": "npm install"}
    second_content = second_content or (
        "\n".join("npm log line %d" % i for i in range(300))
        + "\nnpm WARN deprecated foo\nadded 200 packages in 5s"
    )
    return [
        {"role": "user", "content": "please do the thing"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": tool_name, "input": tool_input}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": file_content}]},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t2", "name": second_call_name, "input": second_input}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t2", "content": second_content}]},
    ]


class TestReadToolNeverCompressed:
    """A tool_result never carries its own tool name - only tool_use_id -
    so identifying "was this a file read" requires resolving back to the
    tool_use block that produced it (_build_tool_use_index)."""

    def _compress_all(self, msgs):
        idx = compressor._build_tool_use_index(msgs)
        return [compressor._compress_message_selectively(m, idx) for m in msgs]

    def test_read_tool_result_passed_through_verbatim(self):
        big_file = "\n".join("line %d of the file" % i for i in range(500))
        msgs = _read_tool_convo("Read", {"file_path": "config.py"}, big_file)
        out = self._compress_all(msgs)
        assert out[2]["content"][0]["content"] == big_file
        assert "tokensnap" not in out[2]["content"][0]["content"]

    def test_bash_tool_result_still_compressed_as_usual(self):
        # The user's second explicit requirement: non-read tools (e.g. npm
        # install output) must still be compressed exactly like before.
        big_file = "irrelevant"
        msgs = _read_tool_convo("Read", {"file_path": "x.py"}, big_file)
        out = self._compress_all(msgs)
        bash_result = out[4]["content"][0]["content"]
        original_bash = msgs[4]["content"][0]["content"]
        assert len(bash_result) < len(original_bash)
        assert "tokensnap" in bash_result

    def test_case_insensitive_tool_name_matching(self):
        big_file = "\n".join("line %d" % i for i in range(500))
        for name in ("read", "READ", "Read", "ReAd"):
            msgs = _read_tool_convo(name, {"file_path": "x.py"}, big_file)
            out = self._compress_all(msgs)
            assert out[2]["content"][0]["content"] == big_file, name

    def test_view_and_open_tool_names_also_exempt(self):
        big_file = "\n".join("line %d" % i for i in range(500))
        for name in ("View", "Open"):
            msgs = _read_tool_convo(name, {"file_path": "x.py"}, big_file)
            out = self._compress_all(msgs)
            assert out[2]["content"][0]["content"] == big_file, name

    def test_grep_tool_result_never_compressed(self):
        # Many matches across a big codebase - the kind of result that would
        # otherwise exceed the compression threshold and get reduced to
        # "errors and warnings" (a grep result may have neither).
        matches = "\n".join(
            "src/file%d.py:%d: TODO fix this" % (i, i) for i in range(300)
        )
        msgs = _read_tool_convo("Grep", {"pattern": "TODO"}, matches)
        out = self._compress_all(msgs)
        assert out[2]["content"][0]["content"] == matches

    def test_glob_tool_result_never_compressed(self):
        paths = "\n".join("src/module%d/file%d.py" % (i, i) for i in range(300))
        msgs = _read_tool_convo("Glob", {"pattern": "**/*.py"}, paths)
        out = self._compress_all(msgs)
        assert out[2]["content"][0]["content"] == paths

    def test_grep_and_glob_case_insensitive(self):
        content = "\n".join("match %d" % i for i in range(300))
        for name in ("grep", "GREP", "Glob", "GLOB"):
            msgs = _read_tool_convo(name, {}, content)
            out = self._compress_all(msgs)
            assert out[2]["content"][0]["content"] == content, name

    def test_grep_invoked_via_bash_is_not_auto_exempt(self):
        # The dedicated "Grep" tool is exempt; a raw shell `grep` command run
        # through Bash is not - "grep" isn't a pure single-purpose read verb
        # the way cat/type are, and is very often part of a longer pipeline.
        # This deliberately stays conservative (falls back to normal Bash
        # handling, which still compresses only past the size threshold).
        big_output = "\n".join("match %d" % i for i in range(500))
        msgs = _read_tool_convo("Bash", {"command": "grep TODO file.py"}, big_output)
        out = self._compress_all(msgs)
        assert out[2]["content"][0]["content"] != big_output

    def test_pure_shell_read_commands_are_exempt(self):
        big_file = "\n".join("line %d of the file" % i for i in range(500))
        for command in ("cat file.txt", "type file.txt", "Get-Content -Path file.txt",
                         "head -n 100 file.txt", "tail file.txt", "more file.txt"):
            msgs = _read_tool_convo("Bash", {"command": command}, big_file)
            out = self._compress_all(msgs)
            assert out[2]["content"][0]["content"] == big_file, command

    def test_composed_shell_commands_are_not_exempt(self):
        # "cat file.txt && rm file.txt" is not a pure read just because it
        # starts with cat - a chained/piped/redirected command still
        # compresses normally, since it does more than just read.
        big_file = "\n".join("line %d of the file" % i for i in range(500))
        for command in ("cat file.txt && rm file.txt", "cat file.txt | grep foo",
                         "cat a.txt > b.txt", "cat a.txt; echo done"):
            msgs = _read_tool_convo("Bash", {"command": command}, big_file)
            out = self._compress_all(msgs)
            assert out[2]["content"][0]["content"] != big_file, command
            assert "tokensnap" in out[2]["content"][0]["content"], command

    def test_ordinary_bash_command_not_exempt(self):
        big_file = "\n".join("line %d of the file" % i for i in range(500))
        msgs = _read_tool_convo("Bash", {"command": "npm install"}, big_file)
        out = self._compress_all(msgs)
        assert out[2]["content"][0]["content"] != big_file
        assert "tokensnap" in out[2]["content"][0]["content"]

    def test_unresolvable_tool_use_id_falls_back_to_normal_compression(self):
        # A safe default: if the originating call can't be resolved (e.g. an
        # empty index), a large result still compresses rather than silently
        # skipping compression for everything.
        big_file = "\n".join("line %d of the file" % i for i in range(500))
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "unknown", "content": big_file}]}
        out = compressor._compress_message_selectively(msg, {})
        assert out["content"][0]["content"] != big_file

    def test_no_index_argument_falls_back_to_normal_compression(self):
        # Backward compatible: existing callers that never pass an index
        # still get ordinary compression, not a silent skip.
        big_file = "\n".join("line %d of the file" % i for i in range(500))
        msg = {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": big_file}]}
        out = compressor._compress_message_selectively(msg)
        assert out["content"][0]["content"] != big_file

    def test_read_tool_result_as_content_blocks_also_exempt(self):
        # Some tool_result contents are a list of {"type":"text",...} blocks
        # rather than a bare string - the exemption must cover that shape too.
        big_file = "\n".join("line %d of the file" % i for i in range(500))
        msgs = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x.py"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": big_file}]}]},
        ]
        idx = compressor._build_tool_use_index(msgs)
        out = compressor._compress_message_selectively(msgs[1], idx)
        assert out["content"][0]["content"][0]["text"] == big_file


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

    def test_labelled_decision_line_is_captured(self):
        # Regression test: _DECISION_RE's "decision:" alternative previously
        # had a trailing \b that can never match right after a colon (both
        # neighboring chars are non-word), so a bare "Decision: X" line -
        # arguably the single most common way to phrase one - was silently
        # never captured unless it also happened to contain phrasing like
        # "we will". Verify all three labelled prefixes now work standalone.
        for line in ("Decision: use SQLite for storage",
                     "Decided: use SQLite for storage",
                     "Plan: use SQLite for storage"):
            msgs = _exchange("what should we use for storage?", line)
            card = compressor.build_memory_card(msgs)
            assert card["decisions"], "not captured: %r" % line


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
