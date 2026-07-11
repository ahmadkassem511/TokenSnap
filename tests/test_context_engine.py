"""Offline tests for tokensnap.context_engine: session ids, event
classification, ingestion, and the Context Tree reconstruction. Every test
redirects the Context Store DB to a throwaway file.
"""

import json

import pytest

from tokensnap import context_engine as ce
from tokensnap import context_store

KEEP_MESSAGES_KEPT = ce.KEEP_EXCHANGES * 2  # exchanges -> messages


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(context_store, "DB_FILE", tmp_path / "context_store.db")
    yield


def _history(n_exchanges, system="You are a helpful assistant."):
    msgs = []
    for i in range(n_exchanges):
        msgs.append({"role": "user", "content": "step %d: please edit src/app.py" % i})
        msgs.append({"role": "assistant", "content": "I'll modify src/app.py for step %d" % i})
    return system, msgs


class TestDeriveSessionId:
    def test_stable_as_conversation_grows(self):
        system, short = _history(2)
        _, longer = _history(6)
        # Same prefix (system + first user message) -> same id.
        assert ce.derive_session_id(system, short) == ce.derive_session_id(system, longer)

    def test_differs_between_conversations(self):
        s1, m1 = _history(3)
        m2 = [{"role": "user", "content": "totally different first task"}]
        assert ce.derive_session_id(s1, m1) != ce.derive_session_id(s1, m2)

    def test_handles_missing_system_and_users(self):
        # No system, no user message -> still returns a stable string.
        sid = ce.derive_session_id(None, [{"role": "assistant", "content": "hi"}])
        assert isinstance(sid, str) and sid


class TestEventClassification:
    def test_error_wins(self):
        assert ce.event_type_for("Traceback (most recent call last): KeyError") == "error"
        assert ce.event_type_for("the build failed with an exception") == "error"

    def test_file_modification(self):
        assert ce.event_type_for("I'll modify auth.py to add a token helper") == "file_modification"
        assert ce.event_type_for("Let's change the config") == "file_modification"

    def test_decision(self):
        assert ce.event_type_for("Decision: we will use SQLite here") == "decision"

    def test_clarification(self):
        assert ce.event_type_for("Just to clarify, do you mean the proxy port?") == "clarification"

    def test_other(self):
        assert ce.event_type_for("sounds good, thanks") == "other"

    def test_plain_user_request_falls_back_to_request_not_other(self):
        # A genuine ask that matches none of the categories above must never
        # be classified 'other' (dropped from the tree) - it falls back to
        # 'request' instead, so the model never loses track of what it was
        # asked to do (see the Adaptive Transparency Mode bug report).
        assert ce.event_type_for("read this project and run it", is_user_request=True) == "request"
        assert ce.event_type_for("run video_pipeline.py for me", is_user_request=True) == "request"

    def test_more_specific_category_still_wins_over_request(self):
        # is_user_request is only the *fallback*; a real decision/error/etc
        # phrasing still takes priority even from a genuine user message.
        assert ce.event_type_for("Decision: use SQLite", is_user_request=True) == "decision"

    def test_non_request_messages_unaffected(self):
        # Assistant chatter / tool output must still fall to 'other', not
        # 'request' - only genuine user asks get the fallback protection.
        assert ce.event_type_for("sounds good, thanks", is_user_request=False) == "other"


class TestOneLineSummary:
    def test_first_nonempty_line(self):
        assert ce.one_line_summary("\n\n  hello world  \nsecond") == "hello world"

    def test_clips_long_line(self):
        s = ce.one_line_summary("x" * 500, maxlen=10)
        assert len(s) == 10 and s.endswith("…")

    def test_empty(self):
        assert ce.one_line_summary("   \n  ") == ""


class TestIngest:
    def test_stores_nonempty_and_returns_count(self):
        system, msgs = _history(3)
        sid = ce.derive_session_id(system, msgs)
        n = ce.ingest(sid, msgs)
        assert n == 6
        assert context_store.event_count(sid) == 6

    def test_skips_empty_text_messages(self):
        sid = "sess"
        msgs = [
            {"role": "user", "content": "real text"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "x", "input": {}}]},
        ]
        assert ce.ingest(sid, msgs) == 1
        assert context_store.event_count(sid) == 1

    def test_uses_positional_index_so_resends_upsert(self):
        system, msgs = _history(3)
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)
        ce.ingest(sid, msgs)  # same conversation re-sent
        assert context_store.event_count(sid) == 6  # not duplicated

    def test_error_message_surfaces_in_tree(self):
        sid = "sess"
        msgs = [
            {"role": "user", "content": "run the tests"},
            {"role": "assistant", "content": "Traceback: KeyError 'exp' — the test failed"},
        ]
        ce.ingest(sid, msgs)
        tree = context_store.get_recent_tree(sid)
        assert any(e["type"] == "error" for e in tree)

    def test_plain_user_ask_surfaces_as_request_not_dropped(self):
        sid = "sess"
        msgs = [{"role": "user", "content": "read this project and run it"}]
        ce.ingest(sid, msgs)
        tree = context_store.get_recent_tree(sid)
        assert tree == [{"id": "1", "summary": "read this project and run it", "type": "request"}]

    def test_tool_result_only_user_message_stays_other_not_request(self):
        # A tool_result carries role="user" per the API, but it's mechanical
        # output, not a genuine ask - it must not get the request fallback,
        # or the tree would fill with noise instead of real instructions.
        sid = "sess"
        msgs = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "some plain tool output"}
        ]}]
        ce.ingest(sid, msgs)
        assert context_store.get_recent_tree(sid) == []  # correctly excluded as 'other'
        ev = context_store.get_first_event(sid)
        assert ev["type"] == "other"


class TestContextTreeBlock:
    def test_contains_json_and_instruction(self):
        tree = [{"id": "3", "summary": "did a thing", "type": "decision"}]
        block = ce.build_context_tree_block(tree, n_omitted=5)
        assert "CONTEXT TREE" in block
        assert "fetch_context" in block
        assert "5 messages" in block
        assert json.dumps(tree, ensure_ascii=False, separators=(",", ":")) in block

    def test_singular_message_wording(self):
        block = ce.build_context_tree_block([], n_omitted=1)
        assert "(1 message)" in block
        assert "messages" not in block.split("]", 1)[0]  # no plural before the JSON


class TestMergeTool:
    def test_adds_to_none(self):
        tools = ce.merge_fetch_context_tool(None)
        assert [t["name"] for t in tools] == ["fetch_context"]

    def test_appends_to_existing(self):
        existing = [{"name": "Bash"}, {"name": "Read"}]
        tools = ce.merge_fetch_context_tool(existing)
        assert [t["name"] for t in tools] == ["Bash", "Read", "fetch_context"]

    def test_does_not_duplicate(self):
        tools = ce.merge_fetch_context_tool([ce.FETCH_CONTEXT_TOOL])
        assert sum(1 for t in tools if t["name"] == "fetch_context") == 1


class TestReconstruct:
    def test_short_history_unchanged(self):
        system, msgs = _history(2)  # 4 messages <= min 8
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)
        new_msgs, new_sys, reconstructed, n_omitted = ce.reconstruct(
            msgs, system, sid, min_messages=8
        )
        assert reconstructed is False
        assert n_omitted == 0
        assert new_msgs == msgs
        assert new_sys == system

    def test_long_history_keeps_tail_and_injects_tree(self):
        system, msgs = _history(10)  # 20 messages
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)
        new_msgs, new_sys, reconstructed, n_omitted = ce.reconstruct(
            msgs, system, sid, tree_size=20, min_messages=8
        )
        assert reconstructed is True
        # Last 2 exchanges kept verbatim -> 4 messages.
        assert len(new_msgs) == KEEP_MESSAGES_KEPT
        assert n_omitted == len(msgs) - len(new_msgs)
        # The tail starts with a user message (API alternation requirement).
        assert new_msgs[0]["role"] == "user"
        # System now carries the Context Tree block.
        parts = new_sys if isinstance(new_sys, str) else "\n".join(
            b["text"] for b in new_sys if isinstance(b, dict) and "text" in b
        )
        assert "CONTEXT TREE" in parts

    def test_tree_omits_the_kept_tail_events(self):
        system, msgs = _history(10)
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)
        _, new_sys, _, _ = ce.reconstruct(msgs, system, sid, min_messages=8)
        # The context tree is present and references stored event ids.
        assert "fetch_context" in (new_sys if isinstance(new_sys, str) else str(new_sys))

    def test_original_task_never_lost_even_when_unclassifiable(self):
        # Regression test: a plainly-phrased first request ("run this tool")
        # doesn't match any of the decision/error/file-mod/clarification
        # regexes. A genuine user ask now falls back to the 'request' event
        # type (not 'other'), so get_recent_tree already keeps it - and
        # get_first_event backstops message 0 specifically even if that
        # somehow failed. Before either fix, this would classify 'other' and
        # drop with no trace once it fell into the omitted head - unlike the
        # classic Memory Card, which always captures `task` regardless of
        # phrasing. The model would then have no way to recover what it was
        # even asked to do (the exact bug reported: "the only thing I can
        # recover is system boilerplate, not your request").
        system = "sys"
        msgs = [{"role": "user", "content": "run video_pipeline.py for me"}]
        for i in range(15):
            msgs.append({"role": "assistant", "content": "checking things, step %d" % i})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t%d" % i, "content": "ok output %d" % i}
            ]})
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)

        # The task is classified 'request' (a genuine user ask), not 'other' -
        # nothing in this history matches decision/error/file-mod/clarify, but
        # get_recent_tree already keeps it via that classification alone.
        tree = context_store.get_recent_tree(sid, 20)
        assert tree == [{"id": "1", "summary": "run video_pipeline.py for me", "type": "request"}]

        _, new_sys, reconstructed, n_omitted = ce.reconstruct(
            msgs, system, sid, tree_size=20, min_messages=8, keep_exchanges=5
        )
        assert reconstructed is True
        assert n_omitted > 0  # the task message did fall into the omitted head
        text = new_sys if isinstance(new_sys, str) else str(new_sys)
        assert "run video_pipeline.py for me" in text
        assert '"type":"request"' in text.replace(" ", "")

    def test_second_instruction_not_just_the_first_is_preserved(self):
        # The exact shape from the real bug report: the OPERATIVE instruction
        # is the *second* user message (the first was a vague ask that got a
        # clarifying question), buried among many tool round-trips after it.
        # get_first_event alone (the earlier, narrower fix) only protects
        # message 0 - this proves every genuine user ask is protected, not
        # just the first.
        system = "sys"
        msgs = [
            {"role": "user", "content": "run this tool on another terminal"},
            {"role": "assistant", "content": "I don't have context on which tool - could you clarify?"},
            {"role": "user", "content": "read this project and know how it works and run it"},
        ]
        for i in range(15):
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": "t%d" % i, "name": "Bash", "input": {}}]})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t%d" % i, "content": "ok %d" % i}]})
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)

        _, new_sys, reconstructed, n_omitted = ce.reconstruct(
            msgs, system, sid, tree_size=20, min_messages=8, keep_exchanges=5
        )
        assert reconstructed is True
        text = new_sys if isinstance(new_sys, str) else str(new_sys)
        assert "read this project and know how it works and run it" in text

    def test_first_event_backstop_when_tree_size_pushes_task_out(self):
        # get_first_event's backstop (relabeled "task") only matters in the
        # rarer case where the tree's recency LIMIT itself pushes message 0
        # out - e.g. many later decisions crowd a small tree_size. Construct
        # exactly that: more 'important' events after message 0 than tree_size.
        system = "sys"
        msgs = [{"role": "user", "content": "the original task, phrased plainly"}]
        for i in range(10):
            msgs.append({"role": "assistant", "content": "Decision: use approach %d" % i})
            msgs.append({"role": "user", "content": "ok %d" % i})
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)

        # With a tiny tree_size, get_recent_tree alone drops message 0 (10
        # later decisions already fill the limit).
        assert not any(
            e["summary"] == "the original task, phrased plainly"
            for e in context_store.get_recent_tree(sid, 3)
        )

        _, new_sys, reconstructed, n_omitted = ce.reconstruct(
            msgs, system, sid, tree_size=3, min_messages=8, keep_exchanges=1
        )
        text = new_sys if isinstance(new_sys, str) else str(new_sys)
        assert "the original task, phrased plainly" in text
        assert '"type":"task"' in text.replace(" ", "")

    def test_task_not_duplicated_when_already_classified(self):
        # If the opening message happens to also match a real event type
        # (e.g. it reads as a decision), get_first_event's entry must not be
        # duplicated alongside get_recent_tree's own entry for the same id.
        system = "sys"
        msgs = [{"role": "user", "content": "Decision: we will use FFmpeg for video_pipeline.py"}]
        for i in range(15):
            msgs.append({"role": "assistant", "content": "step %d" % i})
            msgs.append({"role": "user", "content": "ok %d" % i})
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)
        _, new_sys, _, _ = ce.reconstruct(
            msgs, system, sid, tree_size=20, min_messages=8, keep_exchanges=5
        )
        text = new_sys if isinstance(new_sys, str) else str(new_sys)
        assert text.count("FFmpeg") == 1  # present once, not duplicated

    def test_orphaned_tool_result_converted(self):
        # Craft a history whose cut lands on a tool_result-only user message.
        system = "sys"
        msgs = [{"role": "user", "content": "start the task"}]
        for i in range(9):
            msgs.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": "t%d" % i, "name": "Bash", "input": {}}
            ]})
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t%d" % i, "content": "ok %d" % i}
            ]})
        sid = ce.derive_session_id(system, msgs)
        ce.ingest(sid, msgs)
        new_msgs, _, reconstructed, _ = ce.reconstruct(
            msgs, system, sid, min_messages=8, selective=False
        )
        assert reconstructed is True
        first = new_msgs[0]
        assert first["role"] == "user"
        # A converted orphan is a plain string, never a bare tool_result block.
        if isinstance(first["content"], list):
            assert not all(b.get("type") == "tool_result" for b in first["content"])
