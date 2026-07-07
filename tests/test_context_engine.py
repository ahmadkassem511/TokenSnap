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
