"""Offline tests for tokensnap.context_store: the conversation mirror.

Every test redirects the DB to a throwaway file so nothing touches the real
~/.tokensnap/context_store.db. The module claims its writes "never raise" - a
few tests deliberately point it at a broken path to prove that contract holds.
"""

import pytest

from tokensnap import context_store


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Point context_store at a fresh throwaway DB for every test."""
    monkeypatch.setattr(context_store, "DB_FILE", tmp_path / "context_store.db")
    yield


def _store(index, role="user", content="hello", summary="", event_type="other",
           session="s1"):
    return context_store.store_message(
        session, index, role, content, summary, event_type
    )


class TestInitAndEmpty:
    def test_init_db_creates_file(self):
        context_store.init_db()
        assert context_store.DB_FILE.exists()

    def test_empty_tree_is_empty_list(self):
        assert context_store.get_recent_tree("nobody") == []

    def test_empty_history_is_empty_list(self):
        assert context_store.get_full_history("nobody") == []

    def test_missing_event_is_none(self):
        assert context_store.get_event_by_id(999) is None

    def test_store_autocreates_db(self):
        # No explicit init_db() - the first store_message must self-heal schema.
        assert not context_store.DB_FILE.exists()
        _store(0)
        assert context_store.DB_FILE.exists()


class TestStoreAndFetch:
    def test_roundtrip_full_content(self):
        big = "line one\n" * 500
        eid = _store(0, role="assistant", content=big, summary="did a thing",
                     event_type="decision")
        assert isinstance(eid, int)
        ev = context_store.get_event_by_id(eid)
        assert ev is not None
        assert ev["content"] == big
        assert ev["role"] == "assistant"
        assert ev["summary"] == "did a thing"
        assert ev["event_type"] == "decision"
        assert ev["message_index"] == 0
        assert ev["session_id"] == "s1"

    def test_get_event_by_id_accepts_numeric_string(self):
        eid = _store(0, content="findme")
        ev = context_store.get_event_by_id(str(eid))
        assert ev is not None and ev["content"] == "findme"

    def test_get_event_by_id_non_numeric_is_none(self):
        _store(0)
        assert context_store.get_event_by_id("not-a-number") is None
        assert context_store.get_event_by_id(None) is None

    def test_unknown_event_type_coerced_to_other(self):
        eid = _store(0, event_type="banana")
        assert context_store.get_event_by_id(eid)["event_type"] == "other"


class TestIdempotency:
    def test_resend_updates_in_place_same_id(self):
        first = _store(0, content="v1", summary="first", event_type="error")
        second = _store(0, content="v2", summary="second", event_type="decision")
        # Same natural key -> same row id, updated in place.
        assert first == second
        assert context_store.event_count("s1") == 1
        ev = context_store.get_event_by_id(first)
        assert ev["content"] == "v2"
        assert ev["summary"] == "second"
        assert ev["event_type"] == "decision"

    def test_resending_whole_prefix_does_not_duplicate(self):
        # Simulate the stateless API: the same 3 messages arrive across 2
        # requests, then a 4th is appended.
        for _ in range(2):
            _store(0, content="m0", event_type="decision")
            _store(1, content="m1", event_type="other")
            _store(2, content="m2", event_type="error")
        _store(3, content="m3", event_type="error")
        assert context_store.event_count("s1") == 4


class TestRecentTree:
    def test_excludes_other(self):
        _store(0, summary="plain", event_type="other")
        _store(1, summary="oops", event_type="error")
        tree = context_store.get_recent_tree("s1")
        assert [e["type"] for e in tree] == ["error"]
        assert tree[0]["summary"] == "oops"

    def test_only_id_summary_type_keys(self):
        _store(0, summary="a decision", event_type="decision")
        entry = context_store.get_recent_tree("s1")[0]
        assert set(entry.keys()) == {"id", "summary", "type"}
        assert isinstance(entry["id"], str)  # string for the model to quote back

    def test_chronological_order_oldest_first(self):
        _store(0, summary="one", event_type="decision")
        _store(1, summary="two", event_type="error")
        _store(2, summary="three", event_type="file_modification")
        summaries = [e["summary"] for e in context_store.get_recent_tree("s1")]
        assert summaries == ["one", "two", "three"]

    def test_limit_keeps_most_recent(self):
        for i in range(5):
            _store(i, summary="s%d" % i, event_type="error")
        tree = context_store.get_recent_tree("s1", limit=2)
        # Two most-recent important events, still oldest-first.
        assert [e["summary"] for e in tree] == ["s3", "s4"]

    def test_tree_id_resolves_back_to_full_event(self):
        _store(0, content="the full text", summary="short", event_type="decision")
        entry = context_store.get_recent_tree("s1")[0]
        ev = context_store.get_event_by_id(entry["id"])
        assert ev["content"] == "the full text"


class TestFullHistory:
    def test_includes_other_and_orders(self):
        _store(2, content="c", event_type="other")
        _store(0, content="a", event_type="decision")
        _store(1, content="b", event_type="other")
        hist = context_store.get_full_history("s1")
        assert [e["content"] for e in hist] == ["a", "b", "c"]

    def test_limit_returns_most_recent_oldest_first(self):
        for i in range(5):
            _store(i, content="m%d" % i)
        hist = context_store.get_full_history("s1", limit=2)
        assert [e["content"] for e in hist] == ["m3", "m4"]


class TestSessionIsolation:
    def test_sessions_do_not_leak(self):
        _store(0, content="a", event_type="error", session="s1")
        _store(0, content="b", event_type="error", session="s2")
        assert context_store.event_count("s1") == 1
        assert context_store.event_count("s2") == 1
        assert context_store.event_count() == 2
        assert context_store.get_full_history("s1")[0]["content"] == "a"
        assert context_store.get_recent_tree("s2")[0]["id"] != \
            context_store.get_recent_tree("s1")[0]["id"]


class TestNeverRaises:
    def _break_db(self, monkeypatch, tmp_path):
        # Parent is a regular file, so mkdir() raises OSError, not sqlite3.Error.
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        monkeypatch.setattr(
            context_store, "DB_FILE", blocker / "nested" / "context_store.db"
        )

    def test_store_message_swallows_bad_db_path(self, monkeypatch, tmp_path):
        self._break_db(monkeypatch, tmp_path)
        assert _store(0) is None  # must not raise

    def test_readers_return_empty_on_bad_db(self, monkeypatch, tmp_path):
        self._break_db(monkeypatch, tmp_path)
        assert context_store.get_recent_tree("s1") == []
        assert context_store.get_full_history("s1") == []
        assert context_store.get_event_by_id(1) is None
        assert context_store.event_count() == 0
