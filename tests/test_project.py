"""Offline tests for tokensnap.project: the mutable 'current project' pointer
the proxy reads per request. Every test redirects PROJECT_FILE to a throwaway
path so nothing touches the real ~/.tokensnap/current_project.
"""

import pytest

from tokensnap import project


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(project, "PROJECT_FILE", tmp_path / "current_project")
    monkeypatch.delenv("TOKENSNAP_PROJECT", raising=False)
    yield


def test_defaults_to_unknown_when_nothing_set():
    assert project.get_current_project() == "unknown"


def test_set_then_get_round_trips():
    project.set_current_project("C:/work/my-proj")
    assert project.get_current_project() == "C:/work/my-proj"


def test_set_overwrites_previous_value():
    project.set_current_project("proj-a")
    project.set_current_project("proj-b")
    assert project.get_current_project() == "proj-b"  # most recent launch wins


def test_file_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("TOKENSNAP_PROJECT", "from-env")
    project.set_current_project("from-file")
    assert project.get_current_project() == "from-file"


def test_env_used_as_fallback_when_no_file(monkeypatch):
    monkeypatch.setenv("TOKENSNAP_PROJECT", "from-env")
    assert project.get_current_project() == "from-env"


def test_blank_value_falls_back_to_env_then_unknown(monkeypatch):
    project.set_current_project("   ")  # whitespace collapses to empty
    assert project.get_current_project() == "unknown"
    monkeypatch.setenv("TOKENSNAP_PROJECT", "env-proj")
    assert project.get_current_project() == "env-proj"


def test_set_never_raises_on_unwritable_path(monkeypatch, tmp_path):
    # Parent is a regular file, so mkdir() raises OSError, which must be swallowed.
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(project, "PROJECT_FILE", blocker / "nested" / "current_project")
    project.set_current_project("whatever")  # must not raise
    assert project.get_current_project() == "unknown"
