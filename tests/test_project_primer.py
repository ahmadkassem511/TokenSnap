"""Offline tests for tokensnap.project_primer: scanning a temp project into a
compact Project Card, per-session priming, persistence, and the never-raises
contract. No network, no real ~/.tokensnap.
"""

import json
import subprocess

import pytest

from tokensnap import project_primer as pp


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "LAST_CARD_FILE", tmp_path / "last_card.json")
    pp.reset_cache()
    yield
    pp.reset_cache()


def _make_py_project(root):
    (root / "pyproject.toml").write_text(
        '[project]\n'
        'name = "demo-app"\n'
        'version = "1.0.0"\n'
        'dependencies = ["flask>=2.0", "requests", "sqlalchemy"]\n'
        '\n[project.optional-dependencies]\n'
        'dev = ["pytest"]\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# Demo App\n\n"
        "A tiny web service for demos. It does things.\n",
        encoding="utf-8",
    )
    (root / "app").mkdir()
    (root / "node_modules").mkdir()  # must be ignored
    (root / ".git").mkdir()          # presence only (no real repo)
    (root / "main.py").write_text("print('hi')\n", encoding="utf-8")


class TestGenerateProjectCard:
    def test_scans_python_project(self, tmp_path):
        _make_py_project(tmp_path)
        card = pp.generate_project_card(str(tmp_path))
        assert card["project_name"] == tmp_path.name
        assert card["language"] == "Python"
        assert card["framework"] == "Flask"
        # Only real deps from the dependencies array (not name/version/extras).
        assert "flask" in card["key_dependencies"]
        assert "sqlalchemy" in card["key_dependencies"]
        assert "demo-app" not in card["key_dependencies"]
        assert "1.0.0" not in card["key_dependencies"]

    def test_folder_structure_ignores_noise_dirs(self, tmp_path):
        _make_py_project(tmp_path)
        card = pp.generate_project_card(str(tmp_path))
        assert "app/" in card["folder_structure"]
        assert "main.py" in card["folder_structure"]
        assert "node_modules/" not in card["folder_structure"]
        assert ".git/" not in card["folder_structure"]

    def test_readme_summary_is_first_sentence(self, tmp_path):
        _make_py_project(tmp_path)
        card = pp.generate_project_card(str(tmp_path))
        # Skips the "# Demo App" heading, takes the first real sentence.
        assert card["readme_summary"] == "A tiny web service for demos."

    def test_detects_node_typescript_and_framework(self, tmp_path):
        (tmp_path / "package.json").write_text(
            json.dumps({"dependencies": {"next": "14", "react": "18"},
                        "devDependencies": {"typescript": "5"}}),
            encoding="utf-8",
        )
        (tmp_path / "tsconfig.json").write_text("{}", encoding="utf-8")
        card = pp.generate_project_card(str(tmp_path))
        assert card["language"] == "TypeScript"
        assert card["framework"] == "Next.js"  # next wins over react
        assert "react" in card["key_dependencies"]

    def test_detects_rust_and_go(self, tmp_path):
        (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
        assert pp.generate_project_card(str(tmp_path))["language"] == "Rust"
        (tmp_path / "Cargo.toml").unlink()
        (tmp_path / "go.mod").write_text("module x\n", encoding="utf-8")
        assert pp.generate_project_card(str(tmp_path))["language"] == "Go"

    def test_unknown_directory_returns_empty_card(self):
        card = pp.generate_project_card("/no/such/dir/xyz")
        assert card["project_name"] == ""
        assert card["language"] == ""
        assert card["folder_structure"] == []

    def test_card_is_bounded_in_size(self, tmp_path):
        # Many files + deps must still yield a small card.
        for i in range(200):
            (tmp_path / ("file_%03d.py" % i)).write_text("x", encoding="utf-8")
        card = pp.generate_project_card(str(tmp_path))
        assert len(card["folder_structure"]) <= pp._MAX_ENTRIES
        assert len(pp.format_card(card)) <= pp._CARD_CHAR_CAP + 64

    def test_never_raises_on_weird_input(self, tmp_path):
        # A file where a directory is expected, and a broken package.json.
        (tmp_path / "package.json").write_text("{not valid json", encoding="utf-8")
        card = pp.generate_project_card(str(tmp_path))  # must not raise
        assert card["language"] in ("JavaScript", "TypeScript")


class TestGitInfo:
    def test_reads_real_git_state(self, tmp_path):
        # Build a throwaway repo so the git probe has something real to read.
        def git(*a):
            subprocess.run(["git", *a], cwd=str(tmp_path), check=True,
                           capture_output=True)
        try:
            git("init")
            git("config", "user.email", "t@t.t")
            git("config", "user.name", "t")
            (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
            git("add", "a.txt")
            git("commit", "-m", "initial commit")
            (tmp_path / "b.txt").write_text("new", encoding="utf-8")  # untracked
        except (OSError, subprocess.SubprocessError):
            pytest.skip("git not available")
        card = pp.generate_project_card(str(tmp_path))
        assert card["last_commit_summary"] == "initial commit"
        assert card["git_branch"]  # some branch name (main/master)
        # The untracked file path is captured intact (no leading char lost).
        assert "b.txt" in card["modified_files"]


class TestPrimeForSession:
    def test_primes_once_per_session(self, tmp_path):
        _make_py_project(tmp_path)
        first = pp.prime_for_session("sess-1", str(tmp_path), {})
        second = pp.prime_for_session("sess-1", str(tmp_path), {})
        assert first is not None
        assert first["language"] == "Python"
        assert second is None  # already primed - inject only on the first request

    def test_distinct_sessions_each_prime(self, tmp_path):
        _make_py_project(tmp_path)
        assert pp.prime_for_session("s1", str(tmp_path), {}) is not None
        assert pp.prime_for_session("s2", str(tmp_path), {}) is not None

    def test_unknown_directory_is_not_primed(self):
        assert pp.prime_for_session("s", "unknown", {}) is None
        assert pp.prime_for_session("s", "/no/such/dir", {}) is None

    def test_prime_persists_last_card(self, tmp_path):
        _make_py_project(tmp_path)
        pp.prime_for_session("s", str(tmp_path), {})
        loaded = pp.load_last_card()
        assert loaded is not None
        assert loaded["language"] == "Python"


class TestFormatCard:
    def test_labeled_block_and_json_payload(self, tmp_path):
        _make_py_project(tmp_path)
        text = pp.format_card(pp.generate_project_card(str(tmp_path)))
        assert "PROJECT PRIMER" in text
        assert "demo" in text.lower()
        # Empty fields are dropped from the compact payload.
        assert '"framework":"Flask"' in text.replace(" ", "")

    def test_load_last_card_none_when_absent(self):
        assert pp.load_last_card() is None
