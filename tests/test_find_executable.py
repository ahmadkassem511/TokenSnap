"""Offline tests for tokensnap.utils executable resolution: find_executable
(PATH + npm global bin) and resolve_claude_command (with the npx fallback).

Filesystem/subprocess access is mocked so these are deterministic on any OS.
"""

import os

from tokensnap import utils


class TestNpmGlobalBin:
    def test_none_when_npm_absent(self, monkeypatch):
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: None)
        assert utils._npm_global_bin() is None

    def test_none_on_empty_output(self, monkeypatch):
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: "/usr/bin/npm")

        class R:
            stdout = "  \n"

        monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: R())
        assert utils._npm_global_bin() is None

    def test_none_when_npm_call_fails(self, monkeypatch):
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: "/usr/bin/npm")

        def boom(*a, **k):
            raise OSError("nope")

        monkeypatch.setattr(utils.subprocess, "run", boom)
        assert utils._npm_global_bin() is None

    def test_unix_prefix_bin(self, monkeypatch):
        monkeypatch.setattr(utils.os, "name", "posix")
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: "/usr/bin/npm")

        class R:
            stdout = "/usr/local/lib/node_modules\n"

        monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: R())
        result = utils._npm_global_bin()
        assert os.path.basename(result) == "bin"
        assert "local" in result

    def test_windows_returns_dirname(self, monkeypatch):
        monkeypatch.setattr(utils.os, "name", "nt")
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: "npm")
        root = r"C:\Users\X\AppData\Roaming\npm\node_modules"

        class R:
            stdout = root

        monkeypatch.setattr(utils.subprocess, "run", lambda *a, **k: R())
        # The Windows global bin is the parent of node_modules (where
        # claude.cmd lives), not a separate `bin` dir.
        assert utils._npm_global_bin() == os.path.dirname(root)


class TestFindExecutable:
    def test_found_on_path(self, monkeypatch):
        monkeypatch.setattr(
            utils.shutil, "which",
            lambda name, *a, **k: "/usr/bin/tool" if (name == "tool" and not k.get("path")) else None,
        )
        assert utils.find_executable("tool") == os.path.abspath("/usr/bin/tool")

    def test_found_in_npm_global_bin_when_off_path(self, monkeypatch, tmp_path):
        npmdir = tmp_path / "npm"
        npmdir.mkdir()
        monkeypatch.setattr(utils, "_npm_global_bin", lambda: str(npmdir))

        def fake_which(name, *a, **k):
            # Not on PATH; only resolvable when searching the npm dir directly.
            if name == "claude" and k.get("path") == str(npmdir):
                return str(npmdir / "claude.cmd")
            return None

        monkeypatch.setattr(utils.shutil, "which", fake_which)
        got = utils.find_executable("claude")
        assert got == os.path.abspath(str(npmdir / "claude.cmd"))

    def test_none_when_nowhere(self, monkeypatch):
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: None)
        monkeypatch.setattr(utils, "_npm_global_bin", lambda: None)
        assert utils.find_executable("ghostly") is None

    def test_skips_nonexistent_candidate_dirs(self, monkeypatch):
        # _npm_global_bin points at a dir that doesn't exist -> must be skipped
        # (not crash), and PATH miss -> None overall.
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: None)
        monkeypatch.setattr(utils, "_npm_global_bin", lambda: "/no/such/dir/xyz")
        assert utils.find_executable("claude") is None


class TestResolveClaudeCommand:
    def test_non_claude_command_unchanged(self, monkeypatch):
        # Must not even try to resolve; generic commands pass straight through.
        monkeypatch.setattr(utils, "find_executable", lambda n: None)
        assert utils.resolve_claude_command(["ls", "-la"]) == ["ls", "-la"]

    def test_empty_command_unchanged(self):
        assert utils.resolve_claude_command([]) == []

    def test_substitutes_resolved_path(self, monkeypatch):
        monkeypatch.setattr(utils, "find_executable", lambda n: "/abs/bin/claude")
        assert utils.resolve_claude_command(["claude"]) == ["/abs/bin/claude"]

    def test_preserves_extra_args(self, monkeypatch):
        monkeypatch.setattr(utils, "find_executable", lambda n: "/abs/bin/claude")
        assert utils.resolve_claude_command(["claude", "--dangerously", "x"]) == \
            ["/abs/bin/claude", "--dangerously", "x"]

    def test_recognizes_windows_basenames(self, monkeypatch):
        monkeypatch.setattr(utils, "find_executable", lambda n: "/abs/claude")
        assert utils.resolve_claude_command([r"C:\npm\claude.cmd"]) == ["/abs/claude"]
        assert utils.resolve_claude_command(["CLAUDE.EXE"]) == ["/abs/claude"]

    def test_npx_fallback_when_not_found(self, monkeypatch):
        monkeypatch.setattr(utils, "find_executable", lambda n: None)
        monkeypatch.setattr(
            utils.shutil, "which", lambda n, *a, **k: "/usr/bin/npx" if n == "npx" else None)
        assert utils.resolve_claude_command(["claude"]) == ["npx", "claude"]

    def test_npx_fallback_preserves_args(self, monkeypatch):
        monkeypatch.setattr(utils, "find_executable", lambda n: None)
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: "/usr/bin/npx")
        assert utils.resolve_claude_command(["claude", "--foo"]) == ["npx", "claude", "--foo"]

    def test_none_when_no_claude_and_no_npx(self, monkeypatch):
        monkeypatch.setattr(utils, "find_executable", lambda n: None)
        monkeypatch.setattr(utils.shutil, "which", lambda n, *a, **k: None)
        assert utils.resolve_claude_command(["claude"]) is None
