"""Offline tests for tokensnap.cleaner."""

from tokensnap import cleaner


class TestStripAnsi:
    def test_removes_color_codes(self):
        text = "\x1b[31mred error\x1b[0m and \x1b[1;32mbold green\x1b[0m"
        assert cleaner.strip_ansi(text) == "red error and bold green"

    def test_removes_cursor_movement(self):
        text = "\x1b[2K\x1b[1Gline after clear"
        assert cleaner.strip_ansi(text) == "line after clear"

    def test_removes_osc_title_sequence(self):
        text = "\x1b]0;window title\x07real content"
        assert cleaner.strip_ansi(text) == "real content"

    def test_plain_text_untouched(self):
        text = "def foo():\n    return [1, 2, 3]  # 100% plain"
        assert cleaner.strip_ansi(text) == text


class TestStripProgressBars:
    def test_drops_tqdm_style_bar(self):
        text = "Downloading model\n 45%|████████░░░░░░░░| 45/100 [00:12<00:15]\ndone"
        result = cleaner.strip_progress_bars(text)
        assert "████" not in result
        assert "Downloading model" in result
        assert "done" in result

    def test_drops_pip_style_bar(self):
        text = "Collecting requests\n   ---------> 30% eta 0:00:05\nInstalled ok"
        result = cleaner.strip_progress_bars(text)
        assert "--->" not in result
        assert "Installed ok" in result

    def test_collapses_carriage_return_frames(self):
        # Redrawn line: only the final frame should survive
        text = "step 1\rstep 2\rstep 3 final\nnext line"
        result = cleaner.strip_progress_bars(text)
        assert "step 3 final" in result
        assert "step 1" not in result
        assert "next line" in result

    def test_keeps_normal_percentages(self):
        text = "test coverage is 85% overall"
        assert cleaner.strip_progress_bars(text) == text

    def test_crlf_lines_survive(self):
        # Windows CRLF endings must not be treated as redraw frames
        text = "line one\r\nline two\r\nline three\r\n"
        result = cleaner.strip_progress_bars(text)
        assert result == "line one\nline two\nline three\n"

    def test_crlf_redraw_combination(self):
        # Redraw frames inside a CRLF-terminated line: keep the last frame
        text = "step 1\rstep 2 final\r\nnext line\r\n"
        result = cleaner.strip_progress_bars(text)
        assert result == "step 2 final\nnext line\n"


class TestDedupeConsecutiveLines:
    def test_collapses_long_run(self):
        text = "\n".join(["WARNING: retrying"] * 10 + ["done"])
        result = cleaner.dedupe_consecutive_lines(text)
        lines = result.split("\n")
        assert lines[0] == "WARNING: retrying"
        assert "repeated 9 more times" in lines[1]
        assert lines[2] == "done"

    def test_short_runs_untouched(self):
        text = "a\na\nb"
        assert cleaner.dedupe_consecutive_lines(text) == text

    def test_blank_runs_untouched(self):
        text = "a\n\n\n\nb"
        assert cleaner.dedupe_consecutive_lines(text) == text

    def test_distinct_lines_untouched(self):
        text = "one\ntwo\nthree"
        assert cleaner.dedupe_consecutive_lines(text) == text


class TestCleanText:
    def test_full_pipeline(self):
        noisy = (
            "\x1b[32mBuilding...\x1b[0m\n"
            " 50%|█████░░░░░| 50/100\n"
            + "\n".join(["error: connection refused"] * 5)
            + "\nBuild finished"
        )
        cleaned, removed = cleaner.clean_text(noisy)
        assert "\x1b" not in cleaned
        assert "█" not in cleaned
        assert cleaned.count("error: connection refused") == 1
        assert "Build finished" in cleaned
        assert removed > 0

    def test_clean_input_zero_removed(self):
        cleaned, removed = cleaner.clean_text("hello world")
        assert cleaned == "hello world"
        assert removed == 0

    def test_empty_string(self):
        assert cleaner.clean_text("") == ("", 0)

    def test_idempotent(self):
        noisy = "\x1b[31mx\x1b[0m\n" + "\n".join(["dup"] * 5)
        once, _ = cleaner.clean_text(noisy)
        twice, removed = cleaner.clean_text(once)
        assert twice == once
        assert removed == 0
