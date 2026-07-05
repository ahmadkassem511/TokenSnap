"""Text cleaning: ANSI escape stripping, progress-bar removal, and
consecutive-duplicate-line deduplication.

All functions are pure and offline-testable. `clean_text` is the pipeline
the proxy applies to every text payload in a request.
"""

import re
from typing import Tuple

# CSI sequences: ESC [ params intermediates final-byte  (colors, cursor moves, ...)
_ANSI_CSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
# OSC sequences: ESC ] ... BEL  or  ESC ] ... ESC \   (window titles, hyperlinks)
_ANSI_OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
# Remaining two-byte escapes: ESC + single char (RIS, charset selection, ...)
_ANSI_OTHER = re.compile(r"\x1b[@-Z\\-_=><]")

# Blocks/hashes/arrows that make up drawn progress bars
_BAR_BODY = re.compile(r"[█▓▒░■#]{3,}|[=\-]{5,}>")
# Numeric progress readouts: "42%", "3/120", "1.2 MB/s", "eta 0:02"
_BAR_STATS = re.compile(
    r"\d{1,3}\s*%|\b\d+\s*/\s*\d+\b|\d+(?:\.\d+)?\s*(?:it/s|B/s|KB|kB|MB|GB)\b|\beta\b",
    re.IGNORECASE,
)
# Lines that are nothing but spinner glyphs / dots
_SPINNER_ONLY = re.compile(r"^[\s.⠀-⣿|/\\\-+*]+$")


def strip_ansi(text: str) -> str:
    """Remove ANSI/VT100 escape sequences (colors, cursor control, titles)."""
    text = _ANSI_OSC.sub("", text)
    text = _ANSI_CSI.sub("", text)
    text = _ANSI_OTHER.sub("", text)
    return text


def _is_progress_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _SPINNER_ONLY.match(stripped) and len(stripped) <= 40:
        return True
    return bool(_BAR_BODY.search(stripped) and _BAR_STATS.search(stripped))


def strip_progress_bars(text: str) -> str:
    """Drop progress-bar/spinner lines and collapse carriage-return frames.

    Tools like pip/tqdm redraw a line many times using ``\\r``; only the
    final frame carries information, and even that is usually just a bar.
    """
    out_lines = []
    for line in text.split("\n"):
        # A trailing \r is a CRLF line ending (Windows), not a redraw
        line = line.rstrip("\r")
        # A redrawn line: keep only what was on screen last
        if "\r" in line:
            line = line.split("\r")[-1]
        if _is_progress_line(line):
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


def dedupe_consecutive_lines(text: str, min_run: int = 3) -> str:
    """Collapse runs of >= min_run identical non-empty lines into one line
    plus a repeat marker. Short runs are left alone to stay safe."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        j = i + 1
        while j < len(lines) and lines[j] == line:
            j += 1
        run = j - i
        if run >= min_run and line.strip():
            out.append(line)
            out.append("[tokensnap: previous line repeated %d more times]" % (run - 1))
        else:
            out.extend(lines[i:j])
        i = j
    return "\n".join(out)


def clean_text(text: str) -> Tuple[str, int]:
    """Full cleaning pipeline. Returns (cleaned_text, chars_removed)."""
    if not text or "\x1b" not in text and "\r" not in text and "\n" not in text:
        return text, 0
    original_len = len(text)
    cleaned = strip_ansi(text)
    cleaned = strip_progress_bars(cleaned)
    cleaned = dedupe_consecutive_lines(cleaned)
    return cleaned, max(0, original_len - len(cleaned))
