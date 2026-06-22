"""ConPTY capture test for Antigravity CLI (`agy`).

agy writes its response to the Windows console, not to the stdout file
descriptor, so subprocess pipes capture nothing. This harness gives agy a
pseudo-console (ConPTY via pywinpty) that WE own and read, then shows both the
raw bytes (so we can see terminal control codes / agentic chrome) and an
ANSI-stripped version (what council would actually extract).

Run from a REAL terminal:

    pip install pywinpty      # if not already installed
    python verify_agy_pty.py

Decision this answers:
  - Does -p output come through the PTY at all?  (capture viability)
  - Is it clean text, or wrapped in spinner/header/token-count chrome?
    (how much parsing routing.py needs)
  - Does a wide PTY avoid hard-wrapping the response?  (cols sizing)
"""
from __future__ import annotations

import re
import shutil
import sys
import time

PROMPT = (
    "In exactly two sentences, explain what a council of LLMs is. "
    "End your reply with the literal token DONE_SENTINEL."
)

# Strip ANSI CSI/OSC escape sequences. Council only wants the text payload.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07|\x1b[@-Z\\-_]")


def strip_ansi(s: str) -> str:
    return _ANSI.sub("", s)


def main() -> int:
    exe = shutil.which("agy")
    if not exe:
        print("FATAL: agy not on PATH")
        return 1

    try:
        from winpty import PtyProcess  # pywinpty
    except ImportError:
        print("FATAL: pywinpty not installed. Run:  pip install pywinpty")
        return 1

    # Wide + tall so agy doesn't hard-wrap the response at 80 cols.
    # Pass a LIST (not a quoted string): pywinpty shlex-splits strings with
    # posix=False and keeps the quotes, which breaks paths/args. A list goes
    # through list2cmdline and quotes correctly.
    argv = [exe, "--print", PROMPT]
    print(f"spawn argv: {argv}\n(cols=250, rows=50)\n")

    proc = PtyProcess.spawn(argv, dimensions=(50, 250))
    chunks: list[str] = []
    t0 = time.monotonic()
    deadline = t0 + 180
    while True:
        try:
            data = proc.read(8192)
        except EOFError:
            break
        if data:
            chunks.append(data)
        elif not proc.isalive():
            break
        if time.monotonic() > deadline:
            print("\n[timeout 180s — killing]")
            try:
                proc.terminate(force=True)
            except Exception:
                pass
            break

    raw = "".join(chunks)
    cleaned = strip_ansi(raw)
    dt = time.monotonic() - t0

    print(f"--- elapsed {dt:.1f}s, raw_chars={len(raw)}, cleaned_chars={len(cleaned)} ---\n")
    print("===== RAW (repr, first 1500 chars) — shows control codes/chrome =====")
    print(repr(raw[:1500]))
    print("\n===== ANSI-STRIPPED (what council would extract) =====")
    print(cleaned)
    print("\n===== sentinel present in cleaned output? =====")
    print("DONE_SENTINEL found:", "DONE_SENTINEL" in cleaned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
