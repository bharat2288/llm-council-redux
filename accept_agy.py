"""End-to-end acceptance test for agy-cli routing in the real engine path.

Calls the actual engine entrypoint `routing.fire_theorist` with the agy-cli
spec council uses by default — so this exercises _fire_agy_cli, the
--model "Gemini 3.1 Pro (High)" acceptance, the ConPTY capture, and the
ANSI-strip, exactly as a live /council run would.

Run from a REAL terminal (a pseudo-console is created by pywinpty, but auth +
model selection need the real agy environment):

    python accept_agy.py

PASS = success True, non-empty content, sentinel present.
"""
from __future__ import annotations

from llm_council.config import TheoristSpec
from llm_council.routing import fire_theorist

SPEC = TheoristSpec(
    name="gemini",
    model="Gemini 3.1 Pro (High)",
    effort="high",
    routing="agy-cli",
)
PROMPT = (
    "In exactly two sentences, explain what a multi-model LLM council is and "
    "why it can beat a single model. End your reply with the token DONE_SENTINEL."
)


def main() -> int:
    print("Firing agy-cli theorist through the real engine path...")
    r = fire_theorist(SPEC, PROMPT, timeout_seconds=180)
    print(f"\nsuccess:  {r.success}")
    print(f"routing:  {r.routing}")
    print(f"model:    {r.model}")
    print(f"duration: {r.duration_seconds:.1f}s")
    print(f"error:    {r.error}")
    print(f"\n--- content ({len(r.content)} chars) ---\n{r.content}")

    ok = r.success and bool(r.content.strip()) and "DONE_SENTINEL" in r.content
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"success={r.success}, non_empty={bool(r.content.strip())}, "
          f"sentinel={'DONE_SENTINEL' in r.content}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
