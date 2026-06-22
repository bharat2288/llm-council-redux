"""Single source of truth for per-mode council configs.

Both callers of the engine query this module:

  - The /council CLI skill at ~/.claude/skills/council/SKILL.md, via
    `python -m llm_council defaults --mode <mode>` (cli.py wires the
    subcommand into the same argparse parser as `fire` and `preflight`).

  - The /api/topfour HTTP daemon (pipeline #63), via
    `GET /api/topfour/defaults?mode=<mode>` — handler calls
    `default_config_for_mode(mode)` directly.

Centralizing here is what enforces the "operator gets the same defaults
whether firing /council from a terminal or selecting Council in the
Meshbook chat dropdown" contract. Bumping a default here propagates to
both callers automatically; no two-copies-in-sync drift.

Operating principle (encoded structurally below): use free subscription
CLIs wherever a provider ships one. Today that means claude-cli (Claude),
codex-cli (GPT), and agy / Antigravity CLI (Gemini 3.1 Pro). The only
theorist that genuinely requires paid routing is Grok — xAI has no
subprocess CLI — so it's the sole `routing: openrouter` theorist in
`standard-paid`.

Gemini routing migrated from gemini-cli to agy (Antigravity) on 2026-06-22:
Google sunset the standalone Gemini CLI in favor of Antigravity. agy writes
to the console rather than stdout, so agy-cli routing captures it through a
ConPTY (see routing._fire_agy_cli). The legacy `free-3-model-with-gemini-cli`
mode is kept as a fallback while the `gemini` binary still works.

Reasoning-grade only: the slugs here are deliberately reasoning-grade.
A fast-model council defeats the council's purpose; the skill never
suggests fast-model overrides either.
"""
from __future__ import annotations

import copy
from typing import Any

from llm_council.errors import ConfigError


# -- canonical per-mode configs --------------------------------------------

# Default mode. Gemini rides on agy (Antigravity CLI), whose default model is
# "Gemini 3.1 Pro (High)" — passed explicitly so the run doesn't depend on the
# operator's local agy default. agy has no effort flag (effort is baked into
# the model label), so `effort` here is recorded-only, same as gemini-cli.
_FREE_3_MODEL_WITH_AGY: dict[str, Any] = {
    "mode": "free-3-model-with-agy",
    "theorists": [
        {"name": "claude", "model": "claude-opus-4-8",         "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.5",                  "effort": "xhigh", "routing": "codex-cli"},
        {"name": "gemini", "model": "Gemini 3.1 Pro (High)",    "effort": "high",  "routing": "agy-cli"},
    ],
    "synthesizer": {"model": "claude-opus-4-8", "effort": "xhigh", "routing": "claude-cli"},
}


# Legacy fallback — kept while the standalone `gemini` binary still works.
# Prefer `free-3-model-with-agy`; Google sunset the standalone Gemini CLI.
_FREE_3_MODEL_WITH_GEMINI_CLI: dict[str, Any] = {
    "mode": "free-3-model-with-gemini-cli",
    "theorists": [
        {"name": "claude", "model": "claude-opus-4-8",     "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.5",              "effort": "xhigh", "routing": "codex-cli"},
        {"name": "gemini", "model": "gemini-3-pro-preview", "effort": "high",  "routing": "gemini-cli"},
    ],
    "synthesizer": {"model": "claude-opus-4-8", "effort": "xhigh", "routing": "claude-cli"},
}


_FREE_2_MODEL: dict[str, Any] = {
    "mode": "free-2-model",
    "theorists": [
        {"name": "claude", "model": "claude-opus-4-8", "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.5",          "effort": "xhigh", "routing": "codex-cli"},
    ],
    "synthesizer": {"model": "claude-opus-4-8", "effort": "xhigh", "routing": "claude-cli"},
}


_STANDARD_PAID: dict[str, Any] = {
    "mode": "standard-paid",
    "theorists": [
        # First three: free subscription CLIs (same as free-3-model-with-agy).
        {"name": "claude", "model": "claude-opus-4-8",      "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.5",               "effort": "xhigh", "routing": "codex-cli"},
        {"name": "gemini", "model": "Gemini 3.1 Pro (High)", "effort": "high",  "routing": "agy-cli"},
        # Grok via OpenRouter — the only paid leg. xAI ships no subprocess
        # CLI as of 2026-05-07. OpenRouter's reasoning.effort ceiling is
        # `high`, so don't pass xhigh here even though Grok itself supports
        # deeper reasoning — OpenRouter would reject it.
        {"name": "grok",   "model": "x-ai/grok-4.3",        "effort": "high",  "routing": "openrouter"},
    ],
    # Chairman stays on free claude-cli — no reason to pay for synthesis.
    "synthesizer": {"model": "claude-opus-4-8", "effort": "xhigh", "routing": "claude-cli"},
}


_BY_MODE: dict[str, dict[str, Any]] = {
    "free-3-model-with-agy": _FREE_3_MODEL_WITH_AGY,
    "free-3-model-with-gemini-cli": _FREE_3_MODEL_WITH_GEMINI_CLI,
    "free-2-model": _FREE_2_MODEL,
    "standard-paid": _STANDARD_PAID,
}


# -- public API ------------------------------------------------------------


def known_modes() -> tuple[str, ...]:
    """List the modes this engine version knows about. Stable ordering."""
    return tuple(_BY_MODE.keys())


def default_config_for_mode(mode: str) -> dict[str, Any]:
    """Return a fresh deep copy of the canonical config for `mode`.

    The copy is intentional: callers may mutate the returned dict (e.g.,
    add a topic/project before passing to `config.parse_config`) and we
    don't want those mutations to leak across calls.

    Raises ConfigError with the offending mode + the known-modes list when
    `mode` isn't recognized.
    """
    if mode not in _BY_MODE:
        raise ConfigError(
            f"unknown mode: {mode!r}; known modes are {list(_BY_MODE.keys())}"
        )
    return copy.deepcopy(_BY_MODE[mode])
