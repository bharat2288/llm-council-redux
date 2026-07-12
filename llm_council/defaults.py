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
codex-cli (GPT), agy / Antigravity CLI (Gemini 3.1 Pro), and kimi-cli
(Kimi Code). The paid standard council adds frontier perspectives that do
not have local subscription-CLI routing here, so Grok and GLM ride through
OpenRouter.

Gemini routing migrated from gemini-cli to agy (Antigravity) on 2026-06-22:
Google sunset the standalone Gemini CLI in favor of Antigravity. agy writes
to the console rather than stdout, so agy-cli routing captures it through a
ConPTY (see routing._fire_agy_cli). The legacy `free-3-model-with-gemini-cli`
mode is kept as a fallback while the `gemini` binary still works.

Reasoning-grade only: the slugs here are deliberately reasoning-grade.
A fast-model council defeats the council's purpose; the skill never
suggests fast-model overrides either.

GPT seat + chairman (2026-07-12): the GPT theorist rides gpt-5.6-terra at
max effort; the chairman is gpt-5.6-sol at max effort via codex-cli (both
verified live 2026-07-12). Moving the chair off claude-opus-4-8 removes the
same-model-as-participant synthesis bias (the chair previously shared its
exact model with the claude theorist). GPT-5.6 also exposes an `ultra`
effort tier — deliberately not used: ultra enables subagent delegation,
which is pointless overhead for one-shot council responses.
"""
from __future__ import annotations

import copy
import os
from typing import Any

from llm_council.errors import ConfigError


# -- Fable rebalance toggle (LC-4, ADR-003) ---------------------------------

# When Fable (Anthropic's Mythos-class tier above Opus) is available on the
# claude-cli subscription, flip _USE_FABLE_DEFAULT to True. Every preset is
# then rebalanced at read time:
#   - claude theorist:  claude-opus-4-8            -> claude-fable-5
#   - gpt theorist:     gpt-5.6-terra              -> gpt-5.6-sol (max) —
#                       sol vacates the chair and takes the participant seat
#                       as the stronger 5.6 variant
#   - chairman:         gpt-5.6-sol (codex-cli)    -> claude-fable-5 (max,
#                       claude-cli)
# Known tradeoff, accepted in ADR-003: with the toggle on, the chair shares
# its exact model with the claude participant (both Fable) — the same
# same-model synthesis bias the sol chair removed for Opus. Accepted because
# the judgment-densest seat should hold the strongest available model.
# Per-run override without editing code: LLM_COUNCIL_USE_FABLE=1|0.
# Slug + max effort verified live through claude-cli print mode 2026-07-12.
_USE_FABLE_DEFAULT = False
FABLE_MODEL = "claude-fable-5"

_ENV_TRUE = frozenset({"1", "true", "yes", "on"})
_ENV_FALSE = frozenset({"0", "false", "no", "off"})


def fable_enabled() -> bool:
    """Is the Fable rebalance active? Env override beats the module default."""
    raw = os.environ.get("LLM_COUNCIL_USE_FABLE")
    if raw is None:
        return _USE_FABLE_DEFAULT
    value = raw.strip().lower()
    if value in _ENV_TRUE:
        return True
    if value in _ENV_FALSE:
        return False
    raise ConfigError(
        f"LLM_COUNCIL_USE_FABLE={raw!r} not understood; use 1/0, true/false, "
        "yes/no, or on/off"
    )


def _apply_fable_rebalance(cfg: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a preset config per the LC-4 swap. Mutates + returns `cfg`
    (callers pass a fresh deep copy)."""
    for theorist in cfg["theorists"]:
        if theorist["name"] == "claude" and theorist["routing"] == "claude-cli":
            theorist["model"] = FABLE_MODEL
            theorist["effort"] = "max"
        elif theorist["name"] == "gpt" and theorist["routing"] == "codex-cli":
            theorist["model"] = "gpt-5.6-sol"
            theorist["effort"] = "max"
    cfg["synthesizer"] = {
        "model": FABLE_MODEL,
        "effort": "max",
        "routing": "claude-cli",
    }
    return cfg


# -- canonical per-mode configs --------------------------------------------

# Default mode. Gemini rides on agy (Antigravity CLI), whose default model is
# "Gemini 3.1 Pro (High)" — passed explicitly so the run doesn't depend on the
# operator's local agy default. agy has no effort flag (effort is baked into
# the model label), so `effort` here is recorded-only, same as gemini-cli.
_FREE_3_MODEL_WITH_AGY: dict[str, Any] = {
    "mode": "free-3-model-with-agy",
    "theorists": [
        {"name": "claude", "model": "claude-opus-4-8",         "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.6-terra",            "effort": "max",   "routing": "codex-cli"},
        {"name": "gemini", "model": "Gemini 3.1 Pro (High)",    "effort": "high",  "routing": "agy-cli"},
    ],
    "synthesizer": {"model": "gpt-5.6-sol", "effort": "max", "routing": "codex-cli"},
}


_FREE_4_MODEL_WITH_KIMI: dict[str, Any] = {
    "mode": "free-4-model-with-kimi",
    "theorists": [
        {"name": "claude", "model": "claude-opus-4-8",      "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.6-terra",         "effort": "max",   "routing": "codex-cli"},
        {"name": "gemini", "model": "Gemini 3.1 Pro (High)", "effort": "high",  "routing": "agy-cli"},
        # `kimi-default` is a sentinel for "use the operator's Kimi Code
        # default_model". routing._fire_kimi_cli omits -m for this value; an
        # operator override can still pass any configured Kimi model alias.
        {"name": "kimi",   "model": "kimi-default",          "effort": "high",  "routing": "kimi-cli"},
    ],
    "synthesizer": {"model": "gpt-5.6-sol", "effort": "max", "routing": "codex-cli"},
}


# Legacy fallback — kept while the standalone `gemini` binary still works.
# Prefer `free-3-model-with-agy`; Google sunset the standalone Gemini CLI.
_FREE_3_MODEL_WITH_GEMINI_CLI: dict[str, Any] = {
    "mode": "free-3-model-with-gemini-cli",
    "theorists": [
        {"name": "claude", "model": "claude-opus-4-8",     "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.6-terra",        "effort": "max",   "routing": "codex-cli"},
        {"name": "gemini", "model": "gemini-3-pro-preview", "effort": "high",  "routing": "gemini-cli"},
    ],
    "synthesizer": {"model": "gpt-5.6-sol", "effort": "max", "routing": "codex-cli"},
}


_FREE_2_MODEL: dict[str, Any] = {
    "mode": "free-2-model",
    "theorists": [
        {"name": "claude", "model": "claude-opus-4-8", "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.6-terra",    "effort": "max",   "routing": "codex-cli"},
    ],
    "synthesizer": {"model": "gpt-5.6-sol", "effort": "max", "routing": "codex-cli"},
}


_STANDARD_PAID: dict[str, Any] = {
    "mode": "standard-paid",
    "theorists": [
        # First three: free subscription CLIs (same as free-3-model-with-agy).
        {"name": "claude", "model": "claude-opus-4-8",      "effort": "xhigh", "routing": "claude-cli"},
        {"name": "gpt",    "model": "gpt-5.6-terra",         "effort": "max",   "routing": "codex-cli"},
        {"name": "gemini", "model": "Gemini 3.1 Pro (High)", "effort": "high",  "routing": "agy-cli"},
        # Paid legs via OpenRouter. OpenRouter's reasoning.effort ceiling is
        # `high`, so don't pass xhigh here even though Grok itself supports
        # deeper reasoning — OpenRouter would reject it. GLM-5.2 is exposed
        # by OpenRouter as z-ai/glm-5.2 (verified 2026-06-24).
        {"name": "grok",   "model": "x-ai/grok-4.3",        "effort": "high",  "routing": "openrouter"},
        {"name": "glm",    "model": "z-ai/glm-5.2",         "effort": "high",  "routing": "openrouter"},
    ],
    # Chairman stays on a free subscription CLI — no reason to pay for synthesis.
    "synthesizer": {"model": "gpt-5.6-sol", "effort": "max", "routing": "codex-cli"},
}


_BY_MODE: dict[str, dict[str, Any]] = {
    "free-3-model-with-agy": _FREE_3_MODEL_WITH_AGY,
    "free-4-model-with-kimi": _FREE_4_MODEL_WITH_KIMI,
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
    cfg = copy.deepcopy(_BY_MODE[mode])
    if fable_enabled():
        cfg = _apply_fable_rebalance(cfg)
    return cfg
