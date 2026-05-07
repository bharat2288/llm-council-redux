"""Tests for llm_council.defaults — single source of truth for per-mode configs.

Both callers query this module:
  - The /council CLI skill (via `python -m llm_council defaults --mode <mode>`)
  - The /api/topfour/defaults HTTP endpoint (via the topfour_server's default
    handler, which calls the same module function)

Keeping defaults centralized is what enforces "/council CLI fires the same
config as Meshbook chat-dropdown council" — they both ask the engine.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from llm_council.defaults import default_config_for_mode
from llm_council.errors import ConfigError


# -- module-level function -------------------------------------------------


class TestDefaultConfigForMode:
    def test_free_3_model_returns_three_subscription_theorists(self) -> None:
        cfg = default_config_for_mode("free-3-model-with-gemini-cli")
        assert cfg["mode"] == "free-3-model-with-gemini-cli"
        names = [t["name"] for t in cfg["theorists"]]
        assert names == ["claude", "gpt", "gemini"]
        # All theorists on subscription CLIs (no openrouter routing).
        for t in cfg["theorists"]:
            assert t["routing"] in ("claude-cli", "codex-cli", "gemini-cli")
        # Reasoning-grade defaults.
        assert cfg["theorists"][0]["model"] == "claude-opus-4-7"
        assert cfg["theorists"][0]["effort"] == "xhigh"
        assert cfg["theorists"][1]["model"] == "gpt-5.5"
        assert cfg["theorists"][1]["effort"] == "xhigh"
        assert cfg["theorists"][2]["model"] == "gemini-3-pro-preview"
        # Chairman is also subscription (free).
        assert cfg["synthesizer"]["routing"] == "claude-cli"
        assert cfg["synthesizer"]["model"] == "claude-opus-4-7"

    def test_free_2_model_drops_gemini(self) -> None:
        cfg = default_config_for_mode("free-2-model")
        assert cfg["mode"] == "free-2-model"
        names = [t["name"] for t in cfg["theorists"]]
        assert names == ["claude", "gpt"]
        # Both subscription CLIs.
        for t in cfg["theorists"]:
            assert t["routing"] in ("claude-cli", "codex-cli")
        # Same reasoning-grade defaults as free-3-model for the two it includes.
        assert cfg["theorists"][0]["model"] == "claude-opus-4-7"
        assert cfg["theorists"][0]["effort"] == "xhigh"

    def test_standard_paid_routes_only_grok_via_openrouter(self) -> None:
        # Standard-paid is "free-3 + Grok via OpenRouter" — the only theorist
        # on openrouter is grok, since the others have free subscription CLIs.
        # This is the operating principle ("free wherever possible") encoded
        # as a structural test.
        cfg = default_config_for_mode("standard-paid")
        assert cfg["mode"] == "standard-paid"
        names = [t["name"] for t in cfg["theorists"]]
        assert names == ["claude", "gpt", "gemini", "grok"]

        openrouter_theorists = [t for t in cfg["theorists"] if t["routing"] == "openrouter"]
        assert len(openrouter_theorists) == 1, (
            "standard-paid must route only Grok via openrouter; "
            f"found {len(openrouter_theorists)} on openrouter: "
            f"{[t['name'] for t in openrouter_theorists]}"
        )
        assert openrouter_theorists[0]["name"] == "grok"
        assert openrouter_theorists[0]["model"] == "x-ai/grok-4.3"

        # Chairman stays on subscription (free) — no reason to pay for synthesis.
        assert cfg["synthesizer"]["routing"] == "claude-cli"

    def test_unknown_mode_raises_config_error(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            default_config_for_mode("not-a-real-mode")
        # Error mentions the offending mode + lists known modes.
        assert "not-a-real-mode" in str(excinfo.value)

    def test_returned_config_passes_full_config_validation(self) -> None:
        # The returned dict should be parseable by config.parse_config without
        # adding a topic/project (those come from the caller, not the defaults).
        # Add minimal topic/project here just to exercise the full parser.
        from llm_council.config import parse_config

        for mode in (
            "free-3-model-with-gemini-cli",
            "free-2-model",
            "standard-paid",
        ):
            base = default_config_for_mode(mode)
            full = {**base, "topic": "test", "project": "demo"}
            cfg = parse_config(full)  # raises on shape problems
            assert cfg.mode == mode

    def test_returned_dict_is_independent_per_call(self) -> None:
        # Mutating one returned config must not affect a subsequent call.
        # (Defends against `return _CONSTANT_DICT` style implementations.)
        a = default_config_for_mode("free-3-model-with-gemini-cli")
        a["theorists"][0]["model"] = "MUTATED"
        b = default_config_for_mode("free-3-model-with-gemini-cli")
        assert b["theorists"][0]["model"] == "claude-opus-4-7"


# -- CLI subcommand --------------------------------------------------------


class TestDefaultsCLI:
    """`python -m llm_council defaults --mode <mode>` prints the JSON config.

    Subprocess-based test (rather than calling cli.main directly) so we
    catch issues with __main__ wiring / argparse subcommand registration.
    """

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "llm_council", *args],
            capture_output=True,
            text=True,
            cwd=r"C:\Users\bhara\dev\llm-council",
        )

    def test_known_mode_prints_json_exits_zero(self) -> None:
        result = self._run("defaults", "--mode", "free-3-model-with-gemini-cli")
        assert result.returncode == 0, result.stderr
        # stdout parses as JSON.
        cfg = json.loads(result.stdout)
        assert cfg["mode"] == "free-3-model-with-gemini-cli"
        assert len(cfg["theorists"]) == 3

    def test_each_known_mode_prints_valid_json(self) -> None:
        for mode in (
            "free-3-model-with-gemini-cli",
            "free-2-model",
            "standard-paid",
        ):
            result = self._run("defaults", "--mode", mode)
            assert result.returncode == 0, f"{mode}: {result.stderr}"
            cfg = json.loads(result.stdout)
            assert cfg["mode"] == mode

    def test_unknown_mode_exits_nonzero_with_useful_message(self) -> None:
        result = self._run("defaults", "--mode", "not-a-real-mode")
        assert result.returncode != 0
        assert "not-a-real-mode" in (result.stderr + result.stdout)

    def test_missing_mode_arg_exits_nonzero(self) -> None:
        result = self._run("defaults")
        assert result.returncode != 0
