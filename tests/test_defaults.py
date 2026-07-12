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
        cfg = default_config_for_mode("free-3-model-with-agy")
        assert cfg["mode"] == "free-3-model-with-agy"
        names = [t["name"] for t in cfg["theorists"]]
        assert names == ["claude", "gpt", "gemini"]
        # All theorists on subscription CLIs (no openrouter routing).
        for t in cfg["theorists"]:
            assert t["routing"] in ("claude-cli", "codex-cli", "agy-cli")
        # Reasoning-grade defaults.
        assert cfg["theorists"][0]["model"] == "claude-opus-4-8"
        assert cfg["theorists"][0]["effort"] == "xhigh"
        assert cfg["theorists"][1]["model"] == "gpt-5.6-terra"
        assert cfg["theorists"][1]["effort"] == "max"
        assert cfg["theorists"][2]["model"] == "Gemini 3.1 Pro (High)"
        assert cfg["theorists"][2]["routing"] == "agy-cli"
        # Chairman is also subscription (free) — gpt-5.6-sol so the chair
        # doesn't share its exact model with any participant.
        assert cfg["synthesizer"]["routing"] == "codex-cli"
        assert cfg["synthesizer"]["model"] == "gpt-5.6-sol"
        assert cfg["synthesizer"]["effort"] == "max"

    def test_legacy_free_3_gemini_cli_mode_remains_available(self) -> None:
        cfg = default_config_for_mode("free-3-model-with-gemini-cli")
        assert cfg["mode"] == "free-3-model-with-gemini-cli"
        assert cfg["theorists"][2]["name"] == "gemini"
        assert cfg["theorists"][2]["routing"] == "gemini-cli"
        assert cfg["theorists"][2]["model"] == "gemini-3-pro-preview"

    def test_free_4_model_with_kimi_adds_kimi_subscription_theorist(self) -> None:
        cfg = default_config_for_mode("free-4-model-with-kimi")
        assert cfg["mode"] == "free-4-model-with-kimi"
        names = [t["name"] for t in cfg["theorists"]]
        assert names == ["claude", "gpt", "gemini", "kimi"]
        for t in cfg["theorists"]:
            assert t["routing"] in ("claude-cli", "codex-cli", "agy-cli", "kimi-cli")
        assert cfg["theorists"][3] == {
            "name": "kimi",
            "model": "kimi-default",
            "effort": "high",
            "routing": "kimi-cli",
        }
        assert cfg["synthesizer"]["routing"] == "codex-cli"
        assert cfg["synthesizer"]["model"] == "gpt-5.6-sol"

    def test_free_2_model_drops_gemini(self) -> None:
        cfg = default_config_for_mode("free-2-model")
        assert cfg["mode"] == "free-2-model"
        names = [t["name"] for t in cfg["theorists"]]
        assert names == ["claude", "gpt"]
        # Both subscription CLIs.
        for t in cfg["theorists"]:
            assert t["routing"] in ("claude-cli", "codex-cli")
        # Same reasoning-grade defaults as free-3-model for the two it includes.
        assert cfg["theorists"][0]["model"] == "claude-opus-4-8"
        assert cfg["theorists"][0]["effort"] == "xhigh"

    def test_standard_paid_routes_grok_and_glm_via_openrouter(self) -> None:
        # Standard-paid is "free-3 + paid frontier perspectives" — Claude,
        # GPT, and Gemini stay on subscription CLIs, while Grok and GLM
        # route through OpenRouter.
        # This is the operating principle ("free wherever possible") encoded
        # as a structural test.
        cfg = default_config_for_mode("standard-paid")
        assert cfg["mode"] == "standard-paid"
        names = [t["name"] for t in cfg["theorists"]]
        assert names == ["claude", "gpt", "gemini", "grok", "glm"]

        openrouter_theorists = [t for t in cfg["theorists"] if t["routing"] == "openrouter"]
        assert len(openrouter_theorists) == 2, (
            "standard-paid must route Grok and GLM via openrouter; "
            f"found {len(openrouter_theorists)} on openrouter: "
            f"{[t['name'] for t in openrouter_theorists]}"
        )
        assert [(t["name"], t["model"]) for t in openrouter_theorists] == [
            ("grok", "x-ai/grok-4.3"),
            ("glm", "z-ai/glm-5.2"),
        ]

        # Chairman stays on subscription (free) — no reason to pay for synthesis.
        assert cfg["synthesizer"]["routing"] == "codex-cli"

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
            "free-3-model-with-agy",
            "free-4-model-with-kimi",
            "free-3-model-with-gemini-cli",
            "free-2-model",
            "standard-paid",
        ):
            base = default_config_for_mode(mode)
            full = {**base, "topic": "test", "project": "demo"}
            cfg = parse_config(full)  # raises on shape problems
            assert cfg.mode == mode

    def test_fable_toggle_off_by_default(self, monkeypatch) -> None:
        monkeypatch.delenv("LLM_COUNCIL_USE_FABLE", raising=False)
        cfg = default_config_for_mode("free-4-model-with-kimi")
        assert cfg["theorists"][0]["model"] == "claude-opus-4-8"
        assert cfg["theorists"][1]["model"] == "gpt-5.6-terra"
        assert cfg["synthesizer"]["model"] == "gpt-5.6-sol"

    def test_fable_toggle_rebalances_every_mode(self, monkeypatch) -> None:
        # LC-4 (ADR-003): Fable takes the claude seat and the chair; sol
        # vacates the chair and takes the gpt participant seat.
        monkeypatch.setenv("LLM_COUNCIL_USE_FABLE", "1")
        for mode in (
            "free-3-model-with-agy",
            "free-4-model-with-kimi",
            "free-3-model-with-gemini-cli",
            "free-2-model",
            "standard-paid",
        ):
            cfg = default_config_for_mode(mode)
            by_name = {t["name"]: t for t in cfg["theorists"]}
            assert by_name["claude"]["model"] == "claude-fable-5", mode
            assert by_name["claude"]["effort"] == "max", mode
            assert by_name["claude"]["routing"] == "claude-cli", mode
            assert by_name["gpt"]["model"] == "gpt-5.6-sol", mode
            assert by_name["gpt"]["effort"] == "max", mode
            assert cfg["synthesizer"] == {
                "model": "claude-fable-5",
                "effort": "max",
                "routing": "claude-cli",
            }, mode
            # Non-claude/gpt seats untouched.
            if "gemini" in by_name:
                assert by_name["gemini"]["model"] in (
                    "Gemini 3.1 Pro (High)",
                    "gemini-3-pro-preview",
                ), mode

    def test_fable_toggle_output_passes_config_validation(self, monkeypatch) -> None:
        from llm_council.config import parse_config

        monkeypatch.setenv("LLM_COUNCIL_USE_FABLE", "true")
        base = default_config_for_mode("free-4-model-with-kimi")
        cfg = parse_config({**base, "topic": "test", "project": "demo"})
        assert cfg.synthesizer.model == "claude-fable-5"

    def test_fable_toggle_does_not_pollute_canonical_presets(self, monkeypatch) -> None:
        # The rebalance must operate on the deep copy, never the module
        # constants — a toggled call followed by an untoggled call returns
        # the canonical config.
        monkeypatch.setenv("LLM_COUNCIL_USE_FABLE", "1")
        default_config_for_mode("free-4-model-with-kimi")
        monkeypatch.setenv("LLM_COUNCIL_USE_FABLE", "0")
        cfg = default_config_for_mode("free-4-model-with-kimi")
        assert cfg["theorists"][0]["model"] == "claude-opus-4-8"
        assert cfg["synthesizer"]["model"] == "gpt-5.6-sol"

    def test_fable_env_gibberish_raises_config_error(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_COUNCIL_USE_FABLE", "maybe")
        with pytest.raises(ConfigError) as excinfo:
            default_config_for_mode("free-2-model")
        assert "LLM_COUNCIL_USE_FABLE" in str(excinfo.value)

    def test_returned_dict_is_independent_per_call(self) -> None:
        # Mutating one returned config must not affect a subsequent call.
        # (Defends against `return _CONSTANT_DICT` style implementations.)
        a = default_config_for_mode("free-3-model-with-agy")
        a["theorists"][0]["model"] = "MUTATED"
        b = default_config_for_mode("free-3-model-with-agy")
        assert b["theorists"][0]["model"] == "claude-opus-4-8"


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
        result = self._run("defaults", "--mode", "free-3-model-with-agy")
        assert result.returncode == 0, result.stderr
        # stdout parses as JSON.
        cfg = json.loads(result.stdout)
        assert cfg["mode"] == "free-3-model-with-agy"
        assert len(cfg["theorists"]) == 3
        assert cfg["theorists"][2]["routing"] == "agy-cli"

    def test_each_known_mode_prints_valid_json(self) -> None:
        for mode in (
            "free-3-model-with-agy",
            "free-4-model-with-kimi",
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
