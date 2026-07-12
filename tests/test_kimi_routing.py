"""Tests for Kimi Code CLI routing."""
from __future__ import annotations

from llm_council.config import TheoristSpec
from llm_council.routing import _fire_kimi_cli, _parse_kimi_stream_json


def test_parse_kimi_stream_json_keeps_assistant_content_only() -> None:
    stdout = "\n".join(
        [
            '{"role":"assistant","content":"First chunk"}',
            '{"role":"meta","type":"session.resume_hint","content":"To resume this session: kimi -r session_x"}',
            '{"role":"assistant","content":"Second chunk"}',
        ]
    )

    assert _parse_kimi_stream_json(stdout) == "First chunk\nSecond chunk"


def test_kimi_cli_uses_default_model_sentinel_without_model_flag(monkeypatch) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr("llm_council.routing._resolve_binary", lambda name, hint: "kimi")

    def fake_run(args, stdin, name, timeout):
        seen["args"] = args
        seen["stdin"] = stdin
        seen["name"] = name
        seen["timeout"] = timeout
        return '{"role":"assistant","content":"Kimi response"}\n{"role":"meta","content":"ignored"}'

    monkeypatch.setattr("llm_council.routing._run_subprocess", fake_run)

    spec = TheoristSpec(
        name="kimi",
        model="kimi-default",
        effort="high",
        routing="kimi-cli",
    )

    assert _fire_kimi_cli(spec, "Question?", 123, ("C:/work/specs",)) == "Kimi response"
    args = seen["args"]
    assert "-m" not in args
    assert args[:5] == ["kimi", "-p", args[2], "--output-format", "stream-json"]
    assert "-y" not in args
    assert args[-2:] == ["--add-dir", "C:/work/specs"]
    assert seen["stdin"] == ""
    assert seen["name"] == "kimi"
    assert seen["timeout"] == 123


def test_kimi_cli_passes_operator_model_alias(monkeypatch) -> None:
    monkeypatch.setattr("llm_council.routing._resolve_binary", lambda name, hint: "kimi")
    seen: dict[str, object] = {}

    def fake_run(args, stdin, name, timeout):
        seen["args"] = args
        return '{"role":"assistant","content":"Custom model response"}'

    monkeypatch.setattr("llm_council.routing._run_subprocess", fake_run)
    spec = TheoristSpec(
        name="kimi",
        model="kimi-k2",
        effort="high",
        routing="kimi-cli",
    )

    assert _fire_kimi_cli(spec, "Question?", 600) == "Custom model response"
    args = seen["args"]
    assert args[args.index("-m") + 1] == "kimi-k2"
