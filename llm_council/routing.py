"""Per-theorist routing dispatch.

Each routing path translates a (model, effort, prompt) triple into:
  - a subprocess invocation (subscription CLIs)
  - or an HTTP call (OpenRouter)
and returns a TheoristResult with the response text + usage metadata.

All routings are blocking by design — `fire_theorist` is called concurrently
from `cli.py` via threading so the four theorists run in parallel without
needing async/aiohttp here. Keeping each routing synchronous makes failure
isolation simpler and lets us swap in real claude-cli/codex-cli subprocesses
without an asyncio bridge.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from llm_council.config import TheoristSpec
from llm_council.errors import RoutingError, TheoristFailure


THEORIST_SYSTEM = (
    "You are a senior specialist on a Council of frontier AI models. "
    "Give concrete, actionable, correct guidance on the user's question. "
    "Be thorough, opinionated, and call out weaknesses in any premise the user states. "
    "Prefer specific commands and code where they help."
)


@dataclass(frozen=True)
class TheoristResult:
    name: str
    model: str
    routing: str
    success: bool
    content: str
    error: Optional[str] = None
    cost_usd: float = 0.0  # 0 for subscription-CLI paths; populated for OpenRouter
    duration_seconds: float = 0.0


def fire_theorist(spec: TheoristSpec, prompt: str, timeout_seconds: int = 600) -> TheoristResult:
    """Dispatch one theorist and return its result. Never raises — returns
    a TheoristResult with success=False on failure so the caller can keep
    going with the survivors."""
    import time

    t0 = time.monotonic()
    try:
        if spec.routing == "claude-cli":
            content = _fire_claude_cli(spec, prompt, timeout_seconds)
            cost = 0.0
        elif spec.routing == "codex-cli":
            content = _fire_codex_cli(spec, prompt, timeout_seconds)
            cost = 0.0
        elif spec.routing == "gemini-cli":
            content = _fire_gemini_cli(spec, prompt, timeout_seconds)
            cost = 0.0
        elif spec.routing == "openrouter":
            content, cost = _fire_openrouter(spec, prompt, timeout_seconds)
        else:
            raise RoutingError(f"unknown routing: {spec.routing!r}")
        return TheoristResult(
            name=spec.name,
            model=spec.model,
            routing=spec.routing,
            success=True,
            content=content,
            cost_usd=cost,
            duration_seconds=time.monotonic() - t0,
        )
    except Exception as exc:  # noqa: BLE001 — caller decides what to do with failures
        return TheoristResult(
            name=spec.name,
            model=spec.model,
            routing=spec.routing,
            success=False,
            content="",
            error=str(exc),
            duration_seconds=time.monotonic() - t0,
        )


# ---------- subscription CLI paths ----------


def _resolve_binary(name: str, hint: str) -> str:
    """Resolve a binary's full path with extension. Required on Windows so
    subprocess can launch .CMD/.BAT shims (e.g. codex via fnm) without
    needing shell=True."""
    path = shutil.which(name)
    if not path:
        raise RoutingError(hint)
    return path


def _fire_claude_cli(spec: TheoristSpec, prompt: str, timeout: int) -> str:
    exe = _resolve_binary(
        "claude",
        "claude-cli not on PATH. Install Claude Code or run from a shell "
        "where `claude --version` succeeds.",
    )
    args = [
        exe,
        "--print",
        "--dangerously-skip-permissions",  # one-shot theorist; no tool prompts
        "--model",
        spec.model,
    ]
    return _run_subprocess(args, _wrap_with_system(prompt), spec.name, timeout)


def _fire_codex_cli(spec: TheoristSpec, prompt: str, timeout: int) -> str:
    exe = _resolve_binary(
        "codex",
        "codex-cli not on PATH. Install Codex CLI and run `codex login` first.",
    )
    # codex uses `exec` subcommand for non-interactive runs.
    #   -m <model>             pick the model directly
    #   --sandbox read-only    council theorists don't write code; cap blast radius
    #   --skip-git-repo-check  don't refuse outside a git repo
    args = [
        exe,
        "exec",
        "-m",
        spec.model,
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
    ]
    return _run_subprocess(args, _wrap_with_system(prompt), spec.name, timeout)


def _fire_gemini_cli(spec: TheoristSpec, prompt: str, timeout: int) -> str:
    exe = _resolve_binary(
        "gemini",
        "gemini-cli not on PATH. The free-3-model preset depends on a "
        "Gemini CLI subprocess interface that is not yet available on this "
        "machine (chat-surface ADR Open Item #6). Switch to free-2-model "
        "or standard-paid (with op-run wrapper).",
    )
    # Gemini CLI exists in some preview form; signature TBD. Refine when
    # the binary actually ships and we can probe `gemini --help`.
    args = [exe, "--print", "--model", spec.model]
    return _run_subprocess(args, _wrap_with_system(prompt), spec.name, timeout)


def _wrap_with_system(prompt: str) -> str:
    """Subscription CLIs don't take a separate --system flag uniformly, so
    we prepend the system framing as a leading block. Each model's prefix
    handler will treat this as part of the user message; the framing is
    explicit enough that they all roleplay correctly."""
    return f"[Council theorist briefing]\n{THEORIST_SYSTEM}\n\n[Operator question]\n{prompt}"


def _run_subprocess(args: list[str], stdin: str, name: str, timeout: int) -> str:
    proc = subprocess.run(
        args,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise TheoristFailure(
            name,
            f"subprocess rc={proc.returncode}: {' | '.join(stderr_tail)}",
        )
    out = (proc.stdout or "").strip()
    if not out:
        raise TheoristFailure(name, "subprocess returned empty stdout")
    return out


# ---------- OpenRouter (paid) path ----------


def _fire_openrouter(spec: TheoristSpec, prompt: str, timeout: int) -> tuple[str, float]:
    """Direct (sync) HTTP call to OpenRouter. Requires OPENROUTER_API_KEY."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RoutingError(
            "OPENROUTER_API_KEY not set. The 'openrouter' routing requires the "
            "key to be in the environment. Common fix: launch the council under "
            "the op-run wrapper so 1Password injects it (see "
            "C:\\launcher\\llm-windows.op.env), e.g.:\n"
            "  op run --env-file=C:\\launcher\\llm-windows.op.env -- "
            "python -m llm_council fire --config <path> --output-dir <dir>\n"
            "Or switch this run to a free-2-model preset which uses claude-cli "
            "+ codex-cli subscriptions (no API key needed)."
        )
    # Imported here so the package doesn't hard-require aiohttp on subscription
    # paths. urllib + json keep the dependency surface minimal.
    import urllib.request
    import urllib.error

    payload = {
        "model": spec.model,
        "max_tokens": 8000,
        "messages": [
            {"role": "system", "content": THEORIST_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": spec.effort},
    }
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/llm-council",
            "X-Title": "Council",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise TheoristFailure(spec.name, f"OpenRouter HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise TheoristFailure(spec.name, f"OpenRouter unreachable: {exc}") from exc

    choice = data["choices"][0]["message"]
    content = choice.get("content") or choice.get("reasoning") or ""
    if not content:
        raise TheoristFailure(spec.name, "OpenRouter returned empty content")
    usage = data.get("usage") or {}
    cost = float(usage.get("cost", 0) or 0)
    return content, cost
