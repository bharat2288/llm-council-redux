"""llm_council — multi-model frontier council engine.

Public callers:
  - `/council` skill at ~/.claude/skills/council/ (subprocess: `python -m llm_council fire`)
  - Meshbook chat dropdown via the HTTP daemon at /api/topfour/stream (pipeline #63)
  - Legacy `council_topfour.py` flat CLI in this same repo (kept for backward compat)

The package implements per-theorist routing so the same engine can run on:
  - subscription CLIs (claude-cli, codex-cli, agy-cli for Gemini) — $0 path,
    no API keys needed, auth inherited from ~/.claude/, ~/.codex/, etc.
  - OpenRouter API — paid path, requires OPENROUTER_API_KEY in env
  - direct provider APIs — fallback path (not yet implemented in v0)

Engine v0 (today): subscription-CLI routing for claude-cli, codex-cli, and agy-cli is
fully functional; OpenRouter routing requires the user to launch the
process under `op run --env-file=...` to inject the API key.
"""

__version__ = "0.1.0"
