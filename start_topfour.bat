@echo off
title Topfour Council Server

REM ============================================================
REM Topfour Council Server (pipeline meshbook #63)
REM ============================================================
REM
REM HTTP daemon wrapping the llm_council engine for the Meshbook
REM chat-dropdown caller. CLI sibling: the /council skill at
REM ~/.claude/skills/council/SKILL.md fires the same engine via
REM subprocess.
REM
REM Endpoints (see topfour_server.py docstring):
REM   GET    /api/topfour/defaults?mode=<mode>
REM   POST   /api/topfour/start
REM   GET    /api/topfour/stream/<run_id>
REM   DELETE /api/topfour/<run_id>
REM
REM ============================================================
REM CREDENTIAL HANDLING (Pattern A from chat-surface ADR)
REM ============================================================
REM
REM This .bat is the simple launcher. The op-run wrapping that
REM injects OPENROUTER_API_KEY happens at the DSM layer — DSM is
REM configured to invoke this .bat under:
REM
REM   op run --env-file=C:\launcher\llm-windows.op.env -- start_topfour.bat
REM
REM 1Password modal pops once when DSM starts the daemon; the
REM secret persists in the daemon process env for its lifetime.
REM Per-request modals do NOT fire (good UX for chat-dropdown).
REM
REM Required reading before changing anything credential-adjacent:
REM   C:\Users\bhara\dev\specs\codex-workflow-system\playbook-credential-sop.md
REM   C:\Users\bhara\dev\specs\codex-workflow-system\playbook-credential-hardening.md
REM
REM Operating principle: free wherever possible. Subscription CLIs
REM (claude-cli, codex-cli, gemini-cli) don't need OPENROUTER_API_KEY
REM at all — only the Grok theorist on standard-paid does.
REM
REM ============================================================

echo ============================================================
echo                  TOPFOUR COUNCIL SERVER
echo ============================================================
echo.
echo Starting server at http://127.0.0.1:5001
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

REM Change to script directory so `python topfour_server.py` resolves
REM the llm_council package on sys.path without needing pip install.
cd /d "%~dp0"

REM Run the server (blocks until Ctrl+C). 127.0.0.1 bind only —
REM no auth on this daemon. Tighten if/when the daemon is exposed
REM beyond localhost.
python topfour_server.py

pause
