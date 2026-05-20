@echo off
REM council-fire.cmd - wrapper for /council paid invocations.
REM
REM Usage:
REM   council-fire.cmd --config <path-to-config.json> --output-dir <dir> [--theorist-timeout 300]
REM
REM The paid preset needs OPENROUTER_API_KEY in the child environment. This
REM wrapper deliberately does not store private launcher paths in the workspace.
REM Set LLM_COUNCIL_OP_ENV_FILE in a private shell before invoking it, or run
REM python -m llm_council fire directly from a shell where OPENROUTER_API_KEY
REM is already injected.

setlocal
cd /d "%~dp0"

if "%LLM_COUNCIL_OP_ENV_FILE%"=="" (
  echo FAIL: LLM_COUNCIL_OP_ENV_FILE is not set.
  echo Set it in a private shell, or run from a shell where OPENROUTER_API_KEY is already injected.
  exit /b 2
)

op run --env-file="%LLM_COUNCIL_OP_ENV_FILE%" -- python -m llm_council fire %*
