"""Topfour HTTP daemon — wraps the llm_council engine for the Meshbook
chat-dropdown caller.

Companion to the /council CLI skill at ~/.claude/skills/council/SKILL.md.
Both callers hit the same engine via the same defaults source of truth;
the only difference is transport (subprocess vs HTTP) and streaming
(synchronous vs SSE).

Pipeline reference: meshbook #63 (cross-project work).

Endpoints (slice 1 — defaults endpoint only; later slices add /start,
/stream, DELETE):

  GET  /api/topfour/defaults?mode=<mode>   canonical config JSON for mode
  POST /api/topfour/start                  fire a council, return {run_id}    [later slice]
  GET  /api/topfour/stream/<run_id>        SSE stream of run events           [later slice]
  DELETE /api/topfour/<run_id>             cancel an in-flight run            [later slice]

Operational notes:

  - Listens on 127.0.0.1:5001 (localhost only). Existing council_server.py
    runs on 5000; topfour gets its own port.
  - DSM-managed: launched under `op run --env-file=C:\\launcher\\llm-windows.op.env`
    so OPENROUTER_API_KEY is in process env from server-start onward
    (Pattern A from review-2026-05-07-mesh discussion). 1Password modal
    pops at server start, persists for the daemon's lifetime, no
    per-request modal interrupts.
  - Open localhost (no auth). Single-user local-first; tighten to bearer
    token if/when ever exposed beyond 127.0.0.1.
  - Concurrency: one council run at a time per process; parallel POSTs
    return 503 (single-user reality, registry overhead unjustified).
"""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from flask import Flask, jsonify, request

from llm_council.defaults import default_config_for_mode, known_modes
from llm_council.errors import CouncilError


def create_app() -> Flask:
    """Flask app factory. Test code uses this with `app.test_client()`;
    the production launcher (start_topfour.bat) calls `app.run(...)`.
    Factory pattern keeps tests cheap (no socket bind) and lets us add
    request-scoped state (in-flight registry) in later slices without
    polluting module-level globals."""
    app = Flask(__name__)
    _register_routes(app)
    return app


# -- error envelope --------------------------------------------------------


def _error_response(code: str, message: str, status: HTTPStatus, **details: Any):
    """Consistent error envelope so callers can parse failures uniformly.

    Shape: {"error": {"code": "...", "message": "...", "details": {...}?}}
    Mirrors the operator-shell / chat-surface error shape in Meshbook so
    the chat-dropdown caller can handle topfour errors with the same code.
    """
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body), status


# -- routes ----------------------------------------------------------------


def _register_routes(app: Flask) -> None:

    @app.get("/api/topfour/defaults")
    def get_defaults():
        mode = request.args.get("mode")
        if not mode:
            return _error_response(
                "invalid_request",
                f"mode query param is required; known modes: {list(known_modes())}",
                HTTPStatus.BAD_REQUEST,
            )
        try:
            cfg = default_config_for_mode(mode)
        except CouncilError as exc:
            return _error_response(
                "unknown_mode",
                str(exc),
                HTTPStatus.BAD_REQUEST,
            )
        return jsonify(cfg)


# -- launcher entry --------------------------------------------------------


def main() -> None:
    """Production launcher entry. Invoked by start_topfour.bat under op-run."""
    app = create_app()
    # 127.0.0.1 — local-only bind. Do not change to 0.0.0.0 without
    # adding auth (see module docstring).
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)


if __name__ == "__main__":
    main()
