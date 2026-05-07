"""Topfour HTTP daemon — wraps the llm_council engine for the Meshbook
chat-dropdown caller.

Companion to the /council CLI skill at ~/.claude/skills/council/SKILL.md.
Both callers hit the same engine via the same defaults source of truth;
the only difference is transport (subprocess vs HTTP) and streaming
(synchronous vs SSE).

Pipeline reference: meshbook #63 (cross-project work).

Endpoints:

  GET  /api/topfour/defaults?mode=<mode>   canonical config JSON for mode
  POST /api/topfour/start                  fire a council, return {run_id, artifact_path}
  GET  /api/topfour/stream/<run_id>        SSE stream of run events           [later slice]
  DELETE /api/topfour/<run_id>             cancel an in-flight run            [later slice]

Operational notes:

  - Listens on 127.0.0.1:5001 (localhost only). Existing council_server.py
    runs on 5000; topfour gets its own port.
  - DSM-managed: launched under `op run --env-file=C:\\launcher\\llm-windows.op.env`
    so OPENROUTER_API_KEY is in process env from server-start onward
    (Pattern A from the chat-surface ADR Open Item discussion). 1Password
    modal pops at server start, persists for the daemon's lifetime, no
    per-request modal interrupts. Required for `standard-paid` mode (Grok
    via OpenRouter); other modes don't read OPENROUTER_API_KEY.
  - Open localhost (no auth). Single-user local-first; tighten to bearer
    token if/when ever exposed beyond 127.0.0.1.
  - Concurrency: one council run at a time per process; parallel POSTs
    return 503 (single-user reality, registry overhead unjustified).
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

from llm_council.artifact import write_artifact
from llm_council.config import parse_config
from llm_council.defaults import default_config_for_mode, known_modes
from llm_council.errors import CouncilError
from llm_council.routing import TheoristResult, fire_theorist
from llm_council.synthesis import run_synthesis


DEFAULT_PROJECT_ROOT = r"C:\Users\bhara\dev"


def create_app(*, project_root: str | None = None) -> Flask:
    """Flask app factory. Test code uses this with `app.test_client()`;
    the production launcher (start_topfour.bat) calls `app.run(...)`.

    `project_root` is the dev root (the directory containing per-project
    folders). Artifacts land at:

        <project_root>/<project>/specs/research/runs/<YYYY-MM-DD-HHMM>-<slug>.md

    Defaults to `C:\\Users\\bhara\\dev` so the path resolves through each
    project's `specs/` junction to the canonical centralized location at
    `dev/specs/<project>/research/runs/` (per artifacts.md). The skill
    CLI uses the same junction-form path; daemon stays consistent so
    both callers produce identical filesystem layouts.

    Tests override project_root to a tmp_path so artifacts don't pollute
    the real specs tree.
    """
    app = Flask(__name__)
    app.config["TOPFOUR_PROJECT_ROOT"] = project_root or DEFAULT_PROJECT_ROOT
    # In-flight registry: chat_id-style mapping for one-run-at-a-time.
    # Slice 4 will expand this to event queues for SSE; slice 5 wires up
    # cancellation. For now it's just a busy flag.
    app.config["_TOPFOUR_INFLIGHT_LOCK"] = threading.Lock()
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


# -- request body parsing --------------------------------------------------


def _build_config_dict_from_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the HTTP request body into the engine's config schema.

    Accepts two shapes:
      - mode-driven: {topic, project, mode, output_dir?}
        → expand defaults from llm_council.defaults, merge topic + project
      - explicit: {topic, project, theorists, synthesizer, output_dir?}
        → use the caller's exact theorist list

    Caller may also combine: pass `mode` AND override individual fields
    (e.g. `theorists` to use a custom roster but still tag the run with a
    mode for telemetry). When both are present, explicit fields win.
    """
    topic = payload.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        raise CouncilError("topic is required and must be a non-empty string")

    project = payload.get("project")
    if not isinstance(project, str) or not project.strip():
        raise CouncilError("project is required and must be a non-empty string")

    mode = payload.get("mode")
    explicit_theorists = payload.get("theorists")
    explicit_synth = payload.get("synthesizer")

    if mode is None and explicit_theorists is None:
        raise CouncilError(
            "request must include either 'mode' (use defaults) or 'theorists' "
            "(explicit list); received neither"
        )

    if mode is not None:
        # Pull defaults; explicit fields override.
        cfg = default_config_for_mode(mode)
    else:
        cfg = {"mode": "custom"}

    if explicit_theorists is not None:
        cfg["theorists"] = explicit_theorists
    if explicit_synth is not None:
        cfg["synthesizer"] = explicit_synth

    cfg["topic"] = topic
    cfg["project"] = project
    # Preserve artifact metadata if caller provided it.
    if "artifact" in payload:
        cfg["artifact"] = payload["artifact"]
    else:
        cfg.setdefault("artifact", {"status": "draft"})
    return cfg


def _resolve_output_dir(app: Flask, payload: dict[str, Any], project: str) -> Path:
    """Where to write the artifact.

    Caller-supplied `output_dir` wins when present (used for tests and
    edge cases). Otherwise default to
    `<project_root>/<project>/research/runs/`.
    """
    explicit = payload.get("output_dir")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit)
    project_root = Path(app.config["TOPFOUR_PROJECT_ROOT"])
    # Junction-form path: <root>/<project>/specs/research/runs/.
    # Matches the skill CLI's convention. On Windows, <project>/specs/
    # is a reparse point that resolves to dev/specs/<project>/, so this
    # write lands at the canonical centralized location.
    return project_root / project / "specs" / "research" / "runs"


# -- engine fire (synchronous in slice 3; refactored to async in slice 4) --


def _fire_council_sync(
    *,
    cfg_dict: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Run the council to completion and return a response dict.

    Mirrors cli._cmd_fire's flow but in-process (no subprocess). Slice 4
    will refactor this into an async generator that emits events for SSE
    consumption; the synchronous version stays as the /start fallback.
    """
    config = parse_config(cfg_dict)

    started_at = datetime.now(timezone.utc)

    # Phase 1 — theorists in parallel via ThreadPoolExecutor.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    theorist_results: list[TheoristResult] = [None] * len(config.theorists)  # type: ignore
    with ThreadPoolExecutor(max_workers=len(config.theorists)) as pool:
        futures = {
            pool.submit(fire_theorist, t, config.topic, 600): idx
            for idx, t in enumerate(config.theorists)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            theorist_results[idx] = fut.result()

    # Phase 2 — chairman synthesis (only if ≥2 theorists succeeded).
    successful = sum(1 for r in theorist_results if r.success)
    if successful >= 2:
        synthesis = run_synthesis(
            spec=config.synthesizer,
            topic=config.topic,
            theorist_results=theorist_results,
            timeout_seconds=600,
        )
    else:
        from llm_council.synthesis import SynthesisResult
        synthesis = SynthesisResult(
            success=False,
            content="",
            error=f"only {successful} theorist(s) succeeded; need ≥2",
        )

    finished_at = datetime.now(timezone.utc)
    artifact_path = write_artifact(
        config=config,
        theorist_results=theorist_results,
        synthesis=synthesis,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
    )

    total_cost = sum(r.cost_usd for r in theorist_results) + synthesis.cost_usd
    duration = (finished_at - started_at).total_seconds()

    return {
        "run_id": str(uuid.uuid4()),
        "artifact_path": str(artifact_path),
        "total_cost_usd": total_cost,
        "duration_seconds": duration,
        "theorists_succeeded": successful,
        "theorists_total": len(theorist_results),
        "synthesis_succeeded": synthesis.success,
    }


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

    @app.post("/api/topfour/start")
    def post_start():
        # One run at a time per process. Tighter than need (single-user
        # reality) but cheap to implement and protects the in-flight
        # registry which slice 5 will expand.
        lock: threading.Lock = app.config["_TOPFOUR_INFLIGHT_LOCK"]
        if not lock.acquire(blocking=False):
            return _error_response(
                "busy",
                "a council run is already in flight; wait for it to complete or DELETE its run_id",
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                return _error_response(
                    "invalid_request",
                    "request body must be a JSON object",
                    HTTPStatus.BAD_REQUEST,
                )

            try:
                cfg_dict = _build_config_dict_from_request(payload)
            except CouncilError as exc:
                code = "unknown_mode" if "unknown mode" in str(exc) else "invalid_request"
                return _error_response(code, str(exc), HTTPStatus.BAD_REQUEST)

            project = cfg_dict["project"]
            output_dir = _resolve_output_dir(app, payload, project)

            try:
                result = _fire_council_sync(cfg_dict=cfg_dict, output_dir=output_dir)
            except CouncilError as exc:
                return _error_response(
                    "engine_error", str(exc), HTTPStatus.INTERNAL_SERVER_ERROR
                )
            return jsonify(result)
        finally:
            lock.release()


# -- launcher entry --------------------------------------------------------


def main() -> None:
    """Production launcher entry. Invoked by start_topfour.bat under op-run."""
    app = create_app()
    # 127.0.0.1 — local-only bind. Do not change to 0.0.0.0 without
    # adding auth (see module docstring).
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)


if __name__ == "__main__":
    main()
