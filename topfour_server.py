"""Topfour HTTP daemon — wraps the llm_council engine for the Meshbook
chat-dropdown caller.

Companion to the /council CLI skill at ~/.claude/skills/council/SKILL.md.
Both callers hit the same engine via the same defaults source of truth;
the only difference is transport (subprocess vs HTTP) and streaming
(synchronous CLI vs SSE).

Pipeline reference: meshbook #63 (cross-project work).

Endpoints:

  GET  /api/topfour/defaults?mode=<mode>   canonical config JSON for mode
  POST /api/topfour/start                  kick off a council, return {run_id} immediately
  GET  /api/topfour/stream/<run_id>        SSE stream of run events
  DELETE /api/topfour/<run_id>             cancel an in-flight run            [later slice]

Lifecycle:

  POST /start parses + validates config, builds a `CouncilRun`, kicks off
  a worker thread, returns `{run_id, status: "started"}` IMMEDIATELY
  (before theorists complete). The worker thread emits events to the run's
  queue as theorists finish; `GET /stream/<run_id>` reads from the queue
  and yields SSE-formatted lines until the `done` event flushes.

Coarse SSE event taxonomy (fine-grained streaming is v0.2):
  - theorist_started  {name, model, routing}
  - theorist_done     {name, success, content_chars, cost_usd, duration_seconds}
  - chairman_started  {model}
  - chairman_done     {success, content_chars, cost_usd, duration_seconds}
  - synthesis_skipped {reason}
  - error             {code, message, theorist?}
  - done              {artifact_path, total_cost_usd, duration_seconds, ...}

Operational notes:
  - 127.0.0.1:5001 (localhost only). council_server.py is on 5000.
  - DSM-managed under op-run for OPENROUTER_API_KEY (chat-surface ADR Pattern A).
  - One council run at a time per process; parallel POSTs return 503.
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any, Iterable, Optional

from flask import Flask, Response, jsonify, request, stream_with_context

from llm_council.artifact import write_artifact
from llm_council.config import Config, parse_config
from llm_council.defaults import default_config_for_mode, known_modes
from llm_council.errors import CouncilError
from llm_council.routing import TheoristResult, fire_theorist
from llm_council.synthesis import SynthesisResult, run_synthesis


DEFAULT_PROJECT_ROOT = r"C:\Users\bhara\dev"


# -- run state -------------------------------------------------------------


# Sentinel pushed onto the queue when the worker thread is done emitting.
# Stream consumer treats this as "no more events" and closes the response.
_END_SENTINEL = object()


@dataclass
class CouncilRun:
    """One in-flight (or recently-completed) council run.

    Owned by the in-flight registry on the Flask app. The worker thread
    pushes events onto `events`; the /stream consumer pulls from it.

    Cancellation: DELETE /api/topfour/<run_id> sets `cancelled = True`.
    The worker checks at safe points and skips remaining phases when set.
    True subprocess termination (SIGKILL on in-flight CLI calls) is v0.2;
    today the worker waits for current-phase theorists to return, then
    honors the flag at the next checkpoint.
    """

    run_id: str
    config: Config
    output_dir: Path
    started_at: datetime
    events: "queue.Queue[Any]" = field(default_factory=queue.Queue)
    finished_at: Optional[datetime] = None
    artifact_path: Optional[Path] = None
    error: Optional[str] = None
    cancelled: bool = False


def create_app(*, project_root: str | None = None) -> Flask:
    """Flask app factory. See module docstring for path conventions."""
    app = Flask(__name__)
    app.config["TOPFOUR_PROJECT_ROOT"] = project_root or DEFAULT_PROJECT_ROOT
    # In-flight registry — one run at a time, but registry is dict-shaped
    # so /stream and /<id> DELETE can look up by run_id even after the
    # worker thread completes (we keep the run record around briefly so
    # late-arriving stream subscribers can still read the buffered events).
    app.config["_TOPFOUR_RUNS"] = {}  # type: dict[str, CouncilRun]
    app.config["_TOPFOUR_INFLIGHT_LOCK"] = threading.Lock()
    _register_routes(app)
    return app


# -- error envelope --------------------------------------------------------


def _error_response(code: str, message: str, status: HTTPStatus, **details: Any):
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body), status


# -- request body parsing --------------------------------------------------


def _build_config_dict_from_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the HTTP request body into the engine's config schema."""
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
        cfg = default_config_for_mode(mode)
    else:
        cfg = {"mode": "custom"}

    if explicit_theorists is not None:
        cfg["theorists"] = explicit_theorists
    if explicit_synth is not None:
        cfg["synthesizer"] = explicit_synth

    cfg["topic"] = topic
    cfg["project"] = project
    if "artifact" in payload:
        cfg["artifact"] = payload["artifact"]
    else:
        cfg.setdefault("artifact", {"status": "draft"})
    return cfg


def _resolve_output_dir(app: Flask, payload: dict[str, Any], project: str) -> Path:
    explicit = payload.get("output_dir")
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit)
    project_root = Path(app.config["TOPFOUR_PROJECT_ROOT"])
    # Junction-form path: <root>/<project>/specs/research/runs/.
    # Matches the skill CLI's convention. On Windows, <project>/specs/
    # is a reparse point that resolves to dev/specs/<project>/, so this
    # write lands at the canonical centralized location.
    return project_root / project / "specs" / "research" / "runs"


# -- worker thread (emits events as the council progresses) ----------------


def _push(run: CouncilRun, event: str, data: dict) -> None:
    """Queue one SSE event for stream consumers."""
    run.events.put({"event": event, "data": data})


def _run_council_worker(run: CouncilRun) -> None:
    """Worker thread: fire theorists in parallel, then chairman, then write
    artifact. Emits events to the run's queue as work progresses.

    Runs to completion regardless of partial failures; per-theorist errors
    surface as `error` events with continue-on. <2 successful theorists
    emit `synthesis_skipped` and skip the chairman phase.
    """
    config = run.config
    try:
        # Phase 1 — theorists in parallel via ThreadPoolExecutor.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        theorist_results: list[Optional[TheoristResult]] = [None] * len(config.theorists)

        # Emit theorist_started for each before submitting so the consumer
        # sees them in the natural "starting up" order rather than tied to
        # whatever finishes first.
        for idx, t in enumerate(config.theorists):
            _push(run, "theorist_started", {
                "name": t.name, "model": t.model, "routing": t.routing,
            })

        with ThreadPoolExecutor(max_workers=len(config.theorists)) as pool:
            futures = {
                pool.submit(fire_theorist, t, config.topic, 600): idx
                for idx, t in enumerate(config.theorists)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                r = fut.result()
                theorist_results[idx] = r
                _push(run, "theorist_done", {
                    "name": r.name,
                    "success": r.success,
                    "content_chars": len(r.content),
                    "cost_usd": r.cost_usd,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                })
                if not r.success:
                    _push(run, "error", {
                        "code": "theorist_failed",
                        "theorist": r.name,
                        "message": r.error or "theorist call failed",
                    })

        # Cancellation checkpoint: between theorist phase and chairman.
        # If the operator hit DELETE while theorists were running, skip
        # synthesis here. Theorist responses already wrote to artifacts
        # (still useful for record), but chairman doesn't fire so no
        # cost is incurred for synthesis on a cancelled run.
        if run.cancelled:
            synthesis = SynthesisResult(
                success=False, content="",
                error="cancelled by operator before chairman synthesis",
            )
            _push(run, "cancelled", {
                "phase": "before_chairman",
                "successful_theorists": sum(1 for r in theorist_results if r and r.success),
            })
        else:
            # Phase 2 — chairman synthesis (only if ≥2 theorists succeeded).
            successful = [r for r in theorist_results if r and r.success]
            if len(successful) < 2:
                synthesis = SynthesisResult(
                    success=False, content="",
                    error=f"only {len(successful)} theorist(s) succeeded; need ≥2",
                )
                _push(run, "synthesis_skipped", {
                    "reason": synthesis.error,
                    "successful_count": len(successful),
                })
            else:
                _push(run, "chairman_started", {"model": config.synthesizer.model})
                synthesis = run_synthesis(
                    spec=config.synthesizer,
                    topic=config.topic,
                    theorist_results=[r for r in theorist_results if r is not None],
                    timeout_seconds=600,
                )
                _push(run, "chairman_done", {
                    "success": synthesis.success,
                    "content_chars": len(synthesis.content),
                    "cost_usd": synthesis.cost_usd,
                    "duration_seconds": synthesis.duration_seconds,
                    "error": synthesis.error,
                })

        # Phase 3 — write artifact (whether cancelled or not, for the record).
        run.finished_at = datetime.now(timezone.utc)
        artifact_path = write_artifact(
            config=config,
            theorist_results=[r for r in theorist_results if r is not None],
            synthesis=synthesis,
            output_dir=run.output_dir,
            started_at=run.started_at,
            finished_at=run.finished_at,
        )
        run.artifact_path = artifact_path

        successful_count = sum(1 for r in theorist_results if r and r.success)
        total_cost = sum(
            (r.cost_usd if r else 0.0) for r in theorist_results
        ) + synthesis.cost_usd
        duration = (run.finished_at - run.started_at).total_seconds()

        _push(run, "done", {
            "artifact_path": str(artifact_path),
            "total_cost_usd": total_cost,
            "duration_seconds": duration,
            "theorists_succeeded": successful_count,
            "theorists_total": len(theorist_results),
            "synthesis_succeeded": synthesis.success,
            "cancelled": run.cancelled,
        })

    except Exception as exc:  # noqa: BLE001 — emit any unhandled failure as a stream event
        run.error = str(exc)
        _push(run, "error", {
            "code": "engine_error",
            "message": str(exc),
        })
    finally:
        # Always signal end-of-stream so consumers don't block forever.
        run.events.put(_END_SENTINEL)


# -- SSE serialization -----------------------------------------------------


def _format_sse(event: str, data: dict) -> bytes:
    """Format one event for the wire. Standard SSE: `event:\\ndata:\\n\\n`."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _stream_events(run: CouncilRun) -> Iterable[bytes]:
    """Pull events from the run's queue and yield SSE bytes until end."""
    while True:
        item = run.events.get()
        if item is _END_SENTINEL:
            return
        yield _format_sse(item["event"], item["data"])


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
            return _error_response("unknown_mode", str(exc), HTTPStatus.BAD_REQUEST)
        return jsonify(cfg)

    @app.post("/api/topfour/start")
    def post_start():
        # One run at a time per process. The lock guards the in-flight slot;
        # it stays held while the worker thread runs and is released when
        # the worker finishes (regardless of success). Stream consumers
        # don't hold the lock — they read from the run's queue.
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
                config = parse_config(cfg_dict)
            except CouncilError as exc:
                code = "unknown_mode" if "unknown mode" in str(exc) else "invalid_request"
                return _error_response(code, str(exc), HTTPStatus.BAD_REQUEST)

            output_dir = _resolve_output_dir(app, payload, config.project)
            run_id = str(uuid.uuid4())
            run = CouncilRun(
                run_id=run_id,
                config=config,
                output_dir=output_dir,
                started_at=datetime.now(timezone.utc),
            )
            app.config["_TOPFOUR_RUNS"][run_id] = run

            # Worker thread releases the in-flight lock when it finishes.
            def _worker():
                try:
                    _run_council_worker(run)
                finally:
                    lock.release()

            threading.Thread(target=_worker, daemon=True).start()

            return jsonify({"run_id": run_id, "status": "started"})
        except Exception:
            # If we hit an exception BEFORE the worker thread starts, we
            # need to release the lock ourselves; otherwise the next POST
            # gets a permanent 503.
            lock.release()
            raise

    @app.get("/api/topfour/stream/<run_id>")
    def get_stream(run_id: str):
        run = app.config["_TOPFOUR_RUNS"].get(run_id)
        if run is None:
            return _error_response(
                "unknown_run",
                f"no run with id {run_id!r}; either it expired or was never started",
                HTTPStatus.NOT_FOUND,
            )
        return Response(
            stream_with_context(_stream_events(run)),
            mimetype="text/event-stream",
        )

    @app.delete("/api/topfour/<run_id>")
    def delete_run(run_id: str):
        run = app.config["_TOPFOUR_RUNS"].get(run_id)
        if run is None:
            return _error_response(
                "unknown_run",
                f"no run with id {run_id!r}",
                HTTPStatus.NOT_FOUND,
            )
        # Set the cancelled flag. The worker thread checks it at safe
        # points (currently: between theorist phase and chairman) and
        # skips remaining work. In-flight theorist subprocesses are NOT
        # killed today (v0.2 work — needs fire_theorist to expose Popen
        # handles); the worker waits for them to return naturally, then
        # honors the flag. Idempotent: DELETE on an already-cancelled or
        # already-done run still returns 200.
        run.cancelled = True
        return jsonify({"run_id": run_id, "cancelled": True})


# -- launcher entry --------------------------------------------------------


def main() -> None:
    """Production launcher entry. Invoked by start_topfour.bat under op-run."""
    app = create_app()
    # 127.0.0.1 — local-only bind. Do not change to 0.0.0.0 without
    # adding auth (see module docstring).
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)


if __name__ == "__main__":
    main()
