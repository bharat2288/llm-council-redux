"""Run-directory lifecycle for council fire/resume.

A run directory persists each theorist's output as soon as the future
resolves, so that a council that completes 2/3 theorists before the
operator kills the process (or codex-cli orphan-hangs) still leaves
recoverable material on disk. The `resume` subcommand reads this
directory and re-runs only the chairman synthesis.

Layout
------
{run_dir}/
    run.json                # config snapshot + live state
    theorists/{name}.json   # per-theorist result metadata
    theorists/{name}.md     # raw theorist content (empty on failure)
    synthesis.json          # chairman result metadata
    synthesis.md            # chairman content

The main CLI thread is the only writer for run.json and synthesis.*.
Per-theorist files are written exactly once each, from the thread that
ran fire_theorist for that name — no cross-thread contention.

run.json schema
---------------
{
  "version": 1,
  "started_at": "<iso>",
  "finished_at": "<iso|null>",
  "status": "running" | "partial" | "synthesized" | "failed",
  "output_dir": "<abs>",
  "artifact_path": "<abs|null>",
  "config": { ...resolved config dict... },
  "theorists": {
      "<name>": {"status": "pending|running|ok|fail",
                 "model": "...", "routing": "...", "effort": "...",
                 "duration_seconds": <float|null>,
                 "content_chars": <int|null>,
                 "cost_usd": <float>,
                 "error": <str|null>}
  },
  "synthesis": {"status": "pending|ok|fail|skipped",
                "model": "...", "duration_seconds": ..., ...}
}
"""
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from llm_council.config import Config
from llm_council.routing import TheoristResult
from llm_council.synthesis import SynthesisResult


RUN_STATE_VERSION = 1


def create_run_dir(output_dir: Path, topic_slug: str, started_at: datetime) -> Path:
    """Create the run directory next to where the flat artifact will land
    and return its absolute path. The directory name shares the
    timestamp+slug stem with the flat artifact so they sort together."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{started_at.strftime('%Y-%m-%d-%H%M')}-{topic_slug}"
    run_dir = (output_dir / stem).resolve()
    (run_dir / "theorists").mkdir(parents=True, exist_ok=True)
    return run_dir


def init_run_json(
    run_dir: Path,
    *,
    config: Config,
    output_dir: Path,
    started_at: datetime,
) -> None:
    """Write the initial run.json before any theorist fires."""
    state = {
        "version": RUN_STATE_VERSION,
        "started_at": started_at.isoformat(),
        "finished_at": None,
        "status": "running",
        "output_dir": str(output_dir.resolve()),
        "artifact_path": None,
        "config": _config_dict(config),
        "theorists": {
            t.name: {
                "status": "pending",
                "model": t.model,
                "routing": t.routing,
                "effort": t.effort,
                "duration_seconds": None,
                "content_chars": None,
                "cost_usd": 0.0,
                "error": None,
            }
            for t in config.theorists
        },
        "synthesis": {
            "status": "pending",
            "model": config.synthesizer.model,
            "routing": config.synthesizer.routing,
            "duration_seconds": None,
            "content_chars": None,
            "cost_usd": 0.0,
            "error": None,
        },
    }
    _write_json(run_dir / "run.json", state)


def persist_theorist(run_dir: Path, result: TheoristResult) -> None:
    """Write the per-theorist .md and .json. Called from the worker thread
    that produced `result`; no other writer touches these paths."""
    theorists_dir = run_dir / "theorists"
    theorists_dir.mkdir(parents=True, exist_ok=True)
    (theorists_dir / f"{result.name}.md").write_text(
        result.content or "", encoding="utf-8"
    )
    _write_json(
        theorists_dir / f"{result.name}.json",
        {
            "name": result.name,
            "model": result.model,
            "routing": result.routing,
            "success": result.success,
            "content_chars": len(result.content),
            "cost_usd": result.cost_usd,
            "duration_seconds": result.duration_seconds,
            "error": result.error,
        },
    )


def update_theorist_state(run_dir: Path, result: TheoristResult) -> None:
    """Update run.json's theorist sub-entry. Main thread only."""
    state = _read_json(run_dir / "run.json")
    state["theorists"][result.name] = {
        "status": "ok" if result.success else "fail",
        "model": result.model,
        "routing": result.routing,
        "effort": state["theorists"][result.name].get("effort"),
        "duration_seconds": result.duration_seconds,
        "content_chars": len(result.content),
        "cost_usd": result.cost_usd,
        "error": result.error,
    }
    _write_json(run_dir / "run.json", state)


def mark_theorist_running(run_dir: Path, name: str) -> None:
    """Flip a theorist's status to 'running' at future-submit time so an
    observer scanning run.json can tell which one is in flight."""
    state = _read_json(run_dir / "run.json")
    if name in state["theorists"]:
        state["theorists"][name]["status"] = "running"
        _write_json(run_dir / "run.json", state)


def persist_synthesis(run_dir: Path, synthesis: SynthesisResult) -> None:
    """Write synthesis.md and synthesis.json after chairman completes."""
    (run_dir / "synthesis.md").write_text(
        synthesis.content or "", encoding="utf-8"
    )
    _write_json(
        run_dir / "synthesis.json",
        {
            "success": synthesis.success,
            "content_chars": len(synthesis.content),
            "cost_usd": synthesis.cost_usd,
            "duration_seconds": synthesis.duration_seconds,
            "error": synthesis.error,
        },
    )


def finalize_run_json(
    run_dir: Path,
    *,
    synthesis: SynthesisResult,
    artifact_path: Optional[Path],
    finished_at: datetime,
    status: str,
) -> None:
    """Stamp final state into run.json. `status` is one of:
    'partial' (too few theorists), 'synthesized' (chairman ok),
    'failed' (chairman failed), 'aborted' (manual recovery left flag)."""
    state = _read_json(run_dir / "run.json")
    state["finished_at"] = finished_at.isoformat()
    state["status"] = status
    state["artifact_path"] = str(artifact_path) if artifact_path else None
    state["synthesis"] = {
        "status": "ok" if synthesis.success else (
            "skipped" if synthesis.error and "not enough theorists" in synthesis.error
            else "fail"
        ),
        "model": state["synthesis"].get("model"),
        "routing": state["synthesis"].get("routing"),
        "duration_seconds": synthesis.duration_seconds,
        "content_chars": len(synthesis.content),
        "cost_usd": synthesis.cost_usd,
        "error": synthesis.error,
    }
    _write_json(run_dir / "run.json", state)


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load run.json. Raises FileNotFoundError if the run-dir is missing
    or doesn't look like a council run."""
    rj = run_dir / "run.json"
    if not rj.exists():
        raise FileNotFoundError(
            f"No run.json under {run_dir}. Not a council run directory."
        )
    return _read_json(rj)


def load_persisted_theorists(run_dir: Path) -> list[TheoristResult]:
    """Reconstruct TheoristResult objects from theorists/*.json + .md.

    Used by `resume` to feed saved outputs into a fresh synthesis. Only
    theorists with a present .json are returned; the order matches the
    order in run.json's config (so the artifact and synthesis prompt see
    a stable theorist sequence)."""
    state = load_run(run_dir)
    ordered_names = [t["name"] for t in state["config"]["theorists"]]
    results: list[TheoristResult] = []
    theorists_dir = run_dir / "theorists"
    for name in ordered_names:
        meta_path = theorists_dir / f"{name}.json"
        if not meta_path.exists():
            continue
        meta = _read_json(meta_path)
        content_path = theorists_dir / f"{name}.md"
        content = content_path.read_text(encoding="utf-8") if content_path.exists() else ""
        results.append(
            TheoristResult(
                name=meta["name"],
                model=meta["model"],
                routing=meta["routing"],
                success=meta["success"],
                content=content,
                error=meta.get("error"),
                cost_usd=meta.get("cost_usd", 0.0),
                duration_seconds=meta.get("duration_seconds", 0.0),
            )
        )
    return results


# ---------- helpers ----------


def _config_dict(config: Config) -> dict[str, Any]:
    """Stable JSON-friendly snapshot of the resolved config."""
    return {
        "topic": config.topic,
        "project": config.project,
        "mode": config.mode,
        "theorists": [
            {
                "name": t.name,
                "model": t.model,
                "effort": t.effort,
                "routing": t.routing,
            }
            for t in config.theorists
        ],
        "synthesizer": {
            "model": config.synthesizer.model,
            "effort": config.synthesizer.effort,
            "routing": config.synthesizer.routing,
        },
        "artifact": {
            "status": config.artifact.status,
            "topic_slug": config.artifact.topic_slug,
        },
    }


def _write_json(path: Path, data: Any) -> None:
    if is_dataclass(data):
        data = asdict(data)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
