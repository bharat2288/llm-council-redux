"""CLI for `python -m llm_council`.

Subcommands:
  fire        — run a council from a config JSON file, write artifact
  preflight   — validate a config without firing (dry-run)
  defaults    — print the canonical config JSON for a named mode
  resume      — re-run chairman synthesis from a partial run directory
  synthesize  — alias for resume (explicit naming for chairman-only runs)

The /council skill at ~/.claude/skills/council/ uses `fire` and falls back
to `resume` when a previous run produced theorist outputs but no final
artifact. The HTTP daemon (#63) uses the package's library functions
directly rather than this CLI, but the contract is the same.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from llm_council import run_state
from llm_council.artifact import write_artifact
from llm_council.config import Config, load_config
from llm_council.defaults import default_config_for_mode, known_modes
from llm_council.errors import CouncilError
from llm_council.routing import TheoristResult, fire_theorist
from llm_council.synthesis import SynthesisResult, run_synthesis


HEARTBEAT_INTERVAL_SECONDS = 30


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m llm_council")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fire = sub.add_parser("fire", help="Run a council from a config JSON.")
    p_fire.add_argument(
        "--config", required=True, type=Path,
        help="Path to config JSON (see llm_council.config.Config for schema).",
    )
    p_fire.add_argument(
        "--output-dir", required=True, type=Path,
        help="Directory to write the artifact into. Created if missing.",
    )
    p_fire.add_argument(
        "--theorist-timeout", type=int, default=600,
        help="Per-theorist timeout in seconds (default 600).",
    )

    p_pre = sub.add_parser("preflight", help="Validate config without firing.")
    p_pre.add_argument("--config", required=True, type=Path)

    p_def = sub.add_parser(
        "defaults",
        help=(
            "Print the canonical config JSON for a named mode. Used by both "
            "the /council CLI skill and (via the topfour HTTP daemon) the "
            "Meshbook chat-dropdown caller as the single source of truth."
        ),
    )
    p_def.add_argument(
        "--mode", required=True,
        help=f"Mode name. Known: {', '.join(known_modes())}",
    )

    for name in ("resume", "synthesize"):
        p = sub.add_parser(
            name,
            help=(
                "Re-run chairman synthesis from a run directory that has "
                "saved theorist outputs. Useful when a previous `fire` was "
                "interrupted or codex-cli timed out but 2+ other theorists "
                "succeeded."
            ),
        )
        p.add_argument(
            "--run-dir", required=True, type=Path,
            help="Path to the run directory (contains run.json + theorists/).",
        )
        p.add_argument(
            "--theorist-timeout", type=int, default=600,
            help="Timeout for the chairman call (default 600).",
        )

    args = parser.parse_args(argv)

    if args.cmd == "preflight":
        return _cmd_preflight(args.config)
    if args.cmd == "fire":
        return _cmd_fire(args.config, args.output_dir, args.theorist_timeout)
    if args.cmd == "defaults":
        return _cmd_defaults(args.mode)
    if args.cmd in ("resume", "synthesize"):
        return _cmd_resume(args.run_dir, args.theorist_timeout)
    parser.print_help()
    return 2


def _cmd_defaults(mode: str) -> int:
    """Print canonical config JSON for a named mode."""
    import json
    try:
        cfg = default_config_for_mode(mode)
    except CouncilError as exc:
        print(f"FAIL defaults: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(cfg, indent=2))
    return 0


def _cmd_preflight(config_path: Path) -> int:
    try:
        config = load_config(config_path)
    except CouncilError as exc:
        print(f"FAIL preflight: {exc}", file=sys.stderr)
        return 1
    print("PREFLIGHT OK")
    print(f"  topic:       {config.topic}")
    print(f"  project:     {config.project}")
    print(f"  mode:        {config.mode}")
    print(f"  theorists:")
    for t in config.theorists:
        print(f"    - {t.name:8} model={t.model:36} effort={t.effort} routing={t.routing}")
    print(f"  synthesizer: {config.synthesizer.model} (effort={config.synthesizer.effort}, routing={config.synthesizer.routing})")
    print(f"  artifact:    status={config.artifact.status} slug={config.artifact.topic_slug}")
    return 0


def _cmd_fire(config_path: Path, output_dir: Path, theorist_timeout: int) -> int:
    try:
        config = load_config(config_path)
    except CouncilError as exc:
        print(f"FAIL config: {exc}", file=sys.stderr)
        return 1

    started_at = datetime.now(timezone.utc)

    # Create the run directory and write run.json BEFORE firing anything.
    # This is the durability anchor: if the process dies after this point
    # but before the flat artifact is written, the operator can still see
    # what was attempted and recover partial work via `resume`.
    run_dir = run_state.create_run_dir(
        output_dir=output_dir,
        topic_slug=config.artifact.topic_slug,
        started_at=started_at,
    )
    run_state.init_run_json(
        run_dir, config=config, output_dir=output_dir, started_at=started_at
    )

    print("=" * 78, flush=True)
    print(f"COUNCIL: {config.topic}", flush=True)
    print(f"  project:  {config.project}", flush=True)
    print(f"  mode:     {config.mode}", flush=True)
    for t in config.theorists:
        print(f"  - {t.name:8} {t.model:36} effort={t.effort:6} routing={t.routing}", flush=True)
    print(f"  chairman: {config.synthesizer.model} (routing={config.synthesizer.routing})", flush=True)
    print(f"  out:      {output_dir}", flush=True)
    print(f"  run-dir:  {run_dir}", flush=True)
    print(f"  resume:   python -m llm_council resume --run-dir \"{run_dir}\"", flush=True)
    print("=" * 78, flush=True)

    # Phase 1: theorists in parallel, with persistence + heartbeat.
    theorist_results: list[TheoristResult] = [None] * len(config.theorists)  # type: ignore
    in_flight: dict[str, float] = {}
    in_flight_lock = threading.Lock()
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        """Print '[theorist WORKING name t=Xs]' every HEARTBEAT_INTERVAL_SECONDS
        for any theorist still in flight. Lets the operator distinguish 'long
        pole codex still reasoning' from 'engine hung'."""
        while not stop_heartbeat.wait(HEARTBEAT_INTERVAL_SECONDS):
            with in_flight_lock:
                now = time.monotonic()
                still = sorted(in_flight.items(), key=lambda kv: kv[1])
            for name, t_start in still:
                elapsed = time.monotonic() - t_start
                print(
                    f"[theorist WORKING] {name:8} t={elapsed:5.0f}s",
                    flush=True,
                )

    hb_thread = threading.Thread(target=heartbeat, daemon=True, name="hb")
    hb_thread.start()

    try:
        with ThreadPoolExecutor(max_workers=len(config.theorists)) as pool:
            futures = {}
            for idx, t in enumerate(config.theorists):
                run_state.mark_theorist_running(run_dir, t.name)
                with in_flight_lock:
                    in_flight[t.name] = time.monotonic()
                print(f"[theorist START]   {t.name:8} {t.model}", flush=True)
                fut = pool.submit(
                    fire_theorist,
                    t,
                    config.topic,
                    theorist_timeout,
                    config.include_dirs,
                )
                futures[fut] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                r = fut.result()
                theorist_results[idx] = r
                with in_flight_lock:
                    in_flight.pop(r.name, None)
                # Persist the result immediately. If the process dies
                # before chairman finishes, this file is the durable trace.
                run_state.persist_theorist(run_dir, r)
                run_state.update_theorist_state(run_dir, r)
                mark = "OK  " if r.success else "FAIL"
                print(
                    f"[theorist {mark}] {r.name:8} {len(r.content):>6}ch  "
                    f"{r.duration_seconds:5.1f}s  ${r.cost_usd:.4f}"
                    + (f"  err={r.error}" if r.error else ""),
                    flush=True,
                )
    finally:
        stop_heartbeat.set()

    successful = sum(1 for r in theorist_results if r.success)
    print(f"\n[phase 1] {successful}/{len(theorist_results)} theorists succeeded", flush=True)

    if successful < 2:
        finished_at = datetime.now(timezone.utc)
        synthesis = _skipped_synthesis("not enough theorists")
        path = write_artifact(
            config=config,
            theorist_results=theorist_results,
            synthesis=synthesis,
            output_dir=output_dir,
            started_at=started_at,
            finished_at=finished_at,
        )
        run_state.persist_synthesis(run_dir, synthesis)
        run_state.finalize_run_json(
            run_dir,
            synthesis=synthesis,
            artifact_path=path,
            finished_at=finished_at,
            status="partial",
        )
        print(f"\n[abort] artifact written for record: {path}", flush=True)
        print(f"[abort] partial run dir: {run_dir}", flush=True)
        return 3

    # Phase 2: chairman synthesis.
    print(f"\n[phase 2] chairman synthesizing from {successful} perspectives...", flush=True)
    synthesis = run_synthesis(
        spec=config.synthesizer,
        topic=config.topic,
        theorist_results=theorist_results,
        timeout_seconds=theorist_timeout,
        include_dirs=config.include_dirs,
    )
    if synthesis.success:
        print(
            f"[chairman done] {len(synthesis.content)}ch  "
            f"{synthesis.duration_seconds:.1f}s  ${synthesis.cost_usd:.4f}",
            flush=True,
        )
    else:
        print(f"[chairman FAIL] {synthesis.error}", flush=True)
    run_state.persist_synthesis(run_dir, synthesis)

    finished_at = datetime.now(timezone.utc)
    path = write_artifact(
        config=config,
        theorist_results=theorist_results,
        synthesis=synthesis,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
    )
    run_state.finalize_run_json(
        run_dir,
        synthesis=synthesis,
        artifact_path=path,
        finished_at=finished_at,
        status="synthesized" if synthesis.success else "failed",
    )
    total_cost = sum(r.cost_usd for r in theorist_results) + synthesis.cost_usd
    duration = (finished_at - started_at).total_seconds()
    print(
        f"\n[done] artifact={path}  duration={duration:.1f}s  cost=${total_cost:.4f}",
        flush=True,
    )
    print(f"[done] run-dir={run_dir}", flush=True)
    return 0 if synthesis.success else 4


def _cmd_resume(run_dir: Path, theorist_timeout: int) -> int:
    """Re-run chairman synthesis from saved theorist outputs.

    Discovers theorists/*.json under run_dir, reconstructs TheoristResult
    objects, and fires the chairman. Writes a fresh flat artifact next to
    the run-dir (sibling timestamped .md/.json) and updates run.json's
    status. The original flat artifact, if any, is left untouched.
    """
    run_dir = run_dir.resolve()
    try:
        state = run_state.load_run(run_dir)
    except FileNotFoundError as exc:
        print(f"FAIL resume: {exc}", file=sys.stderr)
        return 1

    # Rebuild config from the saved run.json snapshot. We don't re-parse
    # the original config JSON because the operator may have moved or
    # edited it, and run.json captures the resolved config that actually
    # fired.
    try:
        config = _config_from_state(state)
    except CouncilError as exc:
        print(f"FAIL resume: cannot rebuild config: {exc}", file=sys.stderr)
        return 1

    theorist_results = run_state.load_persisted_theorists(run_dir)
    successful = [r for r in theorist_results if r.success]

    print("=" * 78, flush=True)
    print(f"COUNCIL RESUME: {config.topic[:64]}", flush=True)
    print(f"  run-dir:  {run_dir}", flush=True)
    print(f"  loaded:   {len(theorist_results)} theorist(s), "
          f"{len(successful)} successful", flush=True)
    for r in theorist_results:
        mark = "OK  " if r.success else "FAIL"
        print(
            f"  - [{mark}] {r.name:8} {len(r.content):>6}ch  "
            f"{r.duration_seconds:5.1f}s",
            flush=True,
        )
    print("=" * 78, flush=True)

    if len(successful) < 2:
        print(
            f"\n[abort] only {len(successful)} successful theorist(s) on disk; "
            f"chairman needs at least 2 to resolve tensions.",
            file=sys.stderr, flush=True,
        )
        return 3

    output_dir = Path(state["output_dir"])
    started_at_iso = state["started_at"]
    started_at = datetime.fromisoformat(started_at_iso)

    print(f"[phase 2] chairman synthesizing from {len(successful)} perspectives...", flush=True)
    synthesis = run_synthesis(
        spec=config.synthesizer,
        topic=config.topic,
        theorist_results=theorist_results,
        timeout_seconds=theorist_timeout,
        include_dirs=config.include_dirs,
    )
    if synthesis.success:
        print(
            f"[chairman done] {len(synthesis.content)}ch  "
            f"{synthesis.duration_seconds:.1f}s  ${synthesis.cost_usd:.4f}",
            flush=True,
        )
    else:
        print(f"[chairman FAIL] {synthesis.error}", flush=True)
    run_state.persist_synthesis(run_dir, synthesis)

    finished_at = datetime.now(timezone.utc)
    # Write a fresh artifact stamped with the resume time so it doesn't
    # collide with any earlier flat artifact from the original fire.
    artifact_started = finished_at
    path = write_artifact(
        config=config,
        theorist_results=theorist_results,
        synthesis=synthesis,
        output_dir=output_dir,
        started_at=artifact_started,
        finished_at=finished_at,
    )
    run_state.finalize_run_json(
        run_dir,
        synthesis=synthesis,
        artifact_path=path,
        finished_at=finished_at,
        status="synthesized" if synthesis.success else "failed",
    )
    print(f"\n[done] artifact={path}", flush=True)
    print(f"[done] run-dir={run_dir}", flush=True)
    return 0 if synthesis.success else 4


def _config_from_state(state: dict) -> Config:
    """Rebuild a Config from a saved run.json. Goes through parse_config
    so we reuse the same validation that fire uses."""
    from llm_council.config import parse_config
    raw = dict(state["config"])
    # parse_config requires the same shape as a config file
    return parse_config(raw)


def _skipped_synthesis(reason: str) -> SynthesisResult:
    return SynthesisResult(success=False, content="", error=reason)


if __name__ == "__main__":
    sys.exit(main())
