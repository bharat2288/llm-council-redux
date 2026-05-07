"""CLI for `python -m llm_council`.

Subcommands:
  fire     — run a council from a config JSON file, write artifact
  preflight — validate a config without firing (dry-run)

The /council skill at ~/.claude/skills/council/ uses `fire`. The HTTP
daemon (#63) will use the package's library functions directly rather
than this CLI, but the contract is the same.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from llm_council.artifact import write_artifact
from llm_council.config import Config, load_config
from llm_council.defaults import default_config_for_mode, known_modes
from llm_council.errors import CouncilError
from llm_council.routing import TheoristResult, fire_theorist
from llm_council.synthesis import run_synthesis


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

    args = parser.parse_args(argv)

    if args.cmd == "preflight":
        return _cmd_preflight(args.config)
    if args.cmd == "fire":
        return _cmd_fire(args.config, args.output_dir, args.theorist_timeout)
    if args.cmd == "defaults":
        return _cmd_defaults(args.mode)
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
    print("=" * 78, flush=True)
    print(f"COUNCIL: {config.topic}", flush=True)
    print(f"  project: {config.project}", flush=True)
    print(f"  mode:    {config.mode}", flush=True)
    for t in config.theorists:
        print(f"  - {t.name:8} {t.model:36} effort={t.effort:6} routing={t.routing}", flush=True)
    print(f"  chairman: {config.synthesizer.model} (routing={config.synthesizer.routing})", flush=True)
    print(f"  out:     {output_dir}", flush=True)
    print("=" * 78, flush=True)

    # Phase 1: theorists in parallel.
    theorist_results: list[TheoristResult] = [None] * len(config.theorists)  # type: ignore
    with ThreadPoolExecutor(max_workers=len(config.theorists)) as pool:
        futures = {
            pool.submit(fire_theorist, t, config.topic, theorist_timeout): idx
            for idx, t in enumerate(config.theorists)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            r = fut.result()
            theorist_results[idx] = r
            mark = "OK  " if r.success else "FAIL"
            print(
                f"[theorist {mark}] {r.name:8} {len(r.content):>6}ch  "
                f"{r.duration_seconds:5.1f}s  ${r.cost_usd:.4f}"
                + (f"  err={r.error}" if r.error else ""),
                flush=True,
            )

    successful = sum(1 for r in theorist_results if r.success)
    print(f"\n[phase 1] {successful}/{len(theorist_results)} theorists succeeded", flush=True)
    if successful < 2:
        finished_at = datetime.now(timezone.utc)
        # Still write the artifact so the operator has the failure record.
        path = write_artifact(
            config=config,
            theorist_results=theorist_results,
            synthesis=_skipped_synthesis("not enough theorists"),
            output_dir=output_dir,
            started_at=started_at,
            finished_at=finished_at,
        )
        print(f"\n[abort] artifact written for record: {path}", flush=True)
        return 3

    # Phase 2: chairman synthesis.
    print(f"\n[phase 2] chairman synthesizing from {successful} perspectives...", flush=True)
    synthesis = run_synthesis(
        spec=config.synthesizer,
        topic=config.topic,
        theorist_results=theorist_results,
        timeout_seconds=theorist_timeout,
    )
    if synthesis.success:
        print(
            f"[chairman done] {len(synthesis.content)}ch  "
            f"{synthesis.duration_seconds:.1f}s  ${synthesis.cost_usd:.4f}",
            flush=True,
        )
    else:
        print(f"[chairman FAIL] {synthesis.error}", flush=True)

    finished_at = datetime.now(timezone.utc)
    path = write_artifact(
        config=config,
        theorist_results=theorist_results,
        synthesis=synthesis,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
    )
    total_cost = sum(r.cost_usd for r in theorist_results) + synthesis.cost_usd
    duration = (finished_at - started_at).total_seconds()
    print(
        f"\n[done] artifact={path}  duration={duration:.1f}s  cost=${total_cost:.4f}",
        flush=True,
    )
    return 0 if synthesis.success else 4


def _skipped_synthesis(reason: str):
    from llm_council.synthesis import SynthesisResult
    return SynthesisResult(success=False, content="", error=reason)


if __name__ == "__main__":
    sys.exit(main())
