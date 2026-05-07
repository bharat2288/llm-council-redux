"""Write the council artifact in the shape the /council skill expects.

Frontmatter contract (per chat-surface ADR Invariant 7 + skill spec):

    type: research
    project: <project>
    date: YYYY-MM-DD
    created_by: council
    status: draft
    topic: <topic>
    mode: <preset>
    theorists: [list of model ids]
    synthesizer: <chairman model id>
    cost_usd: <float>
    duration_seconds: <float>

Body sections:
    # [[<project>-home|<project>]] — Council: <topic>
    *[[dev-hub|Hub]]*

    > Convened <iso-timestamp> via /council. Status: draft — promote to
    > canonical after operator review.

    ## Theorist Responses
    ### <name1>
    <verbatim>
    ### <name2>
    <verbatim>
    ...

    <synthesis content — chairman already structured this with
     ## Synthesis / ## Tensions Worth Flagging / ## Recommendations>
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_council.config import Config
from llm_council.routing import TheoristResult
from llm_council.synthesis import SynthesisResult


def write_artifact(
    *,
    config: Config,
    theorist_results: list[TheoristResult],
    synthesis: SynthesisResult,
    output_dir: Path,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """Write the council markdown artifact and return its path.

    Also writes a sibling `.json` with raw perspectives + usage so callers
    that want machine-readable run data don't have to re-parse the markdown.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_slug = started_at.strftime("%Y-%m-%d-%H%M")
    filename = f"{timestamp_slug}-{config.artifact.topic_slug}.md"
    md_path = output_dir / filename
    json_path = md_path.with_suffix(".json")

    duration = (finished_at - started_at).total_seconds()
    total_cost = sum(r.cost_usd for r in theorist_results) + synthesis.cost_usd

    body = _build_markdown(
        config=config,
        theorist_results=theorist_results,
        synthesis=synthesis,
        started_at=started_at,
        duration_seconds=duration,
        total_cost_usd=total_cost,
    )
    md_path.write_text(body, encoding="utf-8")

    raw = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration,
        "config": {
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
        },
        "theorist_results": [
            {
                "name": r.name,
                "model": r.model,
                "routing": r.routing,
                "success": r.success,
                "content_chars": len(r.content),
                "cost_usd": r.cost_usd,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for r in theorist_results
        ],
        "synthesis": {
            "success": synthesis.success,
            "content_chars": len(synthesis.content),
            "cost_usd": synthesis.cost_usd,
            "duration_seconds": synthesis.duration_seconds,
            "error": synthesis.error,
        },
        "artifact_path": str(md_path),
        "total_cost_usd": total_cost,
    }
    json_path.write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return md_path


def _build_markdown(
    *,
    config: Config,
    theorist_results: list[TheoristResult],
    synthesis: SynthesisResult,
    started_at: datetime,
    duration_seconds: float,
    total_cost_usd: float,
) -> str:
    fm = _frontmatter(
        config=config,
        started_at=started_at,
        duration_seconds=duration_seconds,
        total_cost_usd=total_cost_usd,
    )
    nav = _nav_header(config.project, config.topic)
    convened_line = (
        f"> Convened {started_at.isoformat()} via `/council`. "
        f"Mode: {config.mode}. Status: {config.artifact.status} — promote to "
        f"canonical after operator review by editing the frontmatter."
    )

    parts = [fm, nav, "", convened_line, ""]

    # Failure summary if anything fell over — keeps the run honest about
    # how many theorists actually contributed.
    failures = [r for r in theorist_results if not r.success]
    if failures:
        parts.append("> **Theorists that failed to respond:**")
        for r in failures:
            parts.append(f">   - {r.name} ({r.model}): {r.error}")
        parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("## Theorist Responses")
    parts.append("")
    for r in theorist_results:
        if not r.success:
            continue
        parts.append(f"### {r.name} ({r.model})")
        parts.append("")
        parts.append(r.content.strip())
        parts.append("")

    parts.append("---")
    parts.append("")
    if synthesis.success:
        parts.append(synthesis.content.strip())
    else:
        parts.append("## Synthesis")
        parts.append("")
        parts.append(
            f"_Chairman synthesis failed: {synthesis.error}. Theorist responses "
            f"above are preserved for operator review._"
        )

    parts.append("")
    return "\n".join(parts)


def _frontmatter(
    *,
    config: Config,
    started_at: datetime,
    duration_seconds: float,
    total_cost_usd: float,
) -> str:
    theorist_models = [t.model for t in config.theorists]
    lines = [
        "---",
        "type: research",
        f"project: {config.project}",
        f"date: {started_at.strftime('%Y-%m-%d')}",
        "created_by: council",
        f"status: {config.artifact.status}",
        f"topic: {_yaml_scalar(config.topic)}",
        f"mode: {config.mode}",
        f"theorists: [{', '.join(theorist_models)}]",
        f"synthesizer: {config.synthesizer.model}",
        f"cost_usd: {total_cost_usd:.4f}",
        f"duration_seconds: {duration_seconds:.1f}",
        "---",
    ]
    return "\n".join(lines) + "\n"


def _nav_header(project: str, topic: str) -> str:
    title = f"Council: {topic}"
    return f"# [[{project}-home|{_titleize(project)}]] — {title}\n*[[dev-hub|Hub]]*\n"


def _titleize(slug: str) -> str:
    return " ".join(w.capitalize() for w in slug.replace("_", "-").split("-"))


def _yaml_scalar(s: str) -> str:
    """Wrap a string in quotes if it contains YAML-special chars, else bare."""
    if any(ch in s for ch in ":#'\"[]{}\n"):
        # Use double quotes and escape any internal double quotes.
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s
