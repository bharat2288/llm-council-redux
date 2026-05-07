"""Chairman synthesis pass."""
from __future__ import annotations

import time
from dataclasses import dataclass

from llm_council.config import SynthesizerSpec
from llm_council.routing import TheoristResult, fire_theorist
from llm_council.config import TheoristSpec
from llm_council.errors import SynthesisFailure


CHAIRMAN_SYSTEM = (
    "You are the Chairman of a Council of frontier AI advisors. "
    "Synthesize the theorist responses into a single authoritative answer. "
    "Preserve the strongest concrete recommendations; resolve contradictions; "
    "where theorists disagree, state which position is safer or stronger and why. "
    "Structure the output with these sections in order:\n\n"
    "  ## Synthesis\n"
    "  Headline conclusion + the reasoning that holds across theorists.\n\n"
    "  ## Tensions Worth Flagging\n"
    "  Bullet list of disagreements that the operator should think hard about — "
    "places where the theorists genuinely diverged and the resolution isn't obvious.\n\n"
    "  ## Recommendations\n"
    "  Best-effort answer to the question with explicit caveats and operator "
    "next-step options.\n\n"
    "Use plain-text headings (no parentheses, em-dashes, or colons in ## headings). "
    "Be dense and specific."
)


@dataclass(frozen=True)
class SynthesisResult:
    success: bool
    content: str
    error: str | None = None
    cost_usd: float = 0.0
    duration_seconds: float = 0.0


def run_synthesis(
    spec: SynthesizerSpec,
    topic: str,
    theorist_results: list[TheoristResult],
    timeout_seconds: int = 600,
) -> SynthesisResult:
    """Build the chairman prompt from successful theorist results and fire it."""
    successful = [r for r in theorist_results if r.success and r.content]
    if len(successful) < 2:
        return SynthesisResult(
            success=False,
            content="",
            error=(
                f"only {len(successful)} theorist(s) succeeded; chairman synthesis "
                "needs at least 2 perspectives to resolve tensions"
            ),
        )

    blocks = []
    for r in successful:
        blocks.append(f"\n\n=== {r.name.upper()} ({r.model}) ===\n{r.content}")
    prompt = (
        f"ORIGINAL QUERY:\n{topic}\n\n"
        f"{len(successful)} COUNCIL PERSPECTIVES:\n{''.join(blocks)}\n\n"
        f"---\nProduce the synthesis now per the section structure in your "
        f"system instructions."
    )

    # Reuse the theorist-firing dispatch — chairman is just another model
    # with a different system framing. We pass CHAIRMAN_SYSTEM through a
    # one-off TheoristSpec; routing.py prepends THEORIST_SYSTEM, but for
    # the chairman we want CHAIRMAN_SYSTEM, so we wrap the prompt itself.
    wrapped = f"{CHAIRMAN_SYSTEM}\n\n{prompt}"
    chairman_spec = TheoristSpec(
        name="chairman",
        model=spec.model,
        effort=spec.effort,
        routing=spec.routing,
    )
    t0 = time.monotonic()
    result = fire_theorist(chairman_spec, wrapped, timeout_seconds=timeout_seconds)
    duration = time.monotonic() - t0
    if not result.success:
        return SynthesisResult(
            success=False,
            content="",
            error=result.error or "chairman call failed",
            duration_seconds=duration,
        )
    return SynthesisResult(
        success=True,
        content=result.content,
        cost_usd=result.cost_usd,
        duration_seconds=duration,
    )
