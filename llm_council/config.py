"""Config schema for `python -m llm_council fire --config <json>`.

The JSON shape is the contract between the engine and its callers:
  - the /council skill at ~/.claude/skills/council/SKILL.md
  - the future HTTP daemon (#63) producing the same shape per request

Schema must stay backwards-compatible — additive changes only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_council.errors import ConfigError


VALID_ROUTINGS = frozenset(
    {"claude-cli", "codex-cli", "gemini-cli", "agy-cli", "openrouter"}
)
# Effort vocabulary differs per routing — see Effort table in SKILL.md:
#   claude-cli   accepts low / medium / high / xhigh / max
#   codex-cli    accepts low / medium / high / xhigh
#   openrouter   accepts low / medium / high
#   gemini-cli   exposes no effort flag (effort is recorded but unused)
#   agy-cli      exposes no effort flag — effort is baked into the model
#                label, e.g. "Gemini 3.1 Pro (High)" (recorded but unused)
# We accept the union here and let each routing's underlying CLI/API reject
# values it can't handle. This is intentional: validating per-routing in
# config would require duplicating the matrix here AND in routing.py, and
# the CLI errors are already clear when an unsupported value reaches them.
VALID_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
VALID_MODES = frozenset(
    {
        "standard-paid",
        "free-2-model",
        "free-3-model-with-agy",
        "free-3-model-with-gemini-cli",
    }
)


@dataclass(frozen=True)
class TheoristSpec:
    """One theorist in a council run."""

    name: str  # short tag used in artifact section headers, e.g. "claude", "gpt"
    model: str  # provider-specific model id, e.g. "claude-opus-4-7", "gpt-5.5"
    effort: str  # "low" | "medium" | "high"
    routing: str  # which path to use to invoke this theorist


@dataclass(frozen=True)
class SynthesizerSpec:
    """The chairman that synthesizes theorist responses."""

    model: str
    effort: str
    routing: str


@dataclass(frozen=True)
class ArtifactSpec:
    """How to name + frontmatter the output artifact."""

    status: str = "draft"  # "draft" | "canonical"
    topic_slug: str = ""
    timestamp: str = ""  # ISO 8601


@dataclass(frozen=True)
class Config:
    """A complete council run config."""

    topic: str
    project: str
    mode: str
    theorists: tuple[TheoristSpec, ...]
    synthesizer: SynthesizerSpec
    artifact: ArtifactSpec
    include_dirs: tuple[str, ...] = ()
    # Workspace directories to expose to tool-using theorists. Required for
    # gemini-cli when the topic asks theorists to read files outside CWD —
    # gemini sandboxes hard by default and refuses paths outside its
    # workspace. claude-cli (--dangerously-skip-permissions) and codex-cli
    # (--sandbox read-only) ignore this field. Operator sets at fire time
    # when running a grounded council; see SKILL.md "Grounded mode".


def load_config(path: Path) -> Config:
    """Load + validate a config JSON file. Raises ConfigError on any issue."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config is not valid JSON: {exc}") from exc
    return parse_config(data)


def parse_config(data: Any) -> Config:
    if not isinstance(data, dict):
        raise ConfigError("top-level config must be a JSON object")

    topic = _require_str(data, "topic")
    project = _require_str(data, "project")
    mode = _require_str(data, "mode")
    if mode not in VALID_MODES:
        raise ConfigError(f"mode {mode!r} not in {sorted(VALID_MODES)}")

    raw_theorists = data.get("theorists")
    if not isinstance(raw_theorists, list) or not raw_theorists:
        raise ConfigError("theorists must be a non-empty list")
    theorists = tuple(_parse_theorist(t, idx) for idx, t in enumerate(raw_theorists))

    raw_synth = data.get("synthesizer")
    if not isinstance(raw_synth, dict):
        raise ConfigError("synthesizer must be an object")
    synth = SynthesizerSpec(
        model=_require_str(raw_synth, "model", "synthesizer.model"),
        effort=_validated_effort(
            raw_synth.get("effort", "high"), "synthesizer.effort"
        ),
        routing=_validated_routing(
            raw_synth.get("routing", "claude-cli"), "synthesizer.routing"
        ),
    )

    raw_artifact = data.get("artifact") or {}
    if not isinstance(raw_artifact, dict):
        raise ConfigError("artifact must be an object")
    artifact = ArtifactSpec(
        status=str(raw_artifact.get("status", "draft")),
        topic_slug=str(raw_artifact.get("topic_slug", "") or _slugify(topic)),
        timestamp=str(raw_artifact.get("timestamp", "")),
    )
    if artifact.status not in {"draft", "canonical"}:
        raise ConfigError(
            f"artifact.status must be 'draft' or 'canonical', got {artifact.status!r}"
        )

    raw_include = data.get("include_dirs") or ()
    if not isinstance(raw_include, (list, tuple)) or any(
        not isinstance(p, str) or not p for p in raw_include
    ):
        raise ConfigError("include_dirs must be a list of non-empty strings")
    include_dirs = tuple(raw_include)

    return Config(
        topic=topic,
        project=project,
        mode=mode,
        theorists=theorists,
        synthesizer=synth,
        artifact=artifact,
        include_dirs=include_dirs,
    )


def _parse_theorist(item: Any, idx: int) -> TheoristSpec:
    where = f"theorists[{idx}]"
    if not isinstance(item, dict):
        raise ConfigError(f"{where} must be an object")
    model = _require_str(item, "model", f"{where}.model")
    name = item.get("name") or _name_from_model(model)
    return TheoristSpec(
        name=str(name),
        model=model,
        effort=_validated_effort(item.get("effort", "high"), f"{where}.effort"),
        routing=_validated_routing(item.get("routing", ""), f"{where}.routing"),
    )


def _require_str(d: dict, key: str, label: str | None = None) -> str:
    label = label or key
    val = d.get(key)
    if not isinstance(val, str) or not val:
        raise ConfigError(f"{label} must be a non-empty string")
    return val


def _validated_effort(val: Any, label: str) -> str:
    s = str(val)
    if s not in VALID_EFFORTS:
        raise ConfigError(f"{label} must be one of {sorted(VALID_EFFORTS)}, got {s!r}")
    return s


def _validated_routing(val: Any, label: str) -> str:
    s = str(val)
    if s not in VALID_ROUTINGS:
        raise ConfigError(
            f"{label} must be one of {sorted(VALID_ROUTINGS)}, got {s!r}"
        )
    return s


def _name_from_model(model: str) -> str:
    """Heuristic short tag from a model id when the config didn't supply one."""
    lower = model.lower()
    if "claude" in lower:
        return "claude"
    if "gpt" in lower:
        return "gpt"
    if "gemini" in lower:
        return "gemini"
    if "grok" in lower:
        return "grok"
    return model.split("/")[-1].split("-")[0]


def _slugify(s: str) -> str:
    keep = []
    for ch in s.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in " -_/":
            keep.append("-")
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:60] or "council-run"
