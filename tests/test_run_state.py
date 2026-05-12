"""Tests for llm_council.run_state and the resume CLI subcommand.

The handoff that motivated this module
(`specs/moom-order-forecasting/drafts/handoff-council-partial-resume-diagnosis-2026-05-13.md`)
calls out the exact failure shape these tests exercise:

  - codex-cli timed out after 300s but the artifact still wrote because
    chairman had 2/3 theorists. Operator wanted the theorist outputs and
    chairman to survive a process interruption, not only a clean finish.
  - manual fallback had no shared persistence; partial results vanished.

So the tests anchor on the operator-visible promise: if 2+ theorists
finish, their outputs live on disk before the chairman runs, and can be
re-synthesized later via `resume` without re-firing them.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council import run_state
from llm_council.config import (
    ArtifactSpec,
    Config,
    SynthesizerSpec,
    TheoristSpec,
)
from llm_council.routing import TheoristResult
from llm_council.synthesis import SynthesisResult


def _make_config(topic: str = "Test council topic") -> Config:
    return Config(
        topic=topic,
        project="test-project",
        mode="free-3-model-with-gemini-cli",
        theorists=(
            TheoristSpec(name="claude", model="claude-opus-4-7",
                         effort="xhigh", routing="claude-cli"),
            TheoristSpec(name="gpt", model="gpt-5.5",
                         effort="xhigh", routing="codex-cli"),
            TheoristSpec(name="gemini", model="gemini-3-pro-preview",
                         effort="high", routing="gemini-cli"),
        ),
        synthesizer=SynthesizerSpec(
            model="claude-opus-4-7", effort="xhigh", routing="claude-cli"
        ),
        artifact=ArtifactSpec(status="draft", topic_slug="test-council-topic"),
    )


def _ok_result(name: str, model: str, routing: str, content: str) -> TheoristResult:
    return TheoristResult(
        name=name, model=model, routing=routing, success=True,
        content=content, error=None, cost_usd=0.0, duration_seconds=120.0,
    )


def _fail_result(name: str, model: str, routing: str, error: str) -> TheoristResult:
    return TheoristResult(
        name=name, model=model, routing=routing, success=False,
        content="", error=error, cost_usd=0.0, duration_seconds=300.0,
    )


# -- run_state module -----------------------------------------------------


class TestCreateRunDir:
    def test_creates_dir_with_timestamp_slug_stem(self, tmp_path: Path) -> None:
        started = datetime(2026, 5, 12, 15, 34, tzinfo=timezone.utc)
        run_dir = run_state.create_run_dir(tmp_path, "my-topic", started)
        assert run_dir.is_dir()
        assert run_dir.name == "2026-05-12-1534-my-topic"
        assert (run_dir / "theorists").is_dir()

    def test_idempotent_for_existing_dir(self, tmp_path: Path) -> None:
        started = datetime(2026, 5, 12, 15, 34, tzinfo=timezone.utc)
        run_state.create_run_dir(tmp_path, "x", started)
        # second call should not raise
        again = run_state.create_run_dir(tmp_path, "x", started)
        assert again.exists()


class TestInitRunJson:
    def test_writes_initial_state_with_pending_theorists(self, tmp_path: Path) -> None:
        cfg = _make_config()
        started = datetime(2026, 5, 12, 15, 34, tzinfo=timezone.utc)
        run_dir = run_state.create_run_dir(tmp_path, cfg.artifact.topic_slug, started)
        run_state.init_run_json(run_dir, config=cfg, output_dir=tmp_path,
                                started_at=started)

        state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert state["status"] == "running"
        assert state["artifact_path"] is None
        assert state["finished_at"] is None
        assert set(state["theorists"].keys()) == {"claude", "gpt", "gemini"}
        for t in state["theorists"].values():
            assert t["status"] == "pending"

    def test_config_snapshot_round_trips_through_parse_config(self, tmp_path: Path) -> None:
        """The resume command rebuilds Config from this snapshot. If
        parse_config can't read it back, resume will explode."""
        from llm_council.config import parse_config

        cfg = _make_config()
        started = datetime(2026, 5, 12, 15, 34, tzinfo=timezone.utc)
        run_dir = run_state.create_run_dir(tmp_path, cfg.artifact.topic_slug, started)
        run_state.init_run_json(run_dir, config=cfg, output_dir=tmp_path,
                                started_at=started)
        state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        rebuilt = parse_config(state["config"])
        assert rebuilt.topic == cfg.topic
        assert [t.name for t in rebuilt.theorists] == ["claude", "gpt", "gemini"]
        assert rebuilt.synthesizer.model == cfg.synthesizer.model


class TestPersistTheorist:
    def test_writes_md_and_json_for_successful_result(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "theorists").mkdir(parents=True)
        r = _ok_result("claude", "claude-opus-4-7", "claude-cli", "## Synthesis\nContent")
        run_state.persist_theorist(run_dir, r)
        assert (run_dir / "theorists" / "claude.md").read_text(encoding="utf-8") \
               == "## Synthesis\nContent"
        meta = json.loads(
            (run_dir / "theorists" / "claude.json").read_text(encoding="utf-8")
        )
        assert meta["success"] is True
        assert meta["content_chars"] == len("## Synthesis\nContent")
        assert meta["error"] is None

    def test_writes_empty_md_and_error_for_failed_result(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        (run_dir / "theorists").mkdir(parents=True)
        r = _fail_result(
            "gpt", "gpt-5.5", "codex-cli",
            "subprocess timed out after 600s and was force-killed",
        )
        run_state.persist_theorist(run_dir, r)
        assert (run_dir / "theorists" / "gpt.md").read_text(encoding="utf-8") == ""
        meta = json.loads(
            (run_dir / "theorists" / "gpt.json").read_text(encoding="utf-8")
        )
        assert meta["success"] is False
        assert "timed out" in meta["error"]


class TestUpdateTheoristState:
    def test_flips_status_and_records_timings(self, tmp_path: Path) -> None:
        cfg = _make_config()
        started = datetime.now(timezone.utc)
        run_dir = run_state.create_run_dir(tmp_path, cfg.artifact.topic_slug, started)
        run_state.init_run_json(run_dir, config=cfg, output_dir=tmp_path,
                                started_at=started)

        r_ok = _ok_result("claude", "claude-opus-4-7", "claude-cli", "x" * 1000)
        run_state.update_theorist_state(run_dir, r_ok)
        r_bad = _fail_result("gpt", "gpt-5.5", "codex-cli", "timed out after 600s")
        run_state.update_theorist_state(run_dir, r_bad)

        state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert state["theorists"]["claude"]["status"] == "ok"
        assert state["theorists"]["claude"]["content_chars"] == 1000
        assert state["theorists"]["gpt"]["status"] == "fail"
        assert state["theorists"]["gpt"]["error"].startswith("timed out")
        # untouched theorist stays pending
        assert state["theorists"]["gemini"]["status"] == "pending"


class TestLoadPersistedTheorists:
    def test_only_returns_theorists_with_persisted_json(self, tmp_path: Path) -> None:
        """If a theorist never got persisted (e.g. process killed before
        future resolved), it must not appear in the resume list."""
        cfg = _make_config()
        started = datetime.now(timezone.utc)
        run_dir = run_state.create_run_dir(tmp_path, cfg.artifact.topic_slug, started)
        run_state.init_run_json(run_dir, config=cfg, output_dir=tmp_path,
                                started_at=started)
        # Persist only 2 of 3.
        run_state.persist_theorist(
            run_dir, _ok_result("claude", "claude-opus-4-7", "claude-cli", "claude body")
        )
        run_state.persist_theorist(
            run_dir, _ok_result("gemini", "gemini-3-pro-preview", "gemini-cli", "gemini body")
        )
        loaded = run_state.load_persisted_theorists(run_dir)
        assert [r.name for r in loaded] == ["claude", "gemini"]
        assert loaded[0].content == "claude body"
        assert loaded[1].content == "gemini body"

    def test_preserves_config_order_not_filesystem_order(self, tmp_path: Path) -> None:
        cfg = _make_config()  # order: claude, gpt, gemini
        started = datetime.now(timezone.utc)
        run_dir = run_state.create_run_dir(tmp_path, cfg.artifact.topic_slug, started)
        run_state.init_run_json(run_dir, config=cfg, output_dir=tmp_path,
                                started_at=started)
        # Persist in reverse order.
        run_state.persist_theorist(
            run_dir, _ok_result("gemini", "gemini-3-pro-preview", "gemini-cli", "g")
        )
        run_state.persist_theorist(
            run_dir, _ok_result("claude", "claude-opus-4-7", "claude-cli", "c")
        )
        loaded = run_state.load_persisted_theorists(run_dir)
        assert [r.name for r in loaded] == ["claude", "gemini"]


# -- resume CLI -----------------------------------------------------------


class TestResumeCLI:
    """Black-box tests against `python -m llm_council resume`.

    We stub `run_synthesis` so the test doesn't need real model calls
    but the full CLI plumbing (argparse, run.json load, config rebuild,
    artifact write, finalize_run_json) executes for real.
    """

    def _seed_partial_run(self, tmp_path: Path) -> tuple[Path, Path]:
        cfg = _make_config()
        started = datetime(2026, 5, 12, 15, 34, tzinfo=timezone.utc)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        run_dir = run_state.create_run_dir(
            output_dir, cfg.artifact.topic_slug, started
        )
        run_state.init_run_json(
            run_dir, config=cfg, output_dir=output_dir, started_at=started
        )
        # 2 successful theorists + 1 timeout, just like the route-resolver runs.
        for r in [
            _ok_result("claude", "claude-opus-4-7", "claude-cli",
                       "## Position\nClaude says X."),
            _fail_result("gpt", "gpt-5.5", "codex-cli",
                         "subprocess timed out after 600s and was force-killed"),
            _ok_result("gemini", "gemini-3-pro-preview", "gemini-cli",
                       "## Position\nGemini says Y."),
        ]:
            run_state.persist_theorist(run_dir, r)
            run_state.update_theorist_state(run_dir, r)
        return run_dir, output_dir

    def test_resume_synthesizes_from_two_successful_saved_theorists(
        self, tmp_path: Path
    ) -> None:
        from llm_council import cli as cli_module

        run_dir, output_dir = self._seed_partial_run(tmp_path)
        fake_synth = SynthesisResult(
            success=True,
            content="## Synthesis\nResumed chairman output.",
            cost_usd=0.0,
            duration_seconds=42.0,
        )
        with patch.object(cli_module, "run_synthesis", return_value=fake_synth):
            rc = cli_module.main(
                ["resume", "--run-dir", str(run_dir), "--theorist-timeout", "60"]
            )
        assert rc == 0

        # Final run.json reflects success.
        state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert state["status"] == "synthesized"
        assert state["finished_at"] is not None
        assert state["artifact_path"] is not None

        # synthesis files written.
        assert "Resumed chairman output" in (run_dir / "synthesis.md").read_text(
            encoding="utf-8"
        )

        # Flat artifact lands in output_dir, sibling to the run-dir.
        flat_md = Path(state["artifact_path"])
        assert flat_md.exists()
        assert flat_md.parent == output_dir.resolve()
        body = flat_md.read_text(encoding="utf-8")
        assert "Resumed chairman output" in body
        assert "Claude says X" in body
        assert "Gemini says Y" in body
        # The failed theorist is recorded in the failure summary, not in
        # the Theorist Responses section.
        assert "gpt" in body  # failure block
        assert "Theorists that failed to respond" in body

    def test_resume_rejects_run_with_fewer_than_two_successful(
        self, tmp_path: Path
    ) -> None:
        from llm_council import cli as cli_module

        cfg = _make_config()
        started = datetime.now(timezone.utc)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        run_dir = run_state.create_run_dir(
            output_dir, cfg.artifact.topic_slug, started
        )
        run_state.init_run_json(
            run_dir, config=cfg, output_dir=output_dir, started_at=started
        )
        # Only one success on disk.
        run_state.persist_theorist(
            run_dir, _ok_result("claude", "claude-opus-4-7", "claude-cli", "only one")
        )
        rc = cli_module.main(["resume", "--run-dir", str(run_dir)])
        assert rc == 3

    def test_resume_fails_clean_when_run_dir_has_no_run_json(
        self, tmp_path: Path
    ) -> None:
        from llm_council import cli as cli_module

        empty = tmp_path / "not-a-run"
        empty.mkdir()
        rc = cli_module.main(["resume", "--run-dir", str(empty)])
        assert rc == 1

    def test_synthesize_alias_does_the_same_thing(self, tmp_path: Path) -> None:
        from llm_council import cli as cli_module

        run_dir, _ = self._seed_partial_run(tmp_path)
        fake_synth = SynthesisResult(
            success=True, content="alias works", cost_usd=0.0, duration_seconds=1.0,
        )
        with patch.object(cli_module, "run_synthesis", return_value=fake_synth):
            rc = cli_module.main(["synthesize", "--run-dir", str(run_dir)])
        assert rc == 0
        state = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert state["status"] == "synthesized"
