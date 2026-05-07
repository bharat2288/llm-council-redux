"""Tests for the topfour HTTP daemon (pipeline #63).

Server lives in `topfour_server.py` at the repo root (alongside the
existing `council_server.py` for the legacy 3-model engine — separate
file, separate engine, separate port).

Tests use Flask's test client — no real socket bind, no real subprocesses
fired. Theorist routing is mocked so council fires don't actually shell
out to claude-cli / codex-cli / gemini-cli (those are tested
end-to-end via the CLI skill itself; the daemon just wraps the engine).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_council.routing import TheoristResult

# topfour_server.py lives at the repo root, not inside the package.
# Add the repo root to sys.path so we can import it like any other module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def client(tmp_path):
    """Flask test client for the topfour server.

    The fixture configures `project_root` to a tmp_path so the daemon
    writes artifacts under the test's temp dir instead of the real
    C:/Users/bhara/dev/specs/ tree. This keeps test runs hermetic and
    lets us assert artifact content without polluting the project specs.
    """
    from topfour_server import create_app
    app = create_app(project_root=str(tmp_path))
    app.config["TESTING"] = True
    with app.test_client() as c:
        c.tmp_path = tmp_path  # expose to tests for path assertions
        yield c


@pytest.fixture
def mock_theorist():
    """Patch routing.fire_theorist with a fast canned response so council
    fires complete in milliseconds. Affects both theorist firing AND
    chairman synthesis (chairman uses the same dispatch).
    """
    def fake_fire(spec, prompt, timeout_seconds=600):
        return TheoristResult(
            name=spec.name,
            model=spec.model,
            routing=spec.routing,
            success=True,
            content=(
                f"Mock {spec.name} response. The operator's question is: "
                f"{prompt[:80]}..."
                if "ORIGINAL QUERY" not in prompt
                else (
                    "## Synthesis\n\nMock synthesis.\n\n"
                    "## Tensions Worth Flagging\n\n- mock tension\n\n"
                    "## Recommendations\n\nMock recommendation."
                )
            ),
            cost_usd=0.0,
            duration_seconds=0.01,
        )

    # `topfour_server` and `llm_council.synthesis` both do
    # `from llm_council.routing import fire_theorist` at import time, so
    # each module holds its own local reference. Patching the source
    # module doesn't reach those local refs. Patch each call site
    # explicitly. (Standard Python "where the name is bound" gotcha.)
    with patch("topfour_server.fire_theorist", side_effect=fake_fire), \
         patch("llm_council.synthesis.fire_theorist", side_effect=fake_fire):
        yield


# -- /api/topfour/defaults endpoint ----------------------------------------


class TestDefaultsEndpoint:
    """GET /api/topfour/defaults?mode=<mode>

    Wraps llm_council.defaults.default_config_for_mode over HTTP. This is
    what the Meshbook chat-dropdown caller queries before showing the
    operator the config-confirmation step — same source of truth as the
    /council CLI skill.
    """

    def test_known_mode_returns_canonical_config(self, client) -> None:
        response = client.get("/api/topfour/defaults?mode=free-3-model-with-gemini-cli")
        assert response.status_code == 200
        assert response.is_json
        body = response.get_json()
        assert body["mode"] == "free-3-model-with-gemini-cli"
        assert len(body["theorists"]) == 3

    def test_each_known_mode_round_trips(self, client) -> None:
        for mode in (
            "free-3-model-with-gemini-cli",
            "free-2-model",
            "standard-paid",
        ):
            response = client.get(f"/api/topfour/defaults?mode={mode}")
            assert response.status_code == 200, f"{mode}: {response.data!r}"
            body = response.get_json()
            assert body["mode"] == mode

    def test_response_matches_module_function(self, client) -> None:
        # The endpoint must NOT diverge from the module function — the whole
        # point of the daemon is to expose it over HTTP. Round-trip both
        # callers and assert identical structure.
        from llm_council.defaults import default_config_for_mode

        mode = "standard-paid"
        response = client.get(f"/api/topfour/defaults?mode={mode}")
        assert response.status_code == 200
        http_body = response.get_json()
        module_body = default_config_for_mode(mode)
        assert http_body == module_body

    def test_unknown_mode_returns_400(self, client) -> None:
        response = client.get("/api/topfour/defaults?mode=not-a-real-mode")
        assert response.status_code == 400
        assert response.is_json
        body = response.get_json()
        # Error envelope: {"error": {"code": "...", "message": "..."}}
        assert "error" in body
        assert body["error"]["code"] == "unknown_mode"
        assert "not-a-real-mode" in body["error"]["message"]

    def test_missing_mode_param_returns_400(self, client) -> None:
        response = client.get("/api/topfour/defaults")
        assert response.status_code == 400
        body = response.get_json()
        assert body["error"]["code"] == "invalid_request"


# -- POST /api/topfour/start endpoint --------------------------------------


class TestStartEndpoint:
    """POST /api/topfour/start — fire a council synchronously, return
    {run_id, artifact_path, ...}.

    Slice 3 is synchronous (await full council, then respond). Slice 4
    will refactor to async + add /stream for SSE; this slice's tests
    will continue to pass because /start's response shape doesn't change.
    """

    def _payload(self, **overrides):
        """Minimal valid payload."""
        body = {
            "topic": "Test topic for council fire — small concrete question.",
            "project": "meshbook",
            "mode": "free-3-model-with-gemini-cli",
        }
        body.update(overrides)
        return body

    def test_mode_only_payload_fires_with_defaults(self, client, mock_theorist) -> None:
        # Operator passes a mode and topic; daemon expands defaults from
        # llm_council.defaults and fires.
        response = client.post("/api/topfour/start", json=self._payload())
        assert response.status_code == 200, response.data
        body = response.get_json()
        assert "run_id" in body
        assert "artifact_path" in body
        assert body["total_cost_usd"] == 0.0
        assert body["duration_seconds"] >= 0.0

    def test_artifact_actually_written_to_disk(self, client, mock_theorist) -> None:
        response = client.post("/api/topfour/start", json=self._payload())
        assert response.status_code == 200
        artifact_path = Path(response.get_json()["artifact_path"])
        assert artifact_path.is_file(), f"artifact not at {artifact_path}"
        content = artifact_path.read_text(encoding="utf-8")
        assert "type: research" in content
        assert "Mock claude response" in content  # theorist content rendered
        assert "## Synthesis" in content

    def test_artifact_lands_under_project_specs_research_runs(self, client, mock_theorist) -> None:
        response = client.post("/api/topfour/start", json=self._payload(project="scholia"))
        assert response.status_code == 200
        artifact_path = Path(response.get_json()["artifact_path"])
        # Path should be: <tmp_path>/scholia/specs/research/runs/<file>.md
        relative = artifact_path.relative_to(client.tmp_path)
        assert relative.parts[:4] == ("scholia", "specs", "research", "runs")
        assert relative.suffix == ".md"

    def test_explicit_theorists_override_mode_defaults(self, client, mock_theorist) -> None:
        # Caller can pass `theorists` directly instead of relying on `mode`
        # defaults — this is how the chat-dropdown caller will send the
        # operator's edited config (operator might swap a model, drop one).
        custom = self._payload()
        custom["theorists"] = [
            {"name": "claude", "model": "sonnet", "effort": "high", "routing": "claude-cli"},
            {"name": "gpt",    "model": "gpt-5.5", "effort": "high", "routing": "codex-cli"},
        ]
        custom["synthesizer"] = {"model": "sonnet", "effort": "high", "routing": "claude-cli"}
        response = client.post("/api/topfour/start", json=custom)
        assert response.status_code == 200, response.data
        # Artifact frontmatter should reflect the custom theorists.
        artifact_path = Path(response.get_json()["artifact_path"])
        content = artifact_path.read_text(encoding="utf-8")
        assert "sonnet" in content  # the custom claude model

    def test_missing_topic_returns_400(self, client, mock_theorist) -> None:
        response = client.post("/api/topfour/start", json=self._payload(topic=""))
        assert response.status_code == 400
        body = response.get_json()
        assert body["error"]["code"] == "invalid_request"

    def test_missing_project_returns_400(self, client, mock_theorist) -> None:
        payload = self._payload()
        del payload["project"]
        response = client.post("/api/topfour/start", json=payload)
        assert response.status_code == 400

    def test_unknown_mode_returns_400(self, client, mock_theorist) -> None:
        response = client.post("/api/topfour/start", json=self._payload(mode="not-real"))
        assert response.status_code == 400
        body = response.get_json()
        assert body["error"]["code"] in ("unknown_mode", "invalid_request")

    def test_no_mode_or_theorists_returns_400(self, client, mock_theorist) -> None:
        # Caller must specify ONE of: a mode (use defaults) or theorists
        # (explicit list). Without either, the engine has nothing to fire.
        payload = self._payload()
        del payload["mode"]
        response = client.post("/api/topfour/start", json=payload)
        assert response.status_code == 400
