"""Tests for the topfour HTTP daemon (pipeline #63).

Server lives in `topfour_server.py` at the repo root (alongside the
existing `council_server.py` for the legacy 3-model engine — separate
file, separate engine, separate port).

Tests use Flask's test client — no real socket bind, no real subprocesses
fired. Theorist routing is mocked so council fires don't actually shell
out to claude-cli / codex-cli / agy-cli (those are tested
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
    def fake_fire(spec, prompt, timeout_seconds=600, include_dirs=()):
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
        response = client.get("/api/topfour/defaults?mode=free-3-model-with-agy")
        assert response.status_code == 200
        assert response.is_json
        body = response.get_json()
        assert body["mode"] == "free-3-model-with-agy"
        assert len(body["theorists"]) == 3
        assert body["theorists"][2]["routing"] == "agy-cli"

    def test_each_known_mode_round_trips(self, client) -> None:
        for mode in (
            "free-3-model-with-agy",
            "free-4-model-with-kimi",
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
        assert [t["name"] for t in http_body["theorists"]] == [
            "claude",
            "gpt",
            "gemini",
            "grok",
            "glm",
        ]
        assert http_body["theorists"][4]["routing"] == "openrouter"
        assert http_body["theorists"][4]["model"] == "z-ai/glm-5.2"

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
    """POST /api/topfour/start — kicks off a council in a worker thread
    and returns `{run_id, status: "started"}` immediately. Artifact path
    arrives in the `done` event on /stream/<run_id>.

    Tests subscribe to the stream after starting to get the artifact
    location — this exercises the full request lifecycle the chat-
    dropdown caller will use.
    """

    def _payload(self, **overrides):
        """Minimal valid payload."""
        body = {
            "topic": "Test topic for council fire — small concrete question.",
            "project": "meshbook",
            "mode": "free-3-model-with-agy",
        }
        body.update(overrides)
        return body

    def _start_and_drain(self, client, payload):
        """POST /start, GET /stream until `done`, return (start_body, done_data).
        Helper that captures the typical caller flow in one call."""
        start = client.post("/api/topfour/start", json=payload)
        assert start.status_code == 200, start.data
        run_id = start.get_json()["run_id"]
        stream = client.get(f"/api/topfour/stream/{run_id}")
        assert stream.status_code == 200
        events = _parse_sse(stream.get_data())
        done_events = [e for e in events if e["event"] == "done"]
        assert done_events, f"no done event in stream; events were {[e['event'] for e in events]}"
        return start.get_json(), done_events[0]["data"]

    def test_mode_only_payload_fires_with_defaults(self, client, mock_theorist) -> None:
        # Operator passes a mode and topic; daemon expands defaults from
        # llm_council.defaults and fires.
        start, done = self._start_and_drain(client, self._payload())
        assert "run_id" in start
        assert start["status"] == "started"
        # Run-level metadata arrives in the done event, not the start response.
        assert "artifact_path" in done
        assert done["total_cost_usd"] == 0.0
        assert done["duration_seconds"] >= 0.0

    def test_artifact_actually_written_to_disk(self, client, mock_theorist) -> None:
        _, done = self._start_and_drain(client, self._payload())
        artifact_path = Path(done["artifact_path"])
        assert artifact_path.is_file(), f"artifact not at {artifact_path}"
        content = artifact_path.read_text(encoding="utf-8")
        assert "type: research" in content
        assert "Mock claude response" in content  # theorist content rendered
        assert "## Synthesis" in content

    def test_artifact_lands_under_project_specs_research_runs(self, client, mock_theorist) -> None:
        _, done = self._start_and_drain(client, self._payload(project="scholia"))
        artifact_path = Path(done["artifact_path"])
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
        _, done = self._start_and_drain(client, custom)
        # Artifact frontmatter should reflect the custom theorists.
        artifact_path = Path(done["artifact_path"])
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


# -- /api/topfour/stream SSE -----------------------------------------------


def _parse_sse(raw_bytes: bytes) -> list[dict]:
    """Parse a Flask test-client SSE response into a list of event dicts.

    Each event has the form `event: <name>\\ndata: <json>\\n\\n`. Returns
    [{event, data: dict}, ...] in arrival order.
    """
    import json as _json
    text = raw_bytes.decode("utf-8")
    events: list[dict] = []
    current: dict = {}
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if not line:
            if current:
                if "data" in current:
                    try:
                        current["data"] = _json.loads(current["data"])
                    except _json.JSONDecodeError:
                        pass
                events.append(current)
                current = {}
            continue
        if line.startswith("event: "):
            current["event"] = line[len("event: "):]
        elif line.startswith("data: "):
            current["data"] = line[len("data: "):]
    if current:
        events.append(current)
    return events


class TestStreamEndpoint:
    """POST /api/topfour/start (slice 4 refactor) returns run_id immediately
    and kicks off the council in a background thread.
    GET /api/topfour/stream/<run_id> yields the event stream as SSE.

    Coarse event taxonomy (per #63 spec, fine-grained streaming is v0.2):
      - theorist_started {name, model, routing}
      - theorist_done    {name, success, content_chars, cost_usd, duration_seconds}
      - chairman_started {model}
      - chairman_done    {success, content_chars, cost_usd, duration_seconds}
      - synthesis_skipped {reason}              (only when <2 theorists succeed)
      - error            {code, message}        (per-theorist or run-level)
      - done             {artifact_path, total_cost_usd, duration_seconds}
    """

    def _payload(self, **overrides):
        body = {
            "topic": "Test SSE streaming with mocked theorists.",
            "project": "meshbook",
            "mode": "free-3-model-with-agy",
        }
        body.update(overrides)
        return body

    def test_start_returns_run_id_immediately(self, client, mock_theorist) -> None:
        # In slice 4, start no longer blocks until council completes.
        # It returns the run_id right away; consumer subscribes via /stream.
        response = client.post("/api/topfour/start", json=self._payload())
        assert response.status_code == 200
        body = response.get_json()
        assert "run_id" in body
        # status field tells caller the run is in-flight rather than done.
        assert body["status"] == "started"

    def test_stream_emits_theorist_events_in_order(self, client, mock_theorist) -> None:
        start = client.post("/api/topfour/start", json=self._payload())
        run_id = start.get_json()["run_id"]

        stream = client.get(f"/api/topfour/stream/{run_id}")
        assert stream.status_code == 200
        assert stream.mimetype == "text/event-stream"

        events = _parse_sse(stream.get_data())
        event_names = [e["event"] for e in events]

        # Three theorists in free-3-model: each gets started + done. Then
        # chairman started + done. Then run-level done.
        assert event_names.count("theorist_started") == 3
        assert event_names.count("theorist_done") == 3
        assert event_names.count("chairman_started") == 1
        assert event_names.count("chairman_done") == 1
        assert event_names[-1] == "done"

        # Each theorist_done's name must match a prior theorist_started's name.
        started_names = [e["data"]["name"] for e in events if e["event"] == "theorist_started"]
        done_names = [e["data"]["name"] for e in events if e["event"] == "theorist_done"]
        assert sorted(started_names) == sorted(done_names) == ["claude", "gemini", "gpt"]

    def test_stream_done_event_has_artifact_path(self, client, mock_theorist) -> None:
        start = client.post("/api/topfour/start", json=self._payload(project="scholia"))
        run_id = start.get_json()["run_id"]
        stream = client.get(f"/api/topfour/stream/{run_id}")
        events = _parse_sse(stream.get_data())
        done_events = [e for e in events if e["event"] == "done"]
        assert len(done_events) == 1
        done = done_events[0]["data"]
        assert "artifact_path" in done
        artifact_path = Path(done["artifact_path"])
        # File actually exists; landed under scholia's specs/research/runs.
        assert artifact_path.is_file()
        relative = artifact_path.relative_to(client.tmp_path)
        assert relative.parts[:4] == ("scholia", "specs", "research", "runs")

    def test_stream_unknown_run_id_returns_404(self, client) -> None:
        response = client.get("/api/topfour/stream/not-a-real-id")
        assert response.status_code == 404
        body = response.get_json()
        assert body["error"]["code"] == "unknown_run"

    def test_stream_includes_theorist_done_payload_metadata(
        self, client, mock_theorist
    ) -> None:
        # theorist_done events carry success/cost/timing for the operator UI
        # to render progress.
        start = client.post("/api/topfour/start", json=self._payload())
        run_id = start.get_json()["run_id"]
        stream = client.get(f"/api/topfour/stream/{run_id}")
        events = _parse_sse(stream.get_data())
        theorist_done = next(e for e in events if e["event"] == "theorist_done")
        d = theorist_done["data"]
        assert d["success"] is True
        assert "content_chars" in d
        assert "cost_usd" in d
        assert "duration_seconds" in d


# -- DELETE /api/topfour/<run_id> cancellation -----------------------------


class TestCancellation:
    """DELETE /api/topfour/<run_id> sets the cancelled flag on the run.

    The worker thread checks the flag at safe points (currently: before
    chairman synthesis). When cancelled, the chairman phase is skipped
    and a `cancelled` event is emitted before `done`. The artifact is
    still written for the record (with synthesis_skipped) so the
    operator can review whatever theorist responses came back before
    the cancel.

    True subprocess termination (killing in-flight claude-cli /
    codex-cli / agy-cli) is deferred to v0.2 — would require
    fire_theorist to expose a Popen handle for the worker to kill.
    """

    def _payload(self, **overrides):
        body = {
            "topic": "Test cancellation flow.",
            "project": "meshbook",
            "mode": "free-3-model-with-agy",
        }
        body.update(overrides)
        return body

    def test_delete_unknown_run_returns_404(self, client) -> None:
        response = client.delete("/api/topfour/not-a-real-id")
        assert response.status_code == 404
        body = response.get_json()
        assert body["error"]["code"] == "unknown_run"

    def test_delete_known_run_returns_200_and_marks_cancelled(
        self, client, mock_theorist
    ) -> None:
        # Ordinary completed run — DELETE after-the-fact still marks cancelled
        # but the artifact already wrote with normal flow. Useful for chat
        # surfaces that fire DELETE on tab-close regardless of run state.
        start = client.post("/api/topfour/start", json=self._payload())
        run_id = start.get_json()["run_id"]
        # Drain the stream so the worker thread completes.
        client.get(f"/api/topfour/stream/{run_id}")

        response = client.delete(f"/api/topfour/{run_id}")
        assert response.status_code == 200
        body = response.get_json()
        assert body["cancelled"] is True
        assert body["run_id"] == run_id

    def test_delete_during_run_skips_chairman_and_emits_cancelled(
        self, client
    ) -> None:
        # Use a blocking mock so theorists hold mid-flight long enough for
        # DELETE to land before chairman synthesis starts. The cancelled
        # flag should cause synthesis to be skipped and the artifact to
        # be written with synthesis_skipped status.
        import threading

        release = threading.Event()
        started_count = threading.Semaphore(0)  # signals each theorist start

        def blocking_fire(spec, prompt, timeout_seconds=600, include_dirs=()):
            # Chairman has name "chairman" — only block theorists, not
            # synthesis (synthesis won't be called if cancellation lands
            # before chairman phase, but defensive code anyway).
            if spec.name != "chairman":
                started_count.release()  # signal: this theorist has started
                release.wait(timeout=10)
            return TheoristResult(
                name=spec.name, model=spec.model, routing=spec.routing,
                success=True,
                content=f"Mock {spec.name} response.",
                cost_usd=0.0, duration_seconds=0.01,
            )

        from unittest.mock import patch
        with patch("topfour_server.fire_theorist", side_effect=blocking_fire), \
             patch("llm_council.synthesis.fire_theorist", side_effect=blocking_fire):
            start = client.post("/api/topfour/start", json=self._payload())
            run_id = start.get_json()["run_id"]

            # Wait for at least one theorist to have started (proves the
            # worker thread is in the theorist phase, not still parsing
            # config). Bounded — fail loud if the worker stalls.
            assert started_count.acquire(timeout=5), "no theorist started in time"

            # Now DELETE — cancellation should land before theorists complete.
            response = client.delete(f"/api/topfour/{run_id}")
            assert response.status_code == 200
            assert response.get_json()["cancelled"] is True

            # Release the blocked theorists so the worker can proceed,
            # see the cancelled flag at the chairman checkpoint, and
            # emit the cancelled event.
            release.set()

            stream = client.get(f"/api/topfour/stream/{run_id}")

        events = _parse_sse(stream.get_data())
        event_names = [e["event"] for e in events]
        # Cancelled event must appear; chairman phase must NOT have run.
        assert "cancelled" in event_names
        assert "chairman_started" not in event_names
        # Done event still fires so consumers don't hang.
        assert event_names[-1] == "done"
        # done.cancelled flag tells the consumer the run ended via cancellation.
        done = next(e for e in events if e["event"] == "done")
        assert done["data"].get("cancelled") is True
