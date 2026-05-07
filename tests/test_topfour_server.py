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

import pytest

# topfour_server.py lives at the repo root, not inside the package.
# Add the repo root to sys.path so we can import it like any other module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def client():
    """Flask test client for the topfour server."""
    from topfour_server import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


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
