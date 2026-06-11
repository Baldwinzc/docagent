"""Web API tests that stay offline — they hit endpoints whose guards resolve
before any model is loaded (health, and auth rejection). The answer path itself
is exercised by the LLM-gated suites."""

import pytest
from fastapi.testclient import TestClient

import docagent.web as web


@pytest.fixture
def client():
    return TestClient(web.app)


def test_health_needs_no_models(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_ask_rejects_missing_api_key(client, monkeypatch):
    monkeypatch.setenv("DOCAGENT_API_KEY", "secret")
    # no X-API-Key -> 401 from the auth dependency, before the agent is touched
    r = client.post("/api/ask", json={"question": "anything"})
    assert r.status_code == 401


def test_ask_rejects_wrong_api_key(client, monkeypatch):
    monkeypatch.setenv("DOCAGENT_API_KEY", "secret")
    r = client.post(
        "/api/ask", json={"question": "anything"}, headers={"X-API-Key": "nope"}
    )
    assert r.status_code == 401
