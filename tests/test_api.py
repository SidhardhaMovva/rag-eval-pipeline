"""API tests using FastAPI's TestClient against the real app factory."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def client() -> TestClient:
    app = create_app(
        corpus_dir=REPO_ROOT / "data" / "corpus",
        golden_set_path=REPO_ROOT / "eval" / "golden_set.json",
    )
    return TestClient(app)


def test_health_reports_index(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["documents"] == 6
    assert body["chunks"] > 0


def test_query_returns_cited_answer(client):
    resp = client.post(
        "/query", json={"query": "What does mean reciprocal rank MRR measure?", "k": 5}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False
    assert body["citations"]
    assert body["retrieved"]
    assert body["confidence"] > 0


def test_query_abstains_without_evidence(client):
    resp = client.post("/query", json={"query": "What is the best recipe for a chocolate cake?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is True
    assert body["citations"] == []


def test_ingest_then_query_new_document(client):
    resp = client.post(
        "/ingest",
        json={
            "doc_id": "gardening",
            "text": "Tomatoes need full sunlight and regular watering. "
            "Plant tomatoes after the last frost in spring.",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["total_chunks"] >= 1

    resp = client.post("/query", json={"query": "When should I plant tomatoes?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["abstained"] is False
    assert any("gardening" in c for c in body["citations"])


def test_evaluate_endpoint_returns_metrics(client):
    resp = client.post("/evaluate", json={"k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["queries"] == 22
    assert body["recall@5"] >= 0.9
    assert "mrr" in body


def test_validation_errors(client):
    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/ingest", json={"doc_id": "", "text": "x"}).status_code == 422
    assert client.post("/query", json={"query": "x", "k": 0}).status_code == 422
