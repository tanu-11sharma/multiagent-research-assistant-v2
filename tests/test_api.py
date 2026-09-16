from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["documents_loaded"] == 5


def test_research_endpoint_single_topic():
    response = client.post("/research", json={"question": "What is the CAP theorem?"})
    assert response.status_code == 200
    body = response.json()
    assert body["approved"] is True
    assert len(body["sections"]) == 1
    assert body["sections"][0]["doc_id"] == "distributed_systems"


def test_research_endpoint_multi_topic():
    response = client.post(
        "/research",
        json={
            "question": (
                "How does the Python GIL affect concurrency and how does "
                "distributed systems consensus like Raft work?"
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["sections"]) == 2


def test_research_endpoint_validates_empty_question():
    response = client.post("/research", json={"question": ""})
    assert response.status_code == 422
