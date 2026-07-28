from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_combined_audit_shape() -> None:
    response = client.post("/audit", json={"max_items": 2})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert list(payload) == ["exact_match", "logistic_regression", "k-NN"]
    assert all(isinstance(value, list) for value in payload.values())
    assert all(len(value) <= 2 for value in payload.values())
