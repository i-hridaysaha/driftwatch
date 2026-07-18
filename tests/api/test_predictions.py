from fastapi.testclient import TestClient

VALID_FEATURES = {"age": 30, "region": "EU", "income": 50000}


def _register_baseline(client: TestClient) -> None:
    response = client.post(
        "/models/example-model/baseline",
        json={"records": [{"features": VALID_FEATURES, "prediction_score": 0.4}]},
    )
    assert response.status_code == 201


def _prediction(prediction_id: str, score: float = 0.7) -> dict:
    return {
        "prediction_id": prediction_id,
        "predicted_at": "2026-01-01T00:00:00Z",
        "features": VALID_FEATURES,
        "prediction_value": {"score": score},
        "prediction_score": score,
    }


def test_ingest_predictions_happy_path(client: TestClient) -> None:
    _register_baseline(client)

    response = client.post(
        "/models/example-model/predictions",
        json={"predictions": [_prediction("p1"), _prediction("p2")]},
    )

    assert response.status_code == 201
    assert response.json() == {"inserted": 2, "skipped_duplicate": 0}


def test_ingest_predictions_without_baseline_returns_404(client: TestClient) -> None:
    response = client.post(
        "/models/example-model/predictions", json={"predictions": [_prediction("p1")]}
    )

    assert response.status_code == 404


def test_ingest_predictions_allows_omitting_nullable_feature(client: TestClient) -> None:
    _register_baseline(client)
    record = _prediction("p1")
    record["features"] = {"age": 30, "region": "EU"}  # income is nullable, ok to omit

    response = client.post("/models/example-model/predictions", json={"predictions": [record]})

    assert response.status_code == 201


def test_ingest_predictions_missing_required_feature_returns_422(client: TestClient) -> None:
    _register_baseline(client)
    bad = _prediction("p1")
    bad["features"] = {"age": 30, "income": 50000}  # missing required "region"

    response = client.post("/models/example-model/predictions", json={"predictions": [bad]})

    assert response.status_code == 422


def test_resending_identical_prediction_is_idempotent(client: TestClient) -> None:
    _register_baseline(client)
    record = _prediction("p1")

    first = client.post("/models/example-model/predictions", json={"predictions": [record]})
    second = client.post("/models/example-model/predictions", json={"predictions": [record]})

    assert first.json() == {"inserted": 1, "skipped_duplicate": 0}
    assert second.json() == {"inserted": 0, "skipped_duplicate": 1}


def test_resending_prediction_id_with_different_payload_returns_409(client: TestClient) -> None:
    _register_baseline(client)
    client.post("/models/example-model/predictions", json={"predictions": [_prediction("p1", 0.7)]})

    response = client.post(
        "/models/example-model/predictions", json={"predictions": [_prediction("p1", 0.9)]}
    )

    assert response.status_code == 409


def test_batch_with_conflicting_duplicate_ids_returns_422(client: TestClient) -> None:
    _register_baseline(client)

    response = client.post(
        "/models/example-model/predictions",
        json={"predictions": [_prediction("p1", 0.7), _prediction("p1", 0.9)]},
    )

    assert response.status_code == 422
