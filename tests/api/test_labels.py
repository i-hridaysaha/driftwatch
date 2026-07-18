from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.db.models import Baseline, EvaluationWindow, PerformanceResult

VALID_FEATURES = {"age": 30, "region": "EU", "income": 50000}


def _register_baseline(client: TestClient) -> None:
    response = client.post(
        "/models/example-model/baseline",
        json={"records": [{"features": VALID_FEATURES, "prediction_score": 0.4}]},
    )
    assert response.status_code == 201


def _ingest_prediction(client: TestClient, prediction_id: str, predicted_at: str) -> None:
    response = client.post(
        "/models/example-model/predictions",
        json={
            "predictions": [
                {
                    "prediction_id": prediction_id,
                    "predicted_at": predicted_at,
                    "features": VALID_FEATURES,
                    "prediction_value": {"score": 0.7},
                    "prediction_score": 0.7,
                }
            ]
        },
    )
    assert response.status_code == 201


def test_ingest_labels_happy_path_without_window(client: TestClient) -> None:
    _register_baseline(client)
    _ingest_prediction(client, "p1", "2026-01-01T00:00:00Z")

    response = client.post(
        "/models/example-model/labels",
        json={
            "labels": [
                {"prediction_id": "p1", "label_value": 1, "labeled_at": "2026-01-02T00:00:00Z"}
            ]
        },
    )

    assert response.status_code == 201
    assert response.json() == {"inserted": 1, "skipped_duplicate": 0, "windows_recomputed": []}


def test_ingest_labels_unknown_prediction_returns_422(client: TestClient) -> None:
    _register_baseline(client)

    response = client.post(
        "/models/example-model/labels",
        json={
            "labels": [
                {
                    "prediction_id": "does-not-exist",
                    "label_value": 1,
                    "labeled_at": "2026-01-02T00:00:00Z",
                }
            ]
        },
    )

    assert response.status_code == 422


def test_resending_identical_label_is_idempotent(client: TestClient) -> None:
    _register_baseline(client)
    _ingest_prediction(client, "p1", "2026-01-01T00:00:00Z")
    label = {"prediction_id": "p1", "label_value": 1, "labeled_at": "2026-01-02T00:00:00Z"}

    first = client.post("/models/example-model/labels", json={"labels": [label]})
    second = client.post("/models/example-model/labels", json={"labels": [label]})

    assert first.json()["inserted"] == 1
    assert second.json() == {"inserted": 0, "skipped_duplicate": 1, "windows_recomputed": []}


def test_different_label_value_is_a_correction_not_a_conflict(client: TestClient) -> None:
    _register_baseline(client)
    _ingest_prediction(client, "p1", "2026-01-01T00:00:00Z")

    first = client.post(
        "/models/example-model/labels",
        json={
            "labels": [
                {"prediction_id": "p1", "label_value": 0, "labeled_at": "2026-01-02T00:00:00Z"}
            ]
        },
    )
    second = client.post(
        "/models/example-model/labels",
        json={
            "labels": [
                {"prediction_id": "p1", "label_value": 1, "labeled_at": "2026-01-03T00:00:00Z"}
            ]
        },
    )

    # both accepted as separate label events, not a 409 -- labels are corrections, not immutable
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["inserted"] == 1


def test_label_batch_triggers_recompute_once_per_affected_window(
    client: TestClient, db_session: Session
) -> None:
    _register_baseline(client)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    for i in range(3):
        _ingest_prediction(
            client,
            f"p{i}",
            (window_start + timedelta(minutes=i)).isoformat().replace("+00:00", "Z"),
        )

    baseline_id = db_session.scalars(
        select(Baseline.id).where(
            Baseline.model_id == "example-model", Baseline.is_active.is_(True)
        )
    ).one()
    window = EvaluationWindow(
        model_id="example-model",
        baseline_id=baseline_id,
        window_start=window_start,
        window_end=window_end,
        config_hash="test-hash",
        evaluated_at=datetime.now(UTC),
    )
    db_session.add(window)
    db_session.flush()
    window_id = window.id

    response = client.post(
        "/models/example-model/labels",
        json={
            "labels": [
                {"prediction_id": "p0", "label_value": 1, "labeled_at": "2026-01-02T00:00:00Z"},
                {"prediction_id": "p1", "label_value": 0, "labeled_at": "2026-01-02T00:00:00Z"},
                {"prediction_id": "p2", "label_value": 1, "labeled_at": "2026-01-02T00:00:00Z"},
            ]
        },
    )

    assert response.status_code == 201
    assert response.json()["windows_recomputed"] == [window_id]

    performance_rows = db_session.scalars(
        select(PerformanceResult).where(PerformanceResult.evaluation_window_id == window_id)
    ).all()
    assert len(performance_rows) > 0
    assert {r.metric_name for r in performance_rows} <= {
        "pr_auc",
        "precision_at_threshold",
        "recall_at_threshold",
        "precision_at_k",
        "roc_auc",
    }
