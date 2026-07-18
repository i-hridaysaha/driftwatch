from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.db.models import Baseline, EvaluationWindow

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


def _create_evaluated_window(
    db_session: Session, window_start: datetime, window_end: datetime
) -> None:
    baseline_id = db_session.scalars(
        select(Baseline.id).where(
            Baseline.model_id == "example-model", Baseline.is_active.is_(True)
        )
    ).one()
    db_session.add(
        EvaluationWindow(
            model_id="example-model",
            baseline_id=baseline_id,
            window_start=window_start,
            window_end=window_end,
            config_hash="test-hash",
            evaluated_at=datetime.now(UTC),
        )
    )
    db_session.flush()


def _get_window(db_session: Session, window_start: datetime) -> EvaluationWindow:
    return db_session.scalars(
        select(EvaluationWindow).where(
            EvaluationWindow.model_id == "example-model",
            EvaluationWindow.window_start == window_start,
        )
    ).one()


def test_late_prediction_is_stored_and_counted_not_dropped(
    client: TestClient, db_session: Session
) -> None:
    _register_baseline(client)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _create_evaluated_window(db_session, window_start, window_end)

    late = _prediction("late-1")
    late["predicted_at"] = (window_start + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")

    response = client.post("/models/example-model/predictions", json={"predictions": [late]})

    assert response.status_code == 201
    assert response.json()["inserted"] == 1  # still stored, never rejected
    assert _get_window(db_session, window_start).late_prediction_count == 1


def test_on_time_prediction_does_not_increment_late_count(
    client: TestClient, db_session: Session
) -> None:
    _register_baseline(client)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _create_evaluated_window(db_session, window_start, window_end)

    # falls entirely outside the already-evaluated window -- not late
    on_time = _prediction("on-time-1")
    on_time["predicted_at"] = (window_end + timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    response = client.post("/models/example-model/predictions", json={"predictions": [on_time]})

    assert response.status_code == 201
    assert _get_window(db_session, window_start).late_prediction_count == 0
