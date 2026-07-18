from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.db.models import Baseline

VALID_RECORD = {"features": {"age": 30, "region": "EU", "income": 50000}, "prediction_score": 0.4}


def test_register_baseline_creates_model_and_baseline(
    client: TestClient, db_session: Session
) -> None:
    response = client.post(
        "/models/example-model/baseline", json={"records": [VALID_RECORD, VALID_RECORD]}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["n_records"] == 2

    baseline = db_session.get(Baseline, body["baseline_id"])
    assert baseline is not None
    assert baseline.is_active is True


def test_register_baseline_unknown_model_returns_404(client: TestClient) -> None:
    response = client.post("/models/does-not-exist/baseline", json={"records": [VALID_RECORD]})

    assert response.status_code == 404


def test_register_baseline_rejects_unknown_feature(client: TestClient) -> None:
    bad_record = {"features": {**VALID_RECORD["features"], "extra_field": 1}}

    response = client.post("/models/example-model/baseline", json={"records": [bad_record]})

    assert response.status_code == 422
    assert "unknown features" in str(response.json()["detail"])


def test_register_baseline_rejects_wrong_dtype(client: TestClient) -> None:
    bad_record = {"features": {"age": "not-a-number", "region": "EU", "income": 50000}}

    response = client.post("/models/example-model/baseline", json={"records": [bad_record]})

    assert response.status_code == 422
    assert "must be numeric" in str(response.json()["detail"])


def test_re_registering_baseline_deactivates_prior_one(
    client: TestClient, db_session: Session
) -> None:
    first = client.post("/models/example-model/baseline", json={"records": [VALID_RECORD]})
    second = client.post("/models/example-model/baseline", json={"records": [VALID_RECORD]})

    first_id = first.json()["baseline_id"]
    second_id = second.json()["baseline_id"]

    active_ids = db_session.scalars(
        select(Baseline.id).where(
            Baseline.model_id == "example-model", Baseline.is_active.is_(True)
        )
    ).all()

    assert active_ids == [second_id]
    assert db_session.get(Baseline, first_id).is_active is False  # type: ignore[union-attr]
