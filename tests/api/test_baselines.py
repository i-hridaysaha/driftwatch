import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.db.models import Baseline
from driftwatch.settings import Settings

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


def test_register_baseline_freezes_real_bin_edges(client: TestClient, db_session: Session) -> None:
    records = [
        {
            "features": {"age": age, "region": region, "income": age * 1000},
            "prediction_score": age / 100,
        }
        for age, region in [(20, "EU"), (30, "US"), (40, "EU"), (50, "APAC"), (60, "US")]
    ]

    response = client.post("/models/example-model/baseline", json={"records": records})

    assert response.status_code == 201
    baseline = db_session.get(Baseline, response.json()["baseline_id"])
    assert baseline is not None

    age_bins = baseline.binning_config["features"]["age"]
    assert age_bins["type"] == "continuous"
    assert age_bins["edges"][0] == 20.0
    assert age_bins["edges"][-1] == 60.0

    region_bins = baseline.binning_config["features"]["region"]
    assert region_bins["type"] == "categorical"
    assert region_bins["categories"] == ["APAC", "EU", "US"]

    score_bins = baseline.binning_config["prediction_score"]
    assert score_bins["edges"][0] == 0.2
    assert score_bins["edges"][-1] == 0.6


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
    refused = client.post("/models/example-model/baseline", json={"records": [VALID_RECORD]})
    assert refused.status_code == 409  # replacing the active baseline needs saying so
    second = client.post(
        "/models/example-model/baseline",
        json={"records": [VALID_RECORD], "replace_active": True},
    )

    first_id = first.json()["baseline_id"]
    second_id = second.json()["baseline_id"]

    active_ids = db_session.scalars(
        select(Baseline.id).where(
            Baseline.model_id == "example-model", Baseline.is_active.is_(True)
        )
    ).all()

    assert active_ids == [second_id]
    assert db_session.get(Baseline, first_id).is_active is False  # type: ignore[union-attr]


def test_write_endpoints_require_the_api_key_when_one_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With DRIFTWATCH api_key set, every write needs X-API-Key and the
    liveness probe does not; without it (every other test here) the
    endpoints are open."""
    monkeypatch.setattr("driftwatch.api.deps.get_settings", lambda: Settings(api_key="s3cret"))

    assert client.get("/health").status_code == 200
    bare = client.post("/models/example-model/baseline", json={"records": [VALID_RECORD]})
    assert bare.status_code == 401
    wrong = client.post(
        "/models/example-model/baseline",
        json={"records": [VALID_RECORD]},
        headers={"X-API-Key": "nope"},
    )
    assert wrong.status_code == 401
    right = client.post(
        "/models/example-model/baseline",
        json={"records": [VALID_RECORD]},
        headers={"X-API-Key": "s3cret"},
    )
    assert right.status_code == 201
