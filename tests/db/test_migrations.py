from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from driftwatch.db.models import Baseline, Label, Model, Prediction
from driftwatch.settings import get_settings

EXPECTED_TABLES = {
    "models",
    "baselines",
    "baseline_records",
    "predictions",
    "labels",
    "evaluation_windows",
    "drift_results",
    "performance_results",
}


def test_migration_creates_all_tables(migrated_db: None) -> None:
    engine = create_engine(get_settings().database_url)
    try:
        inspector = inspect(engine)
        assert set(inspector.get_table_names()) >= EXPECTED_TABLES
    finally:
        engine.dispose()


def test_prediction_id_unique_per_model(db_session: Session) -> None:
    db_session.add(Model(model_id="m1"))
    db_session.flush()

    def make_prediction() -> Prediction:
        return Prediction(
            prediction_id="p1",
            model_id="m1",
            predicted_at=datetime.now(UTC),
            features={},
            prediction_value={},
            payload_hash="hash",
        )

    db_session.add(make_prediction())
    db_session.flush()

    db_session.add(make_prediction())
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_label_rejects_unknown_prediction(db_session: Session) -> None:
    db_session.add(Model(model_id="m2"))
    db_session.flush()

    db_session.add(
        Label(
            prediction_id="does-not-exist",
            model_id="m2",
            label_value={"outcome": 1},
            labeled_at=datetime.now(UTC),
            payload_hash="hash",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_only_one_active_baseline_per_model(db_session: Session) -> None:
    db_session.add(Model(model_id="m3"))
    db_session.flush()

    db_session.add(Baseline(model_id="m3", is_active=True, binning_config={}))
    db_session.flush()

    db_session.add(Baseline(model_id="m3", is_active=True, binning_config={}))
    with pytest.raises(IntegrityError):
        db_session.flush()
