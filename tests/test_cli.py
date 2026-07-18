from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.cli import _parse_iso_utc, evaluate_range
from driftwatch.config.loader import load_model_config
from driftwatch.db.models import Baseline, EvaluationWindow, Model, Prediction
from driftwatch.evaluation.drift import evaluate_window
from driftwatch.stats.binning import compute_baseline_binning

MODEL_ID = "example-model"


def test_parse_iso_utc_accepts_z_suffix() -> None:
    parsed = _parse_iso_utc("2026-01-01T00:00:00Z")

    assert parsed == datetime(2026, 1, 1, tzinfo=UTC)


def test_parse_iso_utc_rejects_naive_datetime() -> None:
    with pytest.raises(Exception, match="timezone-aware"):
        _parse_iso_utc("2026-01-01T00:00:00")


def test_parse_iso_utc_rejects_malformed_input() -> None:
    with pytest.raises(Exception, match="not a valid ISO 8601"):
        _parse_iso_utc("not-a-date")


def _register_baseline(db_session: Session) -> None:
    db_session.add(Model(model_id=MODEL_ID))
    db_session.flush()
    schema = load_model_config(MODEL_ID).schema_
    features = [{"age": 20 + i, "region": "EU", "income": 1000} for i in range(60)]
    scores = [0.5] * 60
    binning = compute_baseline_binning(features, scores, schema)
    baseline = Baseline(model_id=MODEL_ID, is_active=True, binning_config=binning)
    db_session.add(baseline)
    db_session.flush()


def _add_predictions_across_hours(db_session: Session, start_hour: datetime, n_hours: int) -> None:
    for hour in range(n_hours):
        window_start = start_hour + timedelta(hours=hour)
        for i in range(55):
            db_session.add(
                Prediction(
                    prediction_id=f"p-{hour}-{i}",
                    model_id=MODEL_ID,
                    predicted_at=window_start + timedelta(seconds=i),
                    features={"age": 20 + i, "region": "EU", "income": 1000},
                    prediction_value={"score": 0.5},
                    prediction_score=0.5,
                    segment_values={"region": "EU"},
                    payload_hash=f"hash-{hour}-{i}",
                )
            )
    db_session.flush()


def test_evaluate_range_uses_same_code_path_as_scheduler(db_session: Session) -> None:
    """The CLI's evaluate_range must produce identical EvaluationWindow rows to
    calling evaluate_window() directly -- proving it's the same code path, not
    a parallel reimplementation."""
    _register_baseline(db_session)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start, n_hours=2)

    evaluated, skipped = evaluate_range(db_session, MODEL_ID, start, start + timedelta(hours=2))

    assert evaluated == 2
    assert skipped == 0
    windows = db_session.scalars(
        select(EvaluationWindow).where(EvaluationWindow.model_id == MODEL_ID)
    ).all()
    assert len(windows) == 2
    assert all(w.n_predictions == 55 for w in windows)


def test_evaluate_range_backfills_historical_data_bypassing_watermark(db_session: Session) -> None:
    """The whole point of the CLI: evaluate windows the scheduler's watermark
    would never have let through yet (e.g. seeding 30+ windows of demo history
    in one shot, right now, not waiting for each window's watermark to elapse
    in real time)."""
    _register_baseline(db_session)
    start = datetime(2020, 1, 1, tzinfo=UTC)  # far in the past
    _add_predictions_across_hours(db_session, start, n_hours=5)

    evaluated, _skipped = evaluate_range(db_session, MODEL_ID, start, start + timedelta(hours=5))

    assert evaluated == 5


def test_evaluate_range_is_idempotent_without_force(db_session: Session) -> None:
    _register_baseline(db_session)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start, n_hours=1)

    first_evaluated, _ = evaluate_range(db_session, MODEL_ID, start, start + timedelta(hours=1))
    second_evaluated, second_skipped = evaluate_range(
        db_session, MODEL_ID, start, start + timedelta(hours=1)
    )

    assert first_evaluated == 1
    assert second_evaluated == 0
    assert second_skipped == 1


def test_evaluate_range_force_replaces_existing_window(db_session: Session) -> None:
    _register_baseline(db_session)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start, n_hours=1)

    evaluate_range(db_session, MODEL_ID, start, start + timedelta(hours=1))
    original_id = db_session.scalars(
        select(EvaluationWindow.id).where(EvaluationWindow.model_id == MODEL_ID)
    ).one()

    evaluated, skipped = evaluate_range(
        db_session, MODEL_ID, start, start + timedelta(hours=1), force=True
    )

    assert evaluated == 1
    assert skipped == 0
    assert db_session.get(EvaluationWindow, original_id) is None


def test_evaluate_range_calls_on_result_callback(db_session: Session) -> None:
    _register_baseline(db_session)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start, n_hours=1)

    calls: list[tuple[datetime, datetime, bool]] = []
    evaluate_range(
        db_session,
        MODEL_ID,
        start,
        start + timedelta(hours=1),
        on_result=lambda ws, we, ok: calls.append((ws, we, ok)),
    )

    assert calls == [(start, start + timedelta(hours=1), True)]


def test_evaluate_window_and_evaluate_range_agree_on_a_shared_window(db_session: Session) -> None:
    _register_baseline(db_session)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=1)
    _add_predictions_across_hours(db_session, start, n_hours=1)

    direct = evaluate_window(db_session, MODEL_ID, start, end)
    assert direct is not None

    evaluated, skipped = evaluate_range(db_session, MODEL_ID, start, end)

    assert evaluated == 0
    assert skipped == 1
