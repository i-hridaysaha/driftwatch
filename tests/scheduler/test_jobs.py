from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.alerting.notifications import MAX_NOTIFICATION_ATTEMPTS
from driftwatch.config.loader import load_model_config
from driftwatch.db.models import (
    Alert,
    AlertKind,
    AlertStatus,
    Baseline,
    EvaluationWindow,
    Model,
    Prediction,
)
from driftwatch.scheduler.jobs import (
    evaluate_pending_windows,
    find_drift_ready_unevaluated_windows,
    retry_pending_notifications,
)
from driftwatch.stats.binning import compute_baseline_binning

MODEL_ID = "example-model"


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
        for i in range(55):  # above aggressive's min_window_size of 50
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


def test_find_drift_ready_unevaluated_windows_respects_watermark(db_session: Session) -> None:
    _register_baseline(db_session)
    start_hour = datetime(2026, 1, 1, 0, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start_hour, n_hours=3)

    # as_of right at the end of hour 3's data, before its watermark (10m) elapses
    as_of = start_hour + timedelta(hours=3)

    windows = find_drift_ready_unevaluated_windows(db_session, MODEL_ID, as_of)

    # hours 0 and 1's drift watermark has fully elapsed; hour 2's window ends
    # exactly at as_of, so its watermark hasn't elapsed yet -- only 0 and 1
    # are drift-ready
    assert windows == [
        (start_hour, start_hour + timedelta(hours=1)),
        (start_hour + timedelta(hours=1), start_hour + timedelta(hours=2)),
    ]


def test_find_drift_ready_windows_excludes_already_evaluated(db_session: Session) -> None:
    _register_baseline(db_session)
    start_hour = datetime(2026, 1, 1, 0, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start_hour, n_hours=2)
    as_of = start_hour + timedelta(hours=3)

    baseline_id = db_session.scalars(
        select(Baseline.id).where(Baseline.model_id == MODEL_ID)
    ).one()
    db_session.add(
        EvaluationWindow(
            model_id=MODEL_ID,
            baseline_id=baseline_id,
            window_start=start_hour,
            window_end=start_hour + timedelta(hours=1),
            config_hash="test-hash",
            evaluated_at=datetime.now(UTC),
        )
    )
    db_session.flush()

    windows = find_drift_ready_unevaluated_windows(db_session, MODEL_ID, as_of)

    assert windows == [(start_hour + timedelta(hours=1), start_hour + timedelta(hours=2))]


def test_find_drift_ready_windows_no_predictions_returns_empty(db_session: Session) -> None:
    _register_baseline(db_session)

    windows = find_drift_ready_unevaluated_windows(db_session, MODEL_ID, datetime.now(UTC))

    assert windows == []


def test_evaluate_pending_windows_evaluates_and_commits(db_session: Session) -> None:
    _register_baseline(db_session)
    start_hour = datetime(2026, 1, 1, 0, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start_hour, n_hours=2)
    as_of = start_hour + timedelta(hours=3)

    count = evaluate_pending_windows(db_session, as_of=as_of)

    assert count == 2
    evaluated = db_session.scalars(
        select(EvaluationWindow).where(EvaluationWindow.model_id == MODEL_ID)
    ).all()
    assert len(evaluated) == 2


def test_evaluate_pending_windows_is_idempotent_on_rerun(db_session: Session) -> None:
    _register_baseline(db_session)
    start_hour = datetime(2026, 1, 1, 0, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start_hour, n_hours=1)
    as_of = start_hour + timedelta(hours=2)

    first_count = evaluate_pending_windows(db_session, as_of=as_of)
    second_count = evaluate_pending_windows(db_session, as_of=as_of)

    assert first_count == 1
    assert second_count == 0  # nothing new to evaluate


def test_evaluate_pending_windows_skips_inactive_models(db_session: Session) -> None:
    _register_baseline(db_session)
    model = db_session.get(Model, MODEL_ID)
    assert model is not None
    model.is_active = False
    db_session.flush()

    start_hour = datetime(2026, 1, 1, 0, tzinfo=UTC)
    _add_predictions_across_hours(db_session, start_hour, n_hours=1)

    count = evaluate_pending_windows(db_session, as_of=start_hour + timedelta(hours=2))

    assert count == 0


class _FlakyChannel:
    """Fails the first `failures` rounds, then delivers: a webhook outage."""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.delivered: list[tuple[int, str]] = []

    def notify(self, alert: Alert, event: str) -> None:
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("webhook unreachable")
        self.delivered.append((alert.id, event))


def test_retry_pending_notifications_delivers_once_the_webhook_is_back(
    db_session: Session,
) -> None:
    """An alert whose opening notification failed is picked up by the
    scheduler's retry job on later ticks until it is delivered, then never
    again for that status."""
    db_session.add(Model(model_id=MODEL_ID))
    db_session.flush()
    alert = Alert(
        model_id=MODEL_ID,
        kind=AlertKind.DRIFT,
        signal_name="psi",
        feature_name="age",
        status=AlertStatus.OPEN,
    )
    db_session.add(alert)
    db_session.flush()
    channel = _FlakyChannel(failures=2)

    assert retry_pending_notifications(db_session, [channel]) == 1  # tick 1: fails
    assert alert.notification_attempts == 1 and alert.last_notified_status is None
    assert retry_pending_notifications(db_session, [channel]) == 1  # tick 2: fails
    assert retry_pending_notifications(db_session, [channel]) == 1  # tick 3: delivered
    assert channel.delivered == [(alert.id, "opened")]
    assert alert.last_notified_status == AlertStatus.OPEN
    assert alert.notification_attempts == 0
    assert retry_pending_notifications(db_session, [channel]) == 0  # nothing pending


def test_retry_pending_notifications_skips_alerts_that_exhausted_their_attempts(
    db_session: Session,
) -> None:
    db_session.add(Model(model_id=MODEL_ID))
    db_session.flush()
    alert = Alert(
        model_id=MODEL_ID,
        kind=AlertKind.DRIFT,
        signal_name="psi",
        feature_name="age",
        status=AlertStatus.OPEN,
        notification_attempts=MAX_NOTIFICATION_ATTEMPTS,
    )
    db_session.add(alert)
    db_session.flush()

    assert retry_pending_notifications(db_session, [_FlakyChannel(failures=0)]) == 0
