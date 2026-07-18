from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from driftwatch.db.models import Baseline, EvaluationWindow, Label, MetricStatus, Model, Prediction
from driftwatch.evaluation.performance import recompute_performance_for_window


def _make_window(
    db_session: Session, model_id: str, window_start: datetime, window_end: datetime
) -> EvaluationWindow:
    db_session.add(Model(model_id=model_id))
    db_session.flush()
    baseline = Baseline(model_id=model_id, is_active=True, binning_config={})
    db_session.add(baseline)
    db_session.flush()
    window = EvaluationWindow(
        model_id=model_id,
        baseline_id=baseline.id,
        window_start=window_start,
        window_end=window_end,
        config_hash="test-hash",
        evaluated_at=datetime.now(UTC),
    )
    db_session.add(window)
    db_session.flush()
    return window


def _add_labeled_prediction(
    db_session: Session,
    model_id: str,
    prediction_id: str,
    predicted_at: datetime,
    score: float,
    label: int,
) -> None:
    db_session.add(
        Prediction(
            prediction_id=prediction_id,
            model_id=model_id,
            predicted_at=predicted_at,
            features={"age": 30, "region": "EU"},
            prediction_value={"score": score},
            prediction_score=score,
            segment_values={"region": "EU"},
            payload_hash=f"hash-{prediction_id}",
        )
    )
    db_session.flush()
    db_session.add(
        Label(
            prediction_id=prediction_id,
            model_id=model_id,
            label_value=label,
            labeled_at=predicted_at + timedelta(hours=2),
            payload_hash=f"lhash-{prediction_id}",
        )
    )
    db_session.flush()


def test_recompute_computes_global_metrics(db_session: Session) -> None:
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    window = _make_window(db_session, "example-model", window_start, window_end)

    for i, (score, label) in enumerate([(0.9, 1), (0.1, 0), (0.8, 1), (0.2, 0)]):
        _add_labeled_prediction(
            db_session, "example-model", f"p{i}", window_start + timedelta(minutes=i), score, label
        )

    results = recompute_performance_for_window(db_session, window.id)

    global_results = {r.metric_name: r for r in results if r.segment_dimension is None}
    assert global_results["pr_auc"].status == MetricStatus.COMPUTED
    assert global_results["pr_auc"].metric_value == 1.0
    assert global_results["pr_auc"].n_labeled == 4
    assert global_results["pr_auc"].is_retroactive is False
    assert window.performance_computed_at is not None
    assert window.n_labels == 4

    # segment breakdown skipped: 4 rows < example-model's min_segment_size of 30
    assert all(r.segment_dimension is None for r in results)


def test_recompute_marks_second_call_as_retroactive(db_session: Session) -> None:
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    window = _make_window(db_session, "example-model", window_start, window_end)

    _add_labeled_prediction(db_session, "example-model", "p0", window_start, 0.9, 1)
    _add_labeled_prediction(
        db_session, "example-model", "p1", window_start + timedelta(minutes=1), 0.2, 0
    )

    first_pass = recompute_performance_for_window(db_session, window.id)
    assert all(r.is_retroactive is False for r in first_pass)
    assert all(r.status == MetricStatus.COMPUTED for r in first_pass)

    second_pass = recompute_performance_for_window(db_session, window.id)
    assert all(r.is_retroactive is True for r in second_pass)


def test_metric_failure_skips_metric_not_window(db_session: Session) -> None:
    """A window where every label is the same class (e.g. the first labels to
    backfill for a rare-event classifier) makes class-balance-dependent metrics
    like roc_auc mathematically undefined. That must be recorded as a visible
    not_computable row with a reason, not silently dropped, and it must not
    prevent the other configured metrics from computing normally."""
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    window = _make_window(db_session, "example-model", window_start, window_end)

    _add_labeled_prediction(db_session, "example-model", "p0", window_start, 0.9, 1)
    _add_labeled_prediction(
        db_session, "example-model", "p1", window_start + timedelta(minutes=1), 0.4, 1
    )

    results = recompute_performance_for_window(db_session, window.id)
    by_metric = {r.metric_name: r for r in results if r.segment_dimension is None}

    assert by_metric["roc_auc"].status == MetricStatus.NOT_COMPUTABLE
    assert by_metric["roc_auc"].metric_value is None
    assert by_metric["roc_auc"].not_computable_reason
    assert "class" in by_metric["roc_auc"].not_computable_reason.lower()

    # the other configured metrics still computed normally in the same pass
    assert by_metric["precision_at_threshold"].status == MetricStatus.COMPUTED
    assert by_metric["recall_at_threshold"].status == MetricStatus.COMPUTED
    assert by_metric["precision_at_k"].status == MetricStatus.COMPUTED

    # the window itself still completed its recompute -- one bad metric doesn't
    # block the others or leave the window looking un-evaluated
    assert window.performance_computed_at is not None
    assert window.n_labels == 2
