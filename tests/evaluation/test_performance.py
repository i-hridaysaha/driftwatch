from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from driftwatch.db.models import Baseline, EvaluationWindow, Label, MetricStatus, Model, Prediction
from driftwatch.evaluation.performance import recompute_performance_for_window

MIN_WINDOW_SIZE = 50  # aggressive profile's evaluation.min_window_size


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
    region: str = "EU",
) -> None:
    db_session.add(
        Prediction(
            prediction_id=prediction_id,
            model_id=model_id,
            predicted_at=predicted_at,
            features={"age": 30, "region": region},
            prediction_value={"score": score},
            prediction_score=score,
            segment_values={"region": region},
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


def _add_many_labeled_predictions(
    db_session: Session,
    model_id: str,
    window_start: datetime,
    n: int,
    *,
    region: str = "EU",
) -> None:
    for i in range(n):
        _add_labeled_prediction(
            db_session,
            model_id,
            f"bulk-{window_start.isoformat()}-{region}-{i}",
            window_start + timedelta(seconds=i),
            score=0.1 + (i % 9) / 10,
            label=i % 2,
            region=region,
        )


def test_recompute_computes_global_metrics(db_session: Session) -> None:
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    window = _make_window(db_session, "example-model", window_start, window_end)

    _add_many_labeled_predictions(db_session, "example-model", window_start, MIN_WINDOW_SIZE)

    results = recompute_performance_for_window(db_session, window.id)

    global_results = {r.metric_name: r for r in results if r.segment_dimension is None}
    assert global_results["pr_auc"].status == MetricStatus.COMPUTED
    assert global_results["pr_auc"].n_labeled == MIN_WINDOW_SIZE
    assert global_results["pr_auc"].is_retroactive is False
    assert window.performance_computed_at is not None
    assert window.n_labels == MIN_WINDOW_SIZE
    assert window.label_watermark is not None


def test_window_below_min_size_marks_global_not_computable_with_reason(
    db_session: Session,
) -> None:
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    window = _make_window(db_session, "example-model", window_start, window_end)

    _add_labeled_prediction(db_session, "example-model", "p0", window_start, 0.9, 1)
    _add_labeled_prediction(
        db_session, "example-model", "p1", window_start + timedelta(minutes=1), 0.2, 0
    )

    results = recompute_performance_for_window(db_session, window.id)
    global_results = [r for r in results if r.segment_dimension is None]

    assert global_results  # rows exist -- not silently absent
    assert all(r.status == MetricStatus.NOT_COMPUTABLE for r in global_results)
    assert all(
        r.not_computable_reason and "2 labeled predictions" in r.not_computable_reason
        for r in global_results
    )


def test_segment_below_min_size_marks_segment_not_computable_with_reason(
    db_session: Session,
) -> None:
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    window = _make_window(db_session, "example-model", window_start, window_end)

    # enough total volume to clear the window-level gate, but concentrated in one
    # segment value so a different, sparsely-populated segment value stays below
    # example-model's min_segment_size of 30
    _add_many_labeled_predictions(db_session, "example-model", window_start, 50, region="EU")
    _add_labeled_prediction(
        db_session, "example-model", "sparse-0", window_start, 0.5, 1, region="US"
    )

    results = recompute_performance_for_window(db_session, window.id)
    us_results = [
        r for r in results if r.segment_dimension == "region" and r.segment_value == "US"
    ]

    assert us_results  # rows exist -- not silently absent
    assert all(r.status == MetricStatus.NOT_COMPUTABLE for r in us_results)
    assert all(
        r.not_computable_reason and "1 labeled predictions" in r.not_computable_reason
        for r in us_results
    )

    eu_results = [
        r for r in results if r.segment_dimension == "region" and r.segment_value == "EU"
    ]
    assert any(r.status == MetricStatus.COMPUTED for r in eu_results)


def test_recompute_marks_second_call_as_retroactive(db_session: Session) -> None:
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    window = _make_window(db_session, "example-model", window_start, window_end)

    _add_many_labeled_predictions(db_session, "example-model", window_start, MIN_WINDOW_SIZE)

    first_pass = recompute_performance_for_window(db_session, window.id)
    assert all(r.is_retroactive is False for r in first_pass)
    assert any(r.status == MetricStatus.COMPUTED for r in first_pass)

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

    for i in range(MIN_WINDOW_SIZE):
        _add_labeled_prediction(
            db_session,
            "example-model",
            f"p{i}",
            window_start + timedelta(seconds=i),
            score=0.1 + (i % 9) / 10,
            label=1,  # every label the same class -> roc_auc is undefined
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
    assert window.n_labels == MIN_WINDOW_SIZE
