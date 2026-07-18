from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.config.loader import load_model_config, load_profile
from driftwatch.db.models import (
    EvaluationWindow,
    Label,
    MetricStatus,
    PerformanceResult,
    Prediction,
)
from driftwatch.metrics.registry import get_metric

LabeledPrediction = tuple[Prediction, Label]


def recompute_performance_for_window(
    session: Session, evaluation_window_id: int
) -> list[PerformanceResult]:
    window = session.get(EvaluationWindow, evaluation_window_id)
    if window is None:
        raise ValueError(f"no evaluation window with id={evaluation_window_id}")

    model_config = load_model_config(window.model_id)
    profile = load_profile(model_config.profile)

    predictions = session.scalars(
        select(Prediction).where(
            Prediction.model_id == window.model_id,
            Prediction.predicted_at >= window.window_start,
            Prediction.predicted_at < window.window_end,
        )
    ).all()

    prediction_ids = [p.prediction_id for p in predictions]
    labels = (
        session.scalars(
            select(Label).where(
                Label.model_id == window.model_id,
                Label.prediction_id.in_(prediction_ids),
            )
        ).all()
        if prediction_ids
        else []
    )

    latest_label_by_prediction: dict[str, Label] = {}
    for label in labels:
        current = latest_label_by_prediction.get(label.prediction_id)
        # Postgres now() is transaction-start time, so labels inserted in the same
        # transaction can share received_at; fall back to id (insertion order) to
        # break ties deterministically.
        if current is None or (label.received_at, label.id) > (current.received_at, current.id):
            latest_label_by_prediction[label.prediction_id] = label

    labeled_predictions: list[LabeledPrediction] = [
        (prediction, latest_label_by_prediction[prediction.prediction_id])
        for prediction in predictions
        if prediction.prediction_id in latest_label_by_prediction
        and prediction.prediction_score is not None
    ]

    is_retroactive = window.performance_computed_at is not None
    results: list[PerformanceResult] = []

    def compute_and_store(
        rows: list[LabeledPrediction],
        segment_dimension: str | None,
        segment_value: str | None,
    ) -> None:
        if segment_dimension is not None and len(rows) < model_config.segments.min_segment_size:
            return
        y_true = [float(label.label_value) for _, label in rows]
        y_pred = [float(prediction.prediction_score) for prediction, _ in rows]  # type: ignore[arg-type]
        for spec in profile.performance_metrics:
            metric_fn = get_metric(spec.name)
            try:
                value = metric_fn(y_true, y_pred, spec.params)
                result = PerformanceResult(
                    evaluation_window_id=window.id,
                    segment_dimension=segment_dimension,
                    segment_value=segment_value,
                    metric_name=spec.name,
                    status=MetricStatus.COMPUTED,
                    metric_value=value,
                    n_labeled=len(rows),
                    is_retroactive=is_retroactive,
                )
            except ValueError as exc:
                result = PerformanceResult(
                    evaluation_window_id=window.id,
                    segment_dimension=segment_dimension,
                    segment_value=segment_value,
                    metric_name=spec.name,
                    status=MetricStatus.NOT_COMPUTABLE,
                    not_computable_reason=str(exc),
                    n_labeled=len(rows),
                    is_retroactive=is_retroactive,
                )
            session.add(result)
            results.append(result)

    if labeled_predictions:
        compute_and_store(labeled_predictions, None, None)

    for dimension in model_config.segments.dimensions:
        segment_values = {
            prediction.segment_values.get(dimension)
            for prediction, _ in labeled_predictions
            if prediction.segment_values.get(dimension) is not None
        }
        for segment_value in segment_values:
            rows = [
                (prediction, label)
                for prediction, label in labeled_predictions
                if prediction.segment_values.get(dimension) == segment_value
            ]
            compute_and_store(rows, dimension, str(segment_value))

    window.performance_computed_at = datetime.now(UTC)
    window.n_labels = len(labeled_predictions)

    return results
