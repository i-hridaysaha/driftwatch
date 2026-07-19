import logging
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.alerting.notifications import (
    NotificationChannel,
    default_channels,
    notify_if_needed,
)
from driftwatch.config.loader import load_model_config, load_profile
from driftwatch.db.models import Alert, EvaluationWindow, Model, Prediction
from driftwatch.durations import parse_duration
from driftwatch.evaluation.drift import evaluate_window
from driftwatch.scheduler.windowing import compute_window_boundaries, is_drift_watermark_elapsed

logger = logging.getLogger(__name__)


def find_drift_ready_unevaluated_windows(
    session: Session, model_id: str, as_of: datetime
) -> list[tuple[datetime, datetime]]:
    """Windows for `model_id` whose DRIFT watermark has elapsed as of `as_of`
    but that have no EvaluationWindow row yet. Governs when drift gets its
    one-time evaluation; has no bearing on performance, which is recomputed
    independently and indefinitely as labels arrive regardless of whether a
    window's drift has already been evaluated.

    Bounds the search from the model's last evaluated window's end (or its
    earliest prediction, if nothing has been evaluated yet) up to `as_of` --
    so a scheduler that was down for hours, or that has never run before,
    catches up on every missed window the next time it ticks. It doesn't need
    one tick per window; that's the whole point of scanning a range instead
    of only asking "what window is 'now' in."
    """
    model_config = load_model_config(model_id)
    profile = load_profile(model_config.profile)
    window_duration = parse_duration(profile.evaluation.window)
    watermark = parse_duration(profile.evaluation.watermark)

    last_evaluated_end = session.scalars(
        select(EvaluationWindow.window_end)
        .where(EvaluationWindow.model_id == model_id)
        .order_by(EvaluationWindow.window_end.desc())
        .limit(1)
    ).first()

    if last_evaluated_end is not None:
        range_start = last_evaluated_end
    else:
        earliest_prediction = session.scalars(
            select(Prediction.predicted_at)
            .where(Prediction.model_id == model_id)
            .order_by(Prediction.predicted_at.asc())
            .limit(1)
        ).first()
        if earliest_prediction is None:
            return []
        range_start = earliest_prediction

    candidate_windows = compute_window_boundaries(range_start, as_of, window_duration)

    already_evaluated = set(
        session.execute(
            select(EvaluationWindow.window_start, EvaluationWindow.window_end).where(
                EvaluationWindow.model_id == model_id,
                EvaluationWindow.window_start >= range_start,
            )
        ).all()
    )

    return [
        (window_start, window_end)
        for window_start, window_end in candidate_windows
        if (window_start, window_end) not in already_evaluated
        and is_drift_watermark_elapsed(window_end, watermark, as_of)
    ]


def evaluate_pending_windows(session: Session, as_of: datetime | None = None) -> int:
    """The scheduled job body: find every active model's drift-ready,
    unevaluated windows and evaluate them (drift + initial performance) via
    the exact same evaluate_window() the CLI backfill command calls. Commits
    after each window so a crash mid-batch only loses the window in flight,
    not the whole run's progress. Returns the number of windows evaluated.

    This is only half of performance's story: ongoing performance revision
    as labels backfill happens separately, triggered directly by the label
    ingestion endpoint (see driftwatch.api.routes.labels), independent of
    this job and unaffected by whether a window's drift has already run."""
    as_of = as_of or datetime.now(UTC)
    model_ids = session.scalars(select(Model.model_id).where(Model.is_active.is_(True))).all()
    channels = default_channels()

    evaluated_count = 0
    for model_id in model_ids:
        try:
            windows = find_drift_ready_unevaluated_windows(session, model_id, as_of)
        except Exception:
            logger.exception("failed to discover pending windows for model %s", model_id)
            continue

        for window_start, window_end in windows:
            try:
                result = evaluate_window(session, model_id, window_start, window_end)
            except Exception:
                session.rollback()
                logger.exception(
                    "failed to evaluate model %s window [%s, %s)",
                    model_id,
                    window_start,
                    window_end,
                )
                continue
            if result is not None:
                evaluated_count += 1
                _notify_touched_alerts(session, result.id, channels)
            session.commit()

    return evaluated_count


def _notify_touched_alerts(
    session: Session, window_id: int, channels: list[NotificationChannel]
) -> None:
    """Alerts created or updated while evaluating this window -- both drift
    and any performance alerts from the initial recompute triggered by
    evaluate_window -- get a chance to notify. notify_if_needed is what
    actually enforces the no-renotify-on-steady-state policy; this just
    finds the candidates."""
    touched = session.scalars(select(Alert).where(Alert.last_seen_window_id == window_id)).all()
    for alert in touched:
        notify_if_needed(alert, channels)
