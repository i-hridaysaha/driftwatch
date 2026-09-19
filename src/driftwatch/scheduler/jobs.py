import logging
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session

from driftwatch.alerting.notifications import (
    MAX_NOTIFICATION_ATTEMPTS,
    NotificationChannel,
    default_channels,
    notify_if_needed,
)
from driftwatch.config.loader import load_model_config, load_profile
from driftwatch.db.models import Alert, EvaluationWindow, Model, Prediction
from driftwatch.durations import parse_duration
from driftwatch.evaluation.drift import evaluate_window
from driftwatch.evaluation.performance import recompute_performance_for_window
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
    as labels backfill is recompute_stale_windows below, driven by the
    performance_stale flag the label ingestion endpoint sets, independent of
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


def recompute_stale_windows(session: Session, channels: list[NotificationChannel]) -> int:
    """The other half of performance: every window label ingestion has
    flagged since the last tick gets its performance recomputed, its
    performance alerts re-evaluated, and any transition notified -- here,
    in the scheduler process, never in the request that carried the labels.

    Claiming is an UPDATE ... WHERE performance_stale, so the row lock makes
    it exclusive and a batch that lands during the recompute simply sets the
    flag again for the next tick. Commits per window, like
    evaluate_pending_windows, so a failure loses one window, not the batch.
    Returns the number of windows recomputed."""
    stale_ids = session.scalars(
        select(EvaluationWindow.id)
        .where(EvaluationWindow.performance_stale.is_(True))
        .order_by(EvaluationWindow.id)
    ).all()
    recomputed = 0
    for window_id in stale_ids:
        claim = session.execute(
            update(EvaluationWindow)
            .where(EvaluationWindow.id == window_id, EvaluationWindow.performance_stale.is_(True))
            .values(performance_stale=False)
        )
        if not cast(CursorResult[Any], claim).rowcount:
            continue  # another scheduler took it, or it was already cleared
        try:
            recompute_performance_for_window(session, window_id)
        except Exception:
            session.rollback()  # the claim rolls back too, so it is retried next tick
            logger.exception("failed to recompute performance for window %s", window_id)
            continue
        recomputed += 1
        _notify_touched_alerts(session, window_id, channels)
        session.commit()
    return recomputed


def retry_pending_notifications(session: Session, channels: list[NotificationChannel]) -> int:
    """Alerts whose last delivered status is behind their real status and
    that have not exhausted MAX_NOTIFICATION_ATTEMPTS get another delivery
    round. A transition whose webhook was down when it happened is therefore
    delivered once the webhook is back, on a later tick, rather than logged
    once and forgotten. Returns the number of alerts retried."""
    pending = session.scalars(
        select(Alert).where(
            Alert.last_notified_status.is_distinct_from(Alert.status),
            Alert.notification_attempts < MAX_NOTIFICATION_ATTEMPTS,
        )
    ).all()
    for alert in pending:
        notify_if_needed(alert, channels)
    if pending:
        session.commit()
    return len(pending)


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
