from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.db.models import (
    DriftResult,
    EvaluationWindow,
    MetricStatus,
    PerformanceResult,
    TestMethod,
)

StreakKind = Literal["breach", "clear", "dead_zone", "not_computable"]


@dataclass(frozen=True)
class Streak:
    kind: StreakKind | None
    length: int


def classify_drift(
    result: DriftResult, fire_threshold: float, clear_threshold: float
) -> StreakKind:
    """Three-way classification for hysteresis, deliberately separate from
    DriftResult.is_significant (which is fire-threshold-only and unchanged --
    still used for that row's own informational display). A statistic
    between clear_threshold and fire_threshold is a "dead zone": it breaks
    whichever streak was building, the same as a value on the opposite side
    would, which is exactly what stops a boundary-hugging statistic from
    flapping an alert open and resolved every window.
    """
    if result.status == MetricStatus.NOT_COMPUTABLE:
        return "not_computable"
    statistic = result.statistic
    assert statistic is not None  # guaranteed by status == COMPUTED
    if statistic >= fire_threshold:
        return "breach"
    if statistic <= clear_threshold:
        return "clear"
    return "dead_zone"


def classify_performance(
    result: PerformanceResult,
    direction: Literal["below", "above"],
    fire_threshold: float,
    clear_threshold: float,
) -> StreakKind:
    """Same three-way classification as classify_drift, generalized over
    direction since a low value is bad for some metrics (precision, PR-AUC)
    and a high value is bad for others (RMSE, MAE)."""
    if result.status == MetricStatus.NOT_COMPUTABLE:
        return "not_computable"
    value = result.metric_value
    assert value is not None  # guaranteed by status == COMPUTED
    if direction == "below":
        if value <= fire_threshold:
            return "breach"
        if value >= clear_threshold:
            return "clear"
        return "dead_zone"
    if value >= fire_threshold:
        return "breach"
    if value <= clear_threshold:
        return "clear"
    return "dead_zone"


def current_streak(classifications_desc: Sequence[StreakKind]) -> Streak:
    """classifications_desc: most-recent-window-first. Returns how many of
    the most recent entries share the same classification, stopping at the
    first one that differs -- e.g. [breach, breach, clear, breach] has a
    current streak of (breach, 2), not 3, because the clear entry two windows
    back interrupts it. Empty input returns (None, 0): no history yet."""
    if not classifications_desc:
        return Streak(kind=None, length=0)
    current = classifications_desc[0]
    length = 0
    for classification in classifications_desc:
        if classification != current:
            break
        length += 1
    return Streak(kind=current, length=length)


def fetch_drift_history(
    session: Session,
    model_id: str,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str | None,
    segment_value: str | None,
    limit: int,
) -> list[DriftResult]:
    """Most recent `limit` DriftResult rows for this exact signal identity,
    most-recent-window-first. Drift never recomputes for a window once
    evaluated, so there's exactly one row per window here already -- no
    dedup needed, unlike fetch_performance_history."""
    query = (
        select(DriftResult)
        .join(EvaluationWindow, DriftResult.evaluation_window_id == EvaluationWindow.id)
        .where(
            EvaluationWindow.model_id == model_id,
            DriftResult.feature_name == feature_name,
            DriftResult.test_method == test_method,
            DriftResult.segment_dimension == segment_dimension,
            DriftResult.segment_value == segment_value,
        )
        .order_by(EvaluationWindow.window_end.desc())
        .limit(limit)
    )
    return list(session.scalars(query))


def fetch_performance_history(
    session: Session,
    model_id: str,
    metric_name: str,
    segment_dimension: str | None,
    segment_value: str | None,
    limit: int,
) -> list[PerformanceResult]:
    """Most recent `limit` PerformanceResult rows for this signal, ONE per
    window -- the latest computed_at for that window. A window that
    recomputed 10 times after label backfill contributes exactly one entry
    here, not ten: a window whose metrics degrade after labels arrive should
    alert, but must not re-alert on every subsequent recompute of the same
    window (the streak is over windows, not over recompute events)."""
    query = (
        select(PerformanceResult)
        .join(EvaluationWindow, PerformanceResult.evaluation_window_id == EvaluationWindow.id)
        .where(
            EvaluationWindow.model_id == model_id,
            PerformanceResult.metric_name == metric_name,
            PerformanceResult.segment_dimension == segment_dimension,
            PerformanceResult.segment_value == segment_value,
        )
        .order_by(EvaluationWindow.window_end.desc(), PerformanceResult.computed_at.desc())
    )
    latest_per_window: dict[int, PerformanceResult] = {}
    window_order: list[int] = []
    for result in session.scalars(query):
        window_id = result.evaluation_window_id
        if window_id in latest_per_window:
            continue
        latest_per_window[window_id] = result
        window_order.append(window_id)
        if len(window_order) >= limit:
            break
    return [latest_per_window[window_id] for window_id in window_order]
