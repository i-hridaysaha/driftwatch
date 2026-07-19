from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.alerting.streaks import (
    Streak,
    classify_drift,
    classify_performance,
    current_streak,
    fetch_drift_history,
    fetch_performance_history,
)
from driftwatch.config.schema import AlertingConfig, Profile
from driftwatch.db.models import (
    Alert,
    AlertKind,
    AlertStatus,
    DriftResult,
    EvaluationWindow,
    PerformanceResult,
    TestMethod,
)


def _history_scan_limit(alerting: AlertingConfig) -> int:
    """The rescan bound is an explicit, validated config value (see
    AlertingConfig.history_scan_windows), not derived from the persistence
    counts here -- that would make the "does history cover what persistence
    requires" invariant tautologically true and unable to ever fail."""
    return alerting.history_scan_windows


def _drift_thresholds(profile: Profile, test_method: TestMethod) -> tuple[float, float]:
    continuous = profile.drift_tests.continuous
    categorical = profile.drift_tests.categorical
    prediction_score = profile.drift_tests.prediction_score
    if test_method == TestMethod.PSI:
        return continuous.psi_threshold, continuous.psi_clear_threshold
    if test_method == TestMethod.KS:
        return continuous.ks_statistic_threshold, continuous.ks_statistic_clear_threshold
    if test_method == TestMethod.CHI_SQUARE:
        return categorical.cramers_v_threshold, categorical.cramers_v_clear_threshold
    return prediction_score.jsd_threshold, prediction_score.jsd_clear_threshold


def _signal_identity(alert: Alert) -> dict[str, Any]:
    return {
        "signal_name": alert.signal_name,
        "feature_name": alert.feature_name,
        "segment_dimension": alert.segment_dimension,
        "segment_value": alert.segment_value,
    }


def _apply_transition(
    session: Session,
    *,
    model_id: str,
    kind: AlertKind,
    signal_name: str,
    feature_name: str | None,
    segment_dimension: str | None,
    segment_value: str | None,
    streak: Streak,
    alerting: AlertingConfig,
    window: EvaluationWindow,
    evidence_statistic: float | None,
    evidence_threshold: float | None,
    evidence_reason: str | None,
) -> tuple[Alert, bool] | None:
    """The open/escalate/resolve state machine for one signal identity,
    given its freshly computed streak. Returns (alert, is_newly_opened) if
    a row was created or updated, else None if nothing changed (streak
    hasn't reached fire persistence yet and no alert exists -- no PENDING
    row is created; the streak is simply recomputed from
    DriftResult/PerformanceResult history again next time)."""
    existing = session.scalars(
        select(Alert).where(
            Alert.model_id == model_id,
            Alert.kind == kind,
            Alert.signal_name == signal_name,
            Alert.feature_name == feature_name,
            Alert.segment_dimension == segment_dimension,
            Alert.segment_value == segment_value,
            Alert.status != AlertStatus.RESOLVED,
            Alert.is_aggregate.is_(False),
        )
    ).first()

    if kind == AlertKind.NOT_COMPUTABLE:
        fire_persistence = alerting.not_computable_persistence_windows
        is_firing = streak.kind == "not_computable"
    else:
        fire_persistence = alerting.fire_persistence_windows
        is_firing = streak.kind == "breach"

    if existing is None:
        if is_firing and streak.length >= fire_persistence:
            alert = Alert(
                model_id=model_id,
                kind=kind,
                signal_name=signal_name,
                feature_name=feature_name,
                segment_dimension=segment_dimension,
                segment_value=segment_value,
                status=AlertStatus.OPEN,
                consecutive_breaching_windows=streak.length
                if kind != AlertKind.NOT_COMPUTABLE
                else 0,
                consecutive_clear_windows=0,
                evidence_statistic=evidence_statistic,
                evidence_threshold=evidence_threshold,
                evidence_baseline_id=window.baseline_id,
                evidence_config_hash=window.config_hash,
                evidence_window_id=window.id,
                evidence_reason=evidence_reason,
                last_seen_window_id=window.id,
            )
            session.add(alert)
            session.flush()
            return alert, True
        return None

    existing.last_seen_window_id = window.id
    existing.last_seen_at = datetime.now(UTC)

    if kind == AlertKind.NOT_COMPUTABLE:
        if streak.kind != "not_computable":
            # computable again -- resolve immediately, no clear-persistence needed:
            # unlike a continuous statistic near a threshold, "did it compute" is
            # a clean binary signal with nothing to flap around.
            existing.status = AlertStatus.RESOLVED
            existing.resolved_at = datetime.now(UTC)
        return existing, False

    if streak.kind == "breach":
        existing.consecutive_breaching_windows = streak.length
        existing.consecutive_clear_windows = 0
        if (
            existing.status == AlertStatus.OPEN
            and streak.length >= alerting.escalate_persistence_windows
        ):
            existing.status = AlertStatus.ESCALATED
            existing.escalated_at = datetime.now(UTC)
            # frozen once, here, same discipline as the open-time evidence --
            # an alert that opened at 0.11 and escalated at 0.42 should still
            # be able to say so, not just that it escalated at some point.
            existing.escalation_evidence_statistic = evidence_statistic
            existing.escalation_evidence_threshold = evidence_threshold
            existing.escalation_evidence_baseline_id = window.baseline_id
            existing.escalation_evidence_config_hash = window.config_hash
            existing.escalation_evidence_window_id = window.id
    elif streak.kind == "clear":
        existing.consecutive_clear_windows = streak.length
        existing.consecutive_breaching_windows = 0
        if streak.length >= alerting.resolve_persistence_windows:
            existing.status = AlertStatus.RESOLVED
            existing.resolved_at = datetime.now(UTC)
    elif streak.kind == "dead_zone":
        existing.consecutive_breaching_windows = 0
        existing.consecutive_clear_windows = 0
    # streak.kind == "not_computable" on a drift/performance-kind alert: the
    # underlying test/metric can't be evaluated right now, so this alert is
    # left exactly as-is (paused, not cleared) until it can be classified
    # breach/clear/dead_zone again.

    return existing, False


def _reconcile_aggregation(
    session: Session,
    window: EvaluationWindow,
    kind: AlertKind,
    new_opens: list[Alert],
    max_alerts_per_run: int,
) -> None:
    """If more than max_alerts_per_run NEW alerts opened for `kind` in this
    run, delete the individual rows and roll them into one aggregate row
    instead (creating or extending it). If this run's new-open count is back
    at or under the cap and an aggregate is still open, resolve it -- the
    overflow condition that justified it is no longer true. Known scope
    limit: NOT_COMPUTABLE aggregation doesn't distinguish whether the
    overflow came from drift tests or performance metrics, since both share
    that one AlertKind; a simultaneous overflow from both in the same run is
    rare enough not to warrant a further identity split here."""
    existing_aggregate = session.scalars(
        select(Alert).where(
            Alert.model_id == window.model_id,
            Alert.kind == kind,
            Alert.is_aggregate.is_(True),
            Alert.status != AlertStatus.RESOLVED,
        )
    ).first()

    if len(new_opens) <= max_alerts_per_run:
        if existing_aggregate is not None:
            existing_aggregate.status = AlertStatus.RESOLVED
            existing_aggregate.resolved_at = datetime.now(UTC)
        return

    identities = [_signal_identity(alert) for alert in new_opens]
    for alert in new_opens:
        session.delete(alert)
    session.flush()

    if existing_aggregate is not None:
        existing_aggregate.aggregated_signal_count = len(new_opens)
        existing_aggregate.aggregated_signals = identities
        existing_aggregate.last_seen_window_id = window.id
        existing_aggregate.last_seen_at = datetime.now(UTC)
        return

    aggregate = Alert(
        model_id=window.model_id,
        kind=kind,
        signal_name=f"aggregate-{kind.value}",
        status=AlertStatus.OPEN,
        evidence_window_id=window.id,
        evidence_baseline_id=window.baseline_id,
        evidence_config_hash=window.config_hash,
        evidence_reason=(
            f"{len(new_opens)} signals opened in one evaluation run, rolled up "
            f"(cap={max_alerts_per_run})"
        ),
        last_seen_window_id=window.id,
        is_aggregate=True,
        aggregated_signal_count=len(new_opens),
        aggregated_signals=identities,
    )
    session.add(aggregate)
    session.flush()


def evaluate_drift_alerts_for_window(
    session: Session, window: EvaluationWindow, profile: Profile
) -> list[Alert]:
    """Called once per evaluated window (from evaluate_window, after
    drift_results are written): updates DRIFT and NOT_COMPUTABLE alerts for
    every (feature, test_method, segment) signal touched by this window.
    Returns every Alert row created or updated this run."""
    alerting = profile.alerting
    max_history = _history_scan_limit(alerting)

    results = session.scalars(
        select(DriftResult).where(DriftResult.evaluation_window_id == window.id)
    ).all()

    touched: list[Alert] = []
    new_drift_opens: list[Alert] = []
    new_not_computable_opens: list[Alert] = []

    for result in results:
        fire_threshold, clear_threshold = _drift_thresholds(profile, result.test_method)
        history = fetch_drift_history(
            session,
            window.model_id,
            result.feature_name,
            result.test_method,
            result.segment_dimension,
            result.segment_value,
            max_history,
        )
        streak = current_streak(
            [classify_drift(r, fire_threshold, clear_threshold) for r in history]
        )

        drift_outcome = _apply_transition(
            session,
            model_id=window.model_id,
            kind=AlertKind.DRIFT,
            signal_name=result.test_method.value,
            feature_name=result.feature_name,
            segment_dimension=result.segment_dimension,
            segment_value=result.segment_value,
            streak=streak,
            alerting=alerting,
            window=window,
            evidence_statistic=result.statistic,
            evidence_threshold=fire_threshold,
            evidence_reason=None,
        )
        if drift_outcome is not None:
            alert, is_new = drift_outcome
            touched.append(alert)
            if is_new:
                new_drift_opens.append(alert)

        nc_outcome = _apply_transition(
            session,
            model_id=window.model_id,
            kind=AlertKind.NOT_COMPUTABLE,
            signal_name=result.test_method.value,
            feature_name=result.feature_name,
            segment_dimension=result.segment_dimension,
            segment_value=result.segment_value,
            streak=streak,
            alerting=alerting,
            window=window,
            evidence_statistic=None,
            evidence_threshold=None,
            evidence_reason=result.not_computable_reason,
        )
        if nc_outcome is not None:
            alert, is_new = nc_outcome
            touched.append(alert)
            if is_new:
                new_not_computable_opens.append(alert)

    _reconcile_aggregation(
        session, window, AlertKind.DRIFT, new_drift_opens, alerting.max_alerts_per_run
    )
    _reconcile_aggregation(
        session,
        window,
        AlertKind.NOT_COMPUTABLE,
        new_not_computable_opens,
        alerting.max_alerts_per_run,
    )

    return touched


def evaluate_performance_alerts_for_window(
    session: Session, window: EvaluationWindow, profile: Profile
) -> list[Alert]:
    """Called every time recompute_performance_for_window runs for this
    window -- at initial evaluation and every later retroactive recompute
    triggered by label backfill. Only metrics with alert_direction configured
    (see MetricSpec) get an alert lifecycle; the rest are informational only.
    Returns every Alert row created or updated this run.

    Constraint 5 (must not re-alert on every recompute of the same window)
    is satisfied by fetch_performance_history deduplicating to the latest
    result per window before the streak is computed -- ten recomputes of one
    window contribute one entry to the streak, not ten."""
    alerting = profile.alerting
    max_history = _history_scan_limit(alerting)

    metric_configs = {
        spec.name: spec for spec in profile.performance_metrics if spec.alert_direction is not None
    }
    if not metric_configs:
        return []

    results = session.scalars(
        select(PerformanceResult).where(
            PerformanceResult.evaluation_window_id == window.id,
            PerformanceResult.metric_name.in_(metric_configs.keys()),
        )
    ).all()

    touched: list[Alert] = []
    new_performance_opens: list[Alert] = []
    new_not_computable_opens: list[Alert] = []

    for result in results:
        spec = metric_configs[result.metric_name]
        assert spec.alert_direction is not None
        assert spec.fire_threshold is not None
        assert spec.clear_threshold is not None
        history = fetch_performance_history(
            session,
            window.model_id,
            result.metric_name,
            result.segment_dimension,
            result.segment_value,
            max_history,
        )
        streak = current_streak(
            [
                classify_performance(
                    r, spec.alert_direction, spec.fire_threshold, spec.clear_threshold
                )
                for r in history
            ]
        )

        performance_outcome = _apply_transition(
            session,
            model_id=window.model_id,
            kind=AlertKind.PERFORMANCE,
            signal_name=result.metric_name,
            feature_name=None,
            segment_dimension=result.segment_dimension,
            segment_value=result.segment_value,
            streak=streak,
            alerting=alerting,
            window=window,
            evidence_statistic=result.metric_value,
            evidence_threshold=spec.fire_threshold,
            evidence_reason=None,
        )
        if performance_outcome is not None:
            alert, is_new = performance_outcome
            touched.append(alert)
            if is_new:
                new_performance_opens.append(alert)

        nc_outcome = _apply_transition(
            session,
            model_id=window.model_id,
            kind=AlertKind.NOT_COMPUTABLE,
            signal_name=result.metric_name,
            feature_name=None,
            segment_dimension=result.segment_dimension,
            segment_value=result.segment_value,
            streak=streak,
            alerting=alerting,
            window=window,
            evidence_statistic=None,
            evidence_threshold=None,
            evidence_reason=result.not_computable_reason,
        )
        if nc_outcome is not None:
            alert, is_new = nc_outcome
            touched.append(alert)
            if is_new:
                new_not_computable_opens.append(alert)

    _reconcile_aggregation(
        session, window, AlertKind.PERFORMANCE, new_performance_opens, alerting.max_alerts_per_run
    )
    _reconcile_aggregation(
        session,
        window,
        AlertKind.NOT_COMPUTABLE,
        new_not_computable_opens,
        alerting.max_alerts_per_run,
    )

    return touched
