from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.alerting.engine import (
    evaluate_drift_alerts_for_window,
    evaluate_performance_alerts_for_window,
)
from driftwatch.config.schema import Profile
from driftwatch.db.models import (
    Alert,
    AlertKind,
    AlertStatus,
    Baseline,
    DriftResult,
    EvaluationWindow,
    MetricStatus,
    Model,
    PerformanceResult,
    TestMethod,
)

MODEL_ID = "example-model"


def _make_profile(
    *,
    fire: int = 2,
    escalate: int = 4,
    resolve: int = 3,
    not_computable: int = 2,
    max_alerts_per_run: int = 3,
    psi_threshold: float = 0.2,
    psi_clear_threshold: float = 0.1,
    history_scan_windows: int | None = None,
) -> Profile:
    if history_scan_windows is None:
        # margin of 2 beyond the largest persistence count, matching
        # AlertingConfig's own validated minimum -- see config/schema.py
        history_scan_windows = max(fire, escalate, resolve, not_computable) + 2
    return Profile.model_validate(
        {
            "name": "test",
            "description": "test profile",
            "evaluation": {"window": "1h", "min_window_size": 1, "watermark": "1m"},
            "drift_tests": {
                "continuous": {
                    "methods": ["psi", "ks"],
                    "psi_threshold": psi_threshold,
                    "psi_clear_threshold": psi_clear_threshold,
                    "ks_statistic_threshold": 0.2,
                    "ks_statistic_clear_threshold": 0.1,
                },
                "categorical": {
                    "methods": ["chi_square"],
                    "cramers_v_threshold": 0.2,
                    "cramers_v_clear_threshold": 0.1,
                },
                "prediction_score": {
                    "method": "jsd",
                    "jsd_threshold": 0.2,
                    "jsd_clear_threshold": 0.1,
                },
            },
            "multiple_comparison_correction": {"method": "benjamini_hochberg", "fdr_alpha": 0.05},
            "alerting": {
                "fire_persistence_windows": fire,
                "escalate_persistence_windows": escalate,
                "resolve_persistence_windows": resolve,
                "not_computable_persistence_windows": not_computable,
                "history_scan_windows": history_scan_windows,
                "max_alerts_per_run": max_alerts_per_run,
            },
            "performance_metrics": [
                {
                    "name": "pr_auc",
                    "alert_direction": "below",
                    "fire_threshold": 0.5,
                    "clear_threshold": 0.6,
                }
            ],
        }
    )


def _setup_model(db_session: Session) -> Baseline:
    db_session.add(Model(model_id=MODEL_ID))
    db_session.flush()
    baseline = Baseline(model_id=MODEL_ID, is_active=True, binning_config={})
    db_session.add(baseline)
    db_session.flush()
    return baseline


def _make_window(
    db_session: Session, baseline: Baseline, index: int, config_hash: str = "hash"
) -> EvaluationWindow:
    window_start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    window = EvaluationWindow(
        model_id=MODEL_ID,
        baseline_id=baseline.id,
        window_start=window_start,
        window_end=window_start + timedelta(hours=1),
        config_hash=config_hash,
        evaluated_at=datetime.now(UTC),
    )
    db_session.add(window)
    db_session.flush()
    return window


def _add_drift_result(
    db_session: Session,
    window: EvaluationWindow,
    *,
    feature_name: str = "age",
    statistic: float | None,
    status: MetricStatus = MetricStatus.COMPUTED,
    not_computable_reason: str | None = None,
    segment_dimension: str | None = None,
    segment_value: str | None = None,
) -> DriftResult:
    result = DriftResult(
        evaluation_window_id=window.id,
        feature_name=feature_name,
        test_method=TestMethod.PSI,
        segment_dimension=segment_dimension,
        segment_value=segment_value,
        status=status,
        statistic=statistic,
        is_significant=None,
        not_computable_reason=not_computable_reason,
        n_baseline=10,
        n_live=10,
    )
    db_session.add(result)
    db_session.flush()
    return result


def _add_performance_result(
    db_session: Session,
    window: EvaluationWindow,
    *,
    value: float | None,
    status: MetricStatus = MetricStatus.COMPUTED,
    is_retroactive: bool = False,
) -> PerformanceResult:
    result = PerformanceResult(
        evaluation_window_id=window.id,
        metric_name="pr_auc",
        status=status,
        metric_value=value,
        n_labeled=10,
        is_retroactive=is_retroactive,
    )
    db_session.add(result)
    db_session.flush()
    return result


def _active_alert(db_session: Session, kind: AlertKind, signal_name: str = "psi") -> Alert | None:
    return db_session.scalars(
        select(Alert).where(
            Alert.model_id == MODEL_ID,
            Alert.kind == kind,
            Alert.signal_name == signal_name,
            Alert.status != AlertStatus.RESOLVED,
            Alert.is_aggregate.is_(False),
        )
    ).first()


def test_no_alert_below_fire_persistence(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=2)

    window = _make_window(db_session, baseline, 0)
    _add_drift_result(db_session, window, statistic=0.5)  # one breach, fire needs 2
    evaluate_drift_alerts_for_window(db_session, window, profile)

    assert _active_alert(db_session, AlertKind.DRIFT) is None


def test_alert_opens_at_fire_persistence(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=2)

    for i in range(2):
        window = _make_window(db_session, baseline, i)
        _add_drift_result(db_session, window, statistic=0.5)
        evaluate_drift_alerts_for_window(db_session, window, profile)

    alert = _active_alert(db_session, AlertKind.DRIFT)
    assert alert is not None
    assert alert.status == AlertStatus.OPEN
    assert alert.evidence_statistic == 0.5
    assert alert.evidence_threshold == 0.2
    assert alert.evidence_baseline_id == baseline.id
    assert alert.evidence_window_id == window.id


def test_sustained_breach_is_one_alert_with_updated_last_seen(db_session: Session) -> None:
    """20 consecutive breaching windows must be ONE open alert, not 20."""
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=2)

    last_window = None
    for i in range(20):
        window = _make_window(db_session, baseline, i)
        _add_drift_result(db_session, window, statistic=0.5)
        evaluate_drift_alerts_for_window(db_session, window, profile)
        last_window = window

    all_alerts = db_session.scalars(
        select(Alert).where(Alert.model_id == MODEL_ID, Alert.kind == AlertKind.DRIFT)
    ).all()
    assert len(all_alerts) == 1
    assert all_alerts[0].last_seen_window_id == last_window.id


def test_escalates_at_escalate_persistence(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=2, escalate=4)

    for i in range(3):
        window = _make_window(db_session, baseline, i)
        _add_drift_result(db_session, window, statistic=0.5)
        evaluate_drift_alerts_for_window(db_session, window, profile)

    alert = _active_alert(db_session, AlertKind.DRIFT)
    assert alert is not None
    assert alert.status == AlertStatus.OPEN  # 3 breaches, escalate needs 4

    window = _make_window(db_session, baseline, 3)
    _add_drift_result(db_session, window, statistic=0.5)
    evaluate_drift_alerts_for_window(db_session, window, profile)

    alert = _active_alert(db_session, AlertKind.DRIFT)
    assert alert is not None
    assert alert.status == AlertStatus.ESCALATED
    assert alert.escalated_at is not None


def test_escalation_evidence_is_a_separate_snapshot_from_open_evidence(
    db_session: Session,
) -> None:
    """An alert that opened at 0.5 and escalated at 0.9 should be able to
    say so -- escalation gets its own frozen-once snapshot, distinct from
    (and not overwriting) the open-time evidence."""
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=1, escalate=3)

    w0 = _make_window(db_session, baseline, 0, config_hash="hash-v1")
    _add_drift_result(db_session, w0, statistic=0.5)
    evaluate_drift_alerts_for_window(db_session, w0, profile)

    alert = _active_alert(db_session, AlertKind.DRIFT)
    assert alert is not None
    assert alert.status == AlertStatus.OPEN
    assert alert.escalation_evidence_statistic is None  # not escalated yet

    w1 = _make_window(db_session, baseline, 1, config_hash="hash-v1")
    _add_drift_result(db_session, w1, statistic=0.6)
    evaluate_drift_alerts_for_window(db_session, w1, profile)
    assert _active_alert(db_session, AlertKind.DRIFT).status == AlertStatus.OPEN  # type: ignore[union-attr]

    w2 = _make_window(db_session, baseline, 2, config_hash="hash-v2")
    _add_drift_result(db_session, w2, statistic=0.9)
    evaluate_drift_alerts_for_window(db_session, w2, profile)

    escalated = _active_alert(db_session, AlertKind.DRIFT)
    assert escalated is not None
    assert escalated.status == AlertStatus.ESCALATED
    # open-time evidence is untouched
    assert escalated.evidence_statistic == 0.5
    assert escalated.evidence_config_hash == "hash-v1"
    # escalation-time evidence reflects the window that actually triggered it
    assert escalated.escalation_evidence_statistic == 0.9
    assert escalated.escalation_evidence_config_hash == "hash-v2"
    assert escalated.escalation_evidence_window_id == w2.id
    assert escalated.escalation_evidence_baseline_id == baseline.id


def test_hysteresis_dead_zone_does_not_flap(db_session: Session) -> None:
    """A statistic hovering right at the fire threshold, alternating with the
    dead zone between clear and fire, must never accumulate enough streak to
    open -- this is exactly what hysteresis exists to prevent."""
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=2, psi_threshold=0.2, psi_clear_threshold=0.1)

    # 0.2 = breach, 0.15 = dead zone (between 0.1 and 0.2) -- alternating
    values = [0.2, 0.15, 0.2, 0.15, 0.2, 0.15]
    for i, value in enumerate(values):
        window = _make_window(db_session, baseline, i)
        _add_drift_result(db_session, window, statistic=value)
        evaluate_drift_alerts_for_window(db_session, window, profile)

    assert _active_alert(db_session, AlertKind.DRIFT) is None


def test_resolves_after_clear_persistence(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=1, resolve=2, psi_threshold=0.2, psi_clear_threshold=0.1)

    window = _make_window(db_session, baseline, 0)
    _add_drift_result(db_session, window, statistic=0.5)
    evaluate_drift_alerts_for_window(db_session, window, profile)
    assert _active_alert(db_session, AlertKind.DRIFT).status == AlertStatus.OPEN  # type: ignore[union-attr]

    window = _make_window(db_session, baseline, 1)
    _add_drift_result(db_session, window, statistic=0.05)  # clear
    evaluate_drift_alerts_for_window(db_session, window, profile)
    assert _active_alert(db_session, AlertKind.DRIFT) is not None  # only 1 clear window so far

    window = _make_window(db_session, baseline, 2)
    _add_drift_result(db_session, window, statistic=0.05)  # 2nd consecutive clear
    evaluate_drift_alerts_for_window(db_session, window, profile)
    assert _active_alert(db_session, AlertKind.DRIFT) is None

    resolved = db_session.scalars(
        select(Alert).where(Alert.model_id == MODEL_ID, Alert.kind == AlertKind.DRIFT)
    ).one()
    assert resolved.status == AlertStatus.RESOLVED
    assert resolved.resolved_at is not None
    # evidence from the ORIGINAL fire is preserved, not overwritten by the clear values
    assert resolved.evidence_statistic == 0.5


def test_reopening_after_resolution_is_a_new_row(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=1, resolve=1, psi_threshold=0.2, psi_clear_threshold=0.1)

    w0 = _make_window(db_session, baseline, 0)
    _add_drift_result(db_session, w0, statistic=0.5)
    evaluate_drift_alerts_for_window(db_session, w0, profile)

    w1 = _make_window(db_session, baseline, 1)
    _add_drift_result(db_session, w1, statistic=0.05)
    evaluate_drift_alerts_for_window(db_session, w1, profile)  # resolves

    w2 = _make_window(db_session, baseline, 2)
    _add_drift_result(db_session, w2, statistic=0.5)
    evaluate_drift_alerts_for_window(db_session, w2, profile)  # fires again

    all_rows = db_session.scalars(
        select(Alert).where(Alert.model_id == MODEL_ID, Alert.kind == AlertKind.DRIFT)
    ).all()
    assert len(all_rows) == 2
    assert {row.status for row in all_rows} == {AlertStatus.RESOLVED, AlertStatus.OPEN}


def test_not_computable_raises_its_own_alert_type(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(not_computable=2)

    for i in range(2):
        window = _make_window(db_session, baseline, i)
        _add_drift_result(
            db_session,
            window,
            statistic=None,
            status=MetricStatus.NOT_COMPUTABLE,
            not_computable_reason="window has 3 predictions, below configured minimum of 50",
        )
        evaluate_drift_alerts_for_window(db_session, window, profile)

    nc_alert = _active_alert(db_session, AlertKind.NOT_COMPUTABLE)
    assert nc_alert is not None
    assert nc_alert.evidence_reason == "window has 3 predictions, below configured minimum of 50"
    # a not_computable streak must never be mistaken for "no drift" -- confirm
    # no DRIFT-kind alert exists for the same signal
    assert _active_alert(db_session, AlertKind.DRIFT) is None


def test_not_computable_resolves_immediately_once_computable_again(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(not_computable=2)

    for i in range(2):
        window = _make_window(db_session, baseline, i)
        _add_drift_result(
            db_session, window, statistic=None, status=MetricStatus.NOT_COMPUTABLE,
            not_computable_reason="broken",
        )
        evaluate_drift_alerts_for_window(db_session, window, profile)
    assert _active_alert(db_session, AlertKind.NOT_COMPUTABLE) is not None

    window = _make_window(db_session, baseline, 2)
    _add_drift_result(db_session, window, statistic=0.05)  # computable again, clear value
    evaluate_drift_alerts_for_window(db_session, window, profile)

    assert _active_alert(db_session, AlertKind.NOT_COMPUTABLE) is None


def test_aggregation_caps_new_opens_into_one_rollup(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=1, max_alerts_per_run=3)

    window = _make_window(db_session, baseline, 0)
    # 5 different segments all breach in the same run -- cap is 3
    for i in range(5):
        _add_drift_result(
            db_session,
            window,
            feature_name="age",
            statistic=0.9,
            segment_dimension="region",
            segment_value=f"region-{i}",
        )
    evaluate_drift_alerts_for_window(db_session, window, profile)

    individual_alerts = db_session.scalars(
        select(Alert).where(
            Alert.model_id == MODEL_ID, Alert.kind == AlertKind.DRIFT, Alert.is_aggregate.is_(False)
        )
    ).all()
    aggregates = db_session.scalars(
        select(Alert).where(
            Alert.model_id == MODEL_ID, Alert.kind == AlertKind.DRIFT, Alert.is_aggregate.is_(True)
        )
    ).all()

    assert individual_alerts == []
    assert len(aggregates) == 1
    assert aggregates[0].aggregated_signal_count == 5
    assert aggregates[0].aggregated_signals is not None
    assert len(aggregates[0].aggregated_signals) == 5


def test_aggregate_resolves_when_back_under_cap(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=1, max_alerts_per_run=3)

    window = _make_window(db_session, baseline, 0)
    for i in range(5):
        _add_drift_result(
            db_session, window, statistic=0.9, segment_dimension="region", segment_value=f"r{i}"
        )
    evaluate_drift_alerts_for_window(db_session, window, profile)
    aggregate = db_session.scalars(
        select(Alert).where(Alert.model_id == MODEL_ID, Alert.is_aggregate.is_(True))
    ).one()
    assert aggregate.status == AlertStatus.OPEN

    # next window: nothing new breaches (all those signals already have no
    # active alert since they were rolled up and deleted, so nothing re-fires
    # from a fresh single breach -- fire=1 means a single new breach WOULD
    # open, so use zero new breaches here to simulate the overflow subsiding)
    window2 = _make_window(db_session, baseline, 1)
    evaluate_drift_alerts_for_window(db_session, window2, profile)

    resolved_aggregate = db_session.get(Alert, aggregate.id)
    assert resolved_aggregate is not None
    assert resolved_aggregate.status == AlertStatus.RESOLVED


def test_performance_alert_does_not_re_alert_on_retroactive_recompute(db_session: Session) -> None:
    """Constraint 5: a window whose metrics degrade after labels arrive
    should alert, but must not re-alert on every subsequent recompute of the
    same window. Simulated here by inserting multiple PerformanceResult rows
    for the SAME window (as recompute_performance_for_window would on
    repeated label backfill) and confirming the streak counts it once."""
    baseline = _setup_model(db_session)
    profile = _make_profile(fire=2)

    window = _make_window(db_session, baseline, 0)
    _add_performance_result(db_session, window, value=0.3, is_retroactive=False)
    evaluate_performance_alerts_for_window(db_session, window, profile)
    # only 1 window's worth of breach so far -- fire needs 2
    assert _active_alert(db_session, AlertKind.PERFORMANCE, "pr_auc") is None

    # same window recomputes again (e.g. more labels backfilled) -- still degraded
    _add_performance_result(db_session, window, value=0.31, is_retroactive=True)
    evaluate_performance_alerts_for_window(db_session, window, profile)
    _add_performance_result(db_session, window, value=0.32, is_retroactive=True)
    evaluate_performance_alerts_for_window(db_session, window, profile)
    _add_performance_result(db_session, window, value=0.33, is_retroactive=True)
    evaluate_performance_alerts_for_window(db_session, window, profile)

    # despite 4 recomputes of the SAME window, the streak is still just 1
    # window's worth of breach -- fire needs 2, so still no alert
    assert _active_alert(db_session, AlertKind.PERFORMANCE, "pr_auc") is None

    window2 = _make_window(db_session, baseline, 1)
    _add_performance_result(db_session, window2, value=0.3)
    evaluate_performance_alerts_for_window(db_session, window2, profile)

    alert = _active_alert(db_session, AlertKind.PERFORMANCE, "pr_auc")
    assert alert is not None
    assert alert.status == AlertStatus.OPEN


def test_evidence_survives_config_change(db_session: Session) -> None:
    """Constraint 4: if config changes later, the alert must still explain
    why it fired under the rules in effect then."""
    baseline = _setup_model(db_session)
    original_profile = _make_profile(fire=1, psi_threshold=0.2)

    window = _make_window(db_session, baseline, 0, config_hash="hash-v1")
    _add_drift_result(db_session, window, statistic=0.5)
    evaluate_drift_alerts_for_window(db_session, window, original_profile)

    alert = _active_alert(db_session, AlertKind.DRIFT)
    assert alert is not None
    assert alert.evidence_statistic == 0.5
    assert alert.evidence_threshold == 0.2
    assert alert.evidence_config_hash == "hash-v1"

    # now the config changes (different threshold) -- re-evaluate a later window
    # under the NEW config; the alert's frozen evidence must be untouched
    new_profile = _make_profile(fire=1, psi_threshold=0.35)
    window2 = _make_window(db_session, baseline, 1, config_hash="hash-v2")
    _add_drift_result(db_session, window2, statistic=0.4)  # breach under old rules, not new
    evaluate_drift_alerts_for_window(db_session, window2, new_profile)

    alert_after = db_session.get(Alert, alert.id)
    assert alert_after is not None
    assert alert_after.evidence_statistic == 0.5  # unchanged
    assert alert_after.evidence_threshold == 0.2  # unchanged
    assert alert_after.evidence_config_hash == "hash-v1"  # unchanged
