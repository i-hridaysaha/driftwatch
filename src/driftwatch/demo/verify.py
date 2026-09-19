"""Post-generation verification: asserts a loaded demo scenario produced
what its own YAML claims, entirely by reading what a real evaluation run
wrote to the database -- never by asserting against hand-picked numbers.
Run via `driftwatch demo-verify <scenario>` after `driftwatch demo
<scenario>`, and in CI for every shipped scenario (see constraint 10: a
scenario that doesn't produce its claimed signal gets its config changed
and regenerated, never its results edited).
"""

import statistics
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from driftwatch.alerting.engine import _drift_thresholds
from driftwatch.alerting.reporting import SuppressedEpisode, find_suppressed_episodes
from driftwatch.alerting.streaks import StreakKind, classify_drift
from driftwatch.config.loader import load_profile
from driftwatch.config.schema import Profile
from driftwatch.db.models import (
    Alert,
    AlertKind,
    AlertStatus,
    DriftResult,
    EvaluationWindow,
    PerformanceResult,
    TestMethod,
)
from driftwatch.db.session import SessionLocal
from driftwatch.demo.loader import load_scenario
from driftwatch.demo.schema import ScenarioConfig


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


AlertIdentity = tuple[str, str, str | None, str | None, str | None]
"""(kind, signal_name, feature_name, segment_dimension, segment_value) --
the same identity tuple that makes two Alert rows "the same recurring
signal" per Alert's own docstring in driftwatch.db.models."""


def _identity(alert: Alert) -> AlertIdentity:
    return (
        str(alert.kind),
        alert.signal_name,
        alert.feature_name,
        alert.segment_dimension,
        alert.segment_value,
    )


def _unexpected_alerts(alerts: list[Alert], expected: set[AlertIdentity]) -> list[Alert]:
    """Every alert row the run wrote that the scenario did not claim. Nothing
    is excluded: an earlier version filtered out the permanently-open
    not_computable alerts the engine raised on a categorical feature
    evaluated inside a segment defined by that same feature, which meant
    "no unexpected alerts" was true only after ignoring three alerts a real
    deployment would have seen. That structural signal is no longer
    evaluated (see driftwatch.evaluation.drift._evaluate_all_features), so
    the check is now exactly what its name says."""
    return [alert for alert in alerts if _identity(alert) not in expected]


def _describe(alerts: list[Alert]) -> str:
    return ", ".join(
        f"{a.kind}/{a.signal_name}/{a.feature_name}/{a.segment_value}:{a.status}" for a in alerts
    )


def _drift_history(
    session: Session,
    model_id: str,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str | None = None,
    segment_value: str | None = None,
) -> list[DriftResult]:
    """Every DriftResult ever written for this exact signal, oldest window
    first -- the full scenario range, not the bounded rescan window the real
    alerting engine uses (verification needs to see the whole story, not
    just what a live evaluation run would have rescanned)."""
    query = (
        session.query(DriftResult)
        .join(EvaluationWindow, DriftResult.evaluation_window_id == EvaluationWindow.id)
        .filter(
            EvaluationWindow.model_id == model_id,
            DriftResult.feature_name == feature_name,
            DriftResult.test_method == test_method,
        )
    )
    if segment_dimension is None:
        query = query.filter(DriftResult.segment_dimension.is_(None))
    else:
        query = query.filter(
            DriftResult.segment_dimension == segment_dimension,
            DriftResult.segment_value == segment_value,
        )
    return query.order_by(EvaluationWindow.window_start.asc()).all()


def _suppressed_breach_episodes(
    session: Session,
    profile: Profile,
    model_id: str,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str | None = None,
    segment_value: str | None = None,
) -> list[SuppressedEpisode]:
    history = _drift_history(
        session, model_id, feature_name, test_method, segment_dimension, segment_value
    )
    fire_threshold, clear_threshold = _drift_thresholds(profile, test_method)
    classifications: list[tuple[int, StreakKind, float | None]] = [
        (r.evaluation_window_id, classify_drift(r, fire_threshold, clear_threshold), r.statistic)
        for r in history
    ]
    persistence = profile.alerting.fire_persistence_windows
    return find_suppressed_episodes(classifications, persistence, "breach")


def _alerts_for(session: Session, model_id: str) -> list[Alert]:
    return session.query(Alert).filter(Alert.model_id == model_id).all()


def _verify_clean(session: Session, scenario: ScenarioConfig, profile: Profile) -> list[Check]:
    alerts = _alerts_for(session, scenario.model_id)
    unexpected = _unexpected_alerts(alerts, expected=set())
    checks = [
        Check(
            "no unexpected alerts",
            not unexpected,
            "none" if not unexpected else f"{len(unexpected)} unexpected: {_describe(unexpected)}",
        )
    ]
    significant = (
        session.query(DriftResult)
        .join(EvaluationWindow, DriftResult.evaluation_window_id == EvaluationWindow.id)
        .filter(
            EvaluationWindow.model_id == scenario.model_id, DriftResult.is_significant.is_(True)
        )
        .count()
    )
    checks.append(
        Check(
            "zero significant drift readings anywhere",
            significant == 0,
            f"{significant} significant",
        )
    )
    return checks


_REGIONS: list[tuple[str | None, str | None]] = [
    (None, None),
    ("region", "EU"),
    ("region", "US"),
    ("region", "APAC"),
]


def _verify_covariate_shift(
    session: Session, scenario: ScenarioConfig, profile: Profile
) -> list[Check]:
    expected: set[AlertIdentity] = {
        ("drift", method, "age", dimension, value)
        for method in ("psi", "ks")
        for dimension, value in _REGIONS
    }
    alerts = _alerts_for(session, scenario.model_id)
    unexpected = _unexpected_alerts(alerts, expected)
    checks = [
        Check(
            "no unexpected alerts",
            not unexpected,
            "none" if not unexpected else f"{len(unexpected)} unexpected: {_describe(unexpected)}",
        )
    ]

    sustained = next(
        (
            a
            for a in alerts
            if a.kind == AlertKind.DRIFT and a.feature_name == "age" and a.segment_dimension is None
        ),
        None,
    )
    checks.append(
        Check(
            "sustained global age shift fires and escalates",
            sustained is not None and sustained.status == AlertStatus.ESCALATED,
            f"status={sustained.status if sustained else 'missing'}",
        )
    )

    suppressed = _suppressed_breach_episodes(
        session, profile, scenario.model_id, "age", TestMethod.PSI
    )
    checks.append(
        Check(
            "single-window blip is suppressed, never opens an alert",
            len(suppressed) == 1,
            f"found {len(suppressed)} suppressed breach episode(s) on global age/psi: {suppressed}",
        )
    )
    return checks


def _describe_journey(matching: list[Alert]) -> str:
    if len(matching) != 1:
        return f"{len(matching)} alert rows"
    alert = matching[0]
    if alert.escalation_evidence_statistic is None or alert.evidence_statistic is None:
        return f"status={alert.status.value}, never escalated"
    return (
        f"status={alert.status.value}, opened at {alert.evidence_statistic:.3f}, "
        f"escalated at {alert.escalation_evidence_statistic:.3f}"
    )


def _verify_segment_isolated(
    session: Session, scenario: ScenarioConfig, profile: Profile
) -> list[Check]:
    expected: set[AlertIdentity] = {
        ("drift", "psi", "age", "region", "APAC"),
        ("drift", "ks", "age", "region", "APAC"),
    }
    alerts = _alerts_for(session, scenario.model_id)
    unexpected = _unexpected_alerts(alerts, expected)
    checks = [
        Check(
            "no unexpected alerts",
            not unexpected,
            "none" if not unexpected else f"{len(unexpected)} unexpected: {_describe(unexpected)}",
        )
    ]

    global_history = _drift_history(session, scenario.model_id, "age", TestMethod.PSI)
    fire_threshold, _clear = _drift_thresholds(profile, TestMethod.PSI)
    stats = [r.statistic for r in global_history if r.statistic is not None]
    max_stat = max(stats) if stats else None
    checks.append(
        Check(
            "global age PSI stays under the fire threshold throughout",
            max_stat is not None and max_stat < fire_threshold,
            f"max={max_stat}, fire_threshold={fire_threshold}",
        )
    )

    # One check per expected alert, and each must have done the whole
    # journey on its own: opened, escalated (the shift lasts longer than the
    # escalate persistence count), and resolved once the shift ended. An
    # earlier version passed if the SET of statuses across the PSI and KS
    # alerts contained "resolved" and one of open/escalated -- which a run
    # where the PSI alert never resolved (its sampling-noise floor at 75
    # APAC rows per window sat in the dead zone) satisfied via the KS alert.
    for test_method in (TestMethod.PSI, TestMethod.KS):
        matching = [
            a
            for a in alerts
            if a.signal_name == test_method.value
            and a.feature_name == "age"
            and a.segment_dimension == "region"
            and a.segment_value == "APAC"
        ]
        alert = matching[0] if len(matching) == 1 else None
        checks.append(
            Check(
                f"APAC age {test_method.value} alert opens, escalates, and (once the shift "
                f"ends) resolves",
                alert is not None
                and alert.escalated_at is not None
                and alert.status == AlertStatus.RESOLVED,
                _describe_journey(matching),
            )
        )
    return checks


def _verify_concept_drift(
    session: Session, scenario: ScenarioConfig, profile: Profile
) -> list[Check]:
    expected: set[AlertIdentity] = {
        (kind, "pr_auc", None, dimension, value)
        for kind in ("performance", "not_computable")
        for dimension, value in _REGIONS
    }
    alerts = _alerts_for(session, scenario.model_id)
    unexpected = _unexpected_alerts(alerts, expected)
    checks = [
        Check(
            "no unexpected alerts",
            not unexpected,
            "none" if not unexpected else f"{len(unexpected)} unexpected: {_describe(unexpected)}",
        )
    ]

    drifted_features = (
        session.query(DriftResult.feature_name)
        .join(EvaluationWindow, DriftResult.evaluation_window_id == EvaluationWindow.id)
        .filter(
            EvaluationWindow.model_id == scenario.model_id, DriftResult.is_significant.is_(True)
        )
        .distinct()
        .all()
    )
    checks.append(
        Check(
            "no feature or prediction-score drift (inputs stay stable)",
            not drifted_features,
            "none" if not drifted_features else f"drifted: {[f for f, in drifted_features]}",
        )
    )

    global_perf_alert = next(
        (
            a
            for a in alerts
            if a.kind == AlertKind.PERFORMANCE
            and a.signal_name == "pr_auc"
            and a.segment_dimension is None
        ),
        None,
    )
    checks.append(
        Check(
            "global PR-AUC degrades and escalates after label_noise",
            global_perf_alert is not None and global_perf_alert.status == AlertStatus.ESCALATED,
            f"status={global_perf_alert.status if global_perf_alert else 'missing'}",
        )
    )

    metric_spec = next(spec for spec in profile.performance_metrics if spec.name == "pr_auc")
    assert metric_spec.fire_threshold is not None
    assert metric_spec.clear_threshold is not None
    final_rows = (
        session.query(EvaluationWindow.window_start, PerformanceResult.metric_value)
        .join(PerformanceResult, PerformanceResult.evaluation_window_id == EvaluationWindow.id)
        .filter(
            EvaluationWindow.model_id == scenario.model_id,
            PerformanceResult.metric_name == "pr_auc",
            PerformanceResult.segment_dimension.is_(None),
            PerformanceResult.is_retroactive.is_(True),
        )
        .order_by(EvaluationWindow.window_start.asc())
        .all()
    )
    final_values = [v for _, v in final_rows]
    event_start = next(e.start_window for e in scenario.events if e.shift_type == "label_noise")
    early = final_values[:event_start]
    late = final_values[event_start:]
    early_mean = statistics.mean(early) if early else None
    late_mean = statistics.mean(late) if late else None
    checks.append(
        Check(
            "PR-AUC healthy before label_noise, degraded after (final backfilled values)",
            early_mean is not None
            and late_mean is not None
            and early_mean >= metric_spec.clear_threshold
            and late_mean <= metric_spec.fire_threshold,
            f"early_mean={early_mean}, late_mean={late_mean}, "
            f"clear_threshold={metric_spec.clear_threshold}, "
            f"fire_threshold={metric_spec.fire_threshold}",
        )
    )

    initial_not_computable = (
        session.query(PerformanceResult)
        .join(EvaluationWindow, PerformanceResult.evaluation_window_id == EvaluationWindow.id)
        .filter(
            EvaluationWindow.model_id == scenario.model_id,
            PerformanceResult.metric_name == "pr_auc",
            PerformanceResult.segment_dimension.is_(None),
            PerformanceResult.is_retroactive.is_(False),
            PerformanceResult.status == "not_computable",
        )
        .count()
    )
    checks.append(
        Check(
            "initial evaluation frequently lacks enough labels "
            "(degradation only visible after backfill)",
            initial_not_computable > 0,
            f"{initial_not_computable} windows initially not_computable for lack of labels",
        )
    )
    return checks


def _verify_burst_shift(
    session: Session, scenario: ScenarioConfig, profile: Profile
) -> list[Check]:
    """The aggressive profile's end-to-end check: the burst opens on its first
    breaching window (fire persistence 1), escalates on its third, and
    resolves; the single-window blip, which patient suppresses, opens and
    resolves too. Both events are global shifts on one feature, so every
    region's segment sees them as well."""
    expected: set[AlertIdentity] = {
        ("drift", method, "age", dimension, value)
        for method in ("psi", "ks")
        for dimension, value in _REGIONS
    }
    alerts = _alerts_for(session, scenario.model_id)
    unexpected = _unexpected_alerts(alerts, expected)
    checks = [
        Check(
            "no unexpected alerts",
            not unexpected,
            "none" if not unexpected else f"{len(unexpected)} unexpected: {_describe(unexpected)}",
        )
    ]

    burst, blip = scenario.events
    window_ids_by_index = {
        int((w.window_start - scenario.start_date).total_seconds() // 3600): w.id
        for w in session.query(EvaluationWindow).filter(
            EvaluationWindow.model_id == scenario.model_id
        )
    }
    for test_method in (TestMethod.PSI, TestMethod.KS):
        rows = sorted(
            (
                a
                for a in alerts
                if a.signal_name == test_method.value
                and a.feature_name == "age"
                and a.segment_dimension is None
            ),
            key=lambda a: a.id,
        )
        first = rows[0] if len(rows) == 2 else None
        second = rows[1] if len(rows) == 2 else None
        checks.append(
            Check(
                f"global age {test_method.value}: the burst opens on its FIRST breaching window, "
                "escalates, and resolves",
                first is not None
                and first.evidence_window_id == window_ids_by_index.get(burst.start_window)
                and first.escalated_at is not None
                and first.status == AlertStatus.RESOLVED,
                f"{len(rows)} alert rows"
                if first is None
                else f"opened at window {burst.start_window}: "
                f"{first.evidence_window_id == window_ids_by_index.get(burst.start_window)}, "
                f"{_describe_journey([first])}",
            )
        )
        checks.append(
            Check(
                f"global age {test_method.value}: the one-window blip opens (aggressive does not "
                "suppress it) and resolves without escalating",
                second is not None
                and second.evidence_window_id == window_ids_by_index.get(blip.start_window)
                and second.escalated_at is None
                and second.status == AlertStatus.RESOLVED,
                f"{len(rows)} alert rows" if second is None else _describe_journey([second]),
            )
        )
    return checks


_VERIFIERS: dict[str, Callable[[Session, ScenarioConfig, Profile], list[Check]]] = {
    "clean": _verify_clean,
    "covariate_shift": _verify_covariate_shift,
    "segment_isolated": _verify_segment_isolated,
    "concept_drift": _verify_concept_drift,
    "burst_shift": _verify_burst_shift,
}


def _verify_scenario_was_loaded(session: Session, scenario: ScenarioConfig) -> Check:
    """`driftwatch demo <scenario>` truncates every app table, not just rows
    for its own model_id -- loading a different scenario afterward silently
    wipes this one out. Without this check, an empty (or wrong-scenario)
    database would make every other check here pass vacuously: zero rows
    means zero unexpected alerts and zero significant drift readings alike.
    This must run first and must be checked, not assumed."""
    window_count = (
        session.query(EvaluationWindow)
        .filter(EvaluationWindow.model_id == scenario.model_id)
        .count()
    )
    return Check(
        "scenario is currently loaded in the database",
        window_count == scenario.total_windows,
        f"found {window_count} evaluated windows for {scenario.model_id!r}, "
        f"expected {scenario.total_windows} -- run `driftwatch demo {scenario.name}` first "
        "if this doesn't match (loading a different scenario truncates every table)",
    )


def verify_scenario(scenario_name: str, *, verbose: bool = False) -> bool:
    if scenario_name not in _VERIFIERS:
        raise ValueError(
            f"no verifier registered for scenario {scenario_name!r} -- known: {sorted(_VERIFIERS)}"
        )
    scenario = load_scenario(scenario_name)
    profile = load_profile(scenario.profile)

    session = SessionLocal()
    try:
        loaded_check = _verify_scenario_was_loaded(session, scenario)
        checks = [loaded_check, *_VERIFIERS[scenario_name](session, scenario, profile)]
    finally:
        session.close()

    if verbose:
        for check in checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"[{status}] {check.name}: {check.detail}")

    return all(check.passed for check in checks)
