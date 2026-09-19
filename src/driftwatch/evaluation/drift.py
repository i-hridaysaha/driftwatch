from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from driftwatch.alerting.engine import evaluate_drift_alerts_for_window
from driftwatch.config.loader import compute_config_hash, load_model_config, load_profile
from driftwatch.config.schema import FeatureSpec, ModelConfig, Profile
from driftwatch.db.models import (
    Baseline,
    BaselineRecord,
    DriftResult,
    EvaluationWindow,
    MetricStatus,
    PerformanceResult,
    Prediction,
    TestMethod,
)
from driftwatch.evaluation.performance import recompute_performance_for_window
from driftwatch.stats.binning import prediction_score_null_floor_table, psi_null_floor_table
from driftwatch.stats.chi_square import ChiSquareResult, chi_square
from driftwatch.stats.correction import benjamini_hochberg
from driftwatch.stats.floors import cramers_v_null_ceiling, jsd_null_ceiling, ks_null_ceiling
from driftwatch.stats.jsd import JSDResult, jsd
from driftwatch.stats.ks import KSResult, ks
from driftwatch.stats.psi import PSIResult, psi, psi_null_ceiling_from_table, psi_null_floor
from driftwatch.stats.result import StatStatus

PREDICTION_SCORE_FEATURE_NAME = "__prediction_score__"
"""Sentinel feature_name for the DriftResult row tracking the prediction-score
distribution (JSD), which isn't one of the model's declared input features."""


def evaluate_window(
    session: Session,
    model_id: str,
    window_start: datetime,
    window_end: datetime,
    *,
    force: bool = False,
) -> EvaluationWindow | None:
    """Evaluate one [window_start, window_end) window for `model_id`. This is
    the ONE code path both the scheduler and the CLI backfill command call --
    the only difference between them is how they decide which windows to pass
    in here (drift-watermark-gated discovery vs an explicit historical
    range).

    Two separate lifecycles happen here, and only one of them is final:

    - DRIFT is computed once, from the predictions that exist in the window
      right now, and is never recomputed for this window again -- see
      driftwatch.scheduler.windowing.is_drift_watermark_elapsed for why.
    - PERFORMANCE is computed for whatever labels exist at this moment, but
      that is only the FIRST of potentially many computations for this
      window: driftwatch.api.routes.labels flags the window as
      performance_stale whenever a label arrives for a prediction in it,
      and the scheduler's next tick recomputes it
      (driftwatch.scheduler.jobs.recompute_stale_windows), independent of
      this function -- indefinitely, with no watermark of its own. Calling
      this function again for an already-evaluated window does NOT pick up
      new labels; that happens through the flag, not by re-calling
      evaluate_window.

    Idempotent for DRIFT, enforced by a database constraint rather than an
    application-level check-then-act: the window is "claimed" by inserting
    its EvaluationWindow row inside a SAVEPOINT, relying on
    uq_evaluation_windows_model_range to raise IntegrityError if another
    process already claimed the same window concurrently. If the window
    already exists, this returns None (a no-op) unless `force=True`, which
    deletes the prior evaluation and its dependent rows first -- used by
    historical backfills/demo regeneration that need byte-identical re-runs.

    Raises ValueError if the model has no active baseline; that's a genuine
    precondition failure for the caller to handle, not a per-window
    not-computable result.
    """
    model_config = load_model_config(model_id)
    profile = load_profile(model_config.profile)

    baseline = session.scalars(
        select(Baseline).where(Baseline.model_id == model_id, Baseline.is_active.is_(True))
    ).first()
    if baseline is None:
        raise ValueError(f"model {model_id!r} has no active baseline")

    config_hash = compute_config_hash(model_config, profile)

    existing = session.scalars(
        select(EvaluationWindow).where(
            EvaluationWindow.model_id == model_id,
            EvaluationWindow.window_start == window_start,
            EvaluationWindow.window_end == window_end,
        )
    ).first()
    if existing is not None:
        if not force:
            return None
        _delete_window_dependents(session, existing.id)
        session.delete(existing)
        session.flush()

    window = EvaluationWindow(
        model_id=model_id,
        baseline_id=baseline.id,
        window_start=window_start,
        window_end=window_end,
        config_hash=config_hash,
        evaluated_at=datetime.now(UTC),
    )
    try:
        with session.begin_nested():
            session.add(window)
            session.flush()
    except IntegrityError:
        return None  # concurrently claimed by another evaluation run

    predictions = list(
        session.scalars(
            select(Prediction).where(
                Prediction.model_id == model_id,
                Prediction.predicted_at >= window_start,
                Prediction.predicted_at < window_end,
            )
        )
    )
    window.n_predictions = len(predictions)

    if len(predictions) < profile.evaluation.min_window_size:
        reason = (
            f"window has {len(predictions)} predictions, below configured "
            f"minimum of {profile.evaluation.min_window_size}"
        )
        drift_results = _not_computable_for_all_tests(window, model_config, profile, reason)
    else:
        baseline_records = list(
            session.scalars(
                select(BaselineRecord).where(BaselineRecord.baseline_id == baseline.id)
            )
        )
        drift_results = _evaluate_all_features(
            window, model_config, profile, baseline.binning_config, baseline_records, predictions
        )

    _apply_benjamini_hochberg(drift_results, profile.multiple_comparison_correction.fdr_alpha)
    for result in drift_results:
        session.add(result)

    session.flush()
    evaluate_drift_alerts_for_window(session, window, profile)
    recompute_performance_for_window(session, window.id)

    return window


def _delete_window_dependents(session: Session, window_id: int) -> None:
    session.execute(delete(DriftResult).where(DriftResult.evaluation_window_id == window_id))
    session.execute(
        delete(PerformanceResult).where(PerformanceResult.evaluation_window_id == window_id)
    )


def _make_drift_result(
    window: EvaluationWindow,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str | None,
    segment_value: str | None,
    n_baseline: int,
    n_live: int,
    *,
    status: MetricStatus,
    statistic: float | None = None,
    p_value: float | None = None,
    is_significant: bool | None = None,
    not_computable_reason: str | None = None,
) -> DriftResult:
    return DriftResult(
        evaluation_window_id=window.id,
        feature_name=feature_name,
        segment_dimension=segment_dimension,
        segment_value=segment_value,
        test_method=test_method,
        status=status,
        statistic=statistic,
        p_value=p_value,
        is_significant=is_significant,
        not_computable_reason=not_computable_reason,
        n_baseline=n_baseline,
        n_live=n_live,
    )


def _configured_methods(feature: FeatureSpec, profile: Profile) -> list[str]:
    if feature.dtype == "continuous":
        return list(profile.drift_tests.continuous.methods)
    return list(profile.drift_tests.categorical.methods)


def _not_computable_for_all_tests(
    window: EvaluationWindow, model_config: ModelConfig, profile: Profile, reason: str
) -> list[DriftResult]:
    results: list[DriftResult] = []
    for feature in model_config.schema_.features:
        for method in _configured_methods(feature, profile):
            results.append(
                _make_drift_result(
                    window,
                    feature.name,
                    TestMethod(method),
                    None,
                    None,
                    0,
                    0,
                    status=MetricStatus.NOT_COMPUTABLE,
                    not_computable_reason=reason,
                )
            )
    results.append(
        _make_drift_result(
            window,
            PREDICTION_SCORE_FEATURE_NAME,
            TestMethod(profile.drift_tests.prediction_score.method),
            None,
            None,
            0,
            0,
            status=MetricStatus.NOT_COMPUTABLE,
            not_computable_reason=reason,
        )
    )
    return results


def _map_status(status: StatStatus) -> MetricStatus:
    return MetricStatus.COMPUTED if status == StatStatus.COMPUTED else MetricStatus.NOT_COMPUTABLE


def _is_significant(status: StatStatus, effect_size: float | None, threshold: float) -> bool | None:
    if status != StatStatus.COMPUTED or effect_size is None:
        return None
    return effect_size >= threshold


def _evaluate_feature(
    window: EvaluationWindow,
    feature: FeatureSpec,
    profile: Profile,
    binning: dict[str, Any],
    baseline_records: list[BaselineRecord],
    predictions: list[Prediction],
    *,
    segment_dimension: str | None,
    segment_value: str | None,
) -> list[DriftResult]:
    results: list[DriftResult] = []
    baseline_raw = [r.features.get(feature.name) for r in baseline_records]
    baseline_present = [v for v in baseline_raw if v is not None]
    live_raw = [p.features.get(feature.name) for p in predictions]
    live_present = [v for v in live_raw if v is not None]

    if feature.dtype == "continuous":
        edges = binning.get("features", {}).get(feature.name, {}).get("edges", [])
        baseline_floats = [float(v) for v in baseline_present]
        live_floats = [float(v) for v in live_present]
        for method in profile.drift_tests.continuous.methods:
            if method == "psi":
                psi_result = _psi_or_below_floor(
                    baseline_floats,
                    live_floats,
                    edges,
                    profile.drift_tests.continuous.psi_clear_threshold,
                    psi_null_floor_table(binning, feature.name, segment_dimension, segment_value),
                )
                results.append(
                    _make_drift_result(
                        window,
                        feature.name,
                        TestMethod.PSI,
                        segment_dimension,
                        segment_value,
                        len(baseline_floats),
                        len(live_floats),
                        status=_map_status(psi_result.status),
                        statistic=psi_result.value,
                        is_significant=_is_significant(
                            psi_result.status,
                            psi_result.value,
                            profile.drift_tests.continuous.psi_threshold,
                        ),
                        not_computable_reason=psi_result.not_computable_reason,
                    )
                )
            elif method == "ks":
                ks_result = _ks_or_below_floor(
                    baseline_floats,
                    live_floats,
                    profile.drift_tests.continuous.ks_statistic_clear_threshold,
                )
                results.append(
                    _make_drift_result(
                        window,
                        feature.name,
                        TestMethod.KS,
                        segment_dimension,
                        segment_value,
                        len(baseline_floats),
                        len(live_floats),
                        status=_map_status(ks_result.status),
                        statistic=ks_result.statistic,
                        p_value=ks_result.p_value,
                        is_significant=_is_significant(
                            ks_result.status,
                            ks_result.statistic,
                            profile.drift_tests.continuous.ks_statistic_threshold,
                        ),
                        not_computable_reason=ks_result.not_computable_reason,
                    )
                )
    else:
        categories = binning.get("features", {}).get(feature.name, {}).get("categories", [])
        baseline_strs = [str(v) for v in baseline_present]
        live_strs = [str(v) for v in live_present]
        for _method in profile.drift_tests.categorical.methods:
            chi_result = _chi_square_or_below_floor(
                baseline_strs,
                live_strs,
                categories,
                profile.drift_tests.categorical.cramers_v_clear_threshold,
            )
            results.append(
                _make_drift_result(
                    window,
                    feature.name,
                    TestMethod.CHI_SQUARE,
                    segment_dimension,
                    segment_value,
                    len(baseline_strs),
                    len(live_strs),
                    status=_map_status(chi_result.status),
                    statistic=chi_result.cramers_v,
                    p_value=chi_result.p_value,
                    is_significant=_is_significant(
                        chi_result.status,
                        chi_result.cramers_v,
                        profile.drift_tests.categorical.cramers_v_threshold,
                    ),
                    not_computable_reason=chi_result.not_computable_reason,
                )
            )
    return results


def _psi_or_below_floor(
    baseline_floats: list[float],
    live_floats: list[float],
    edges: list[float],
    clear_threshold: float,
    null_table: dict[str, list[float]] | None = None,
) -> PSIResult:
    """PSI, unless this pair of sample sizes is too small for the profile's
    clear threshold to mean anything -- in which case NOT_COMPUTABLE with
    the reason, never a number.

    Under no shift at all, PSI reads sampling noise that grows as the
    samples shrink. If that noise ceiling sits above the clear threshold, a
    perfectly quiet segment cannot reliably classify as "clear": it lives
    in the dead zone or breaches on noise, an alert on it can never
    assemble the run of clear windows resolution needs, and a fire
    threshold near the ceiling opens alerts on nothing. Gating on such a
    number would be exactly the flapping the hysteresis exists to prevent,
    so it is refused here, with a reason a reader can act on (more rows per
    window, wider windows, or a wider clear threshold). min_window_size /
    min_segment_size are floors on whether a statistic can be computed at
    all; this is the floor on whether PSI is worth gating on, and it
    depends on the threshold.

    The ceiling comes from the baseline's own simulated null when the
    baseline carries one (driftwatch.stats.psi.simulate_psi_null, stored at
    registration for this feature and, for a segment, this slice), which
    sees the real feature's shape, the epsilon floor on empty bins and the
    baseline's own sampling term. A baseline registered before the tables
    existed falls back to the chi-square approximation
    (driftwatch.stats.psi.psi_null_floor), which is optimistic below about
    a hundred rows."""
    if not baseline_floats or not live_floats or not edges:
        return psi(baseline_floats, live_floats, edges)
    n_buckets = len(edges) + 1  # interior bins plus the two overflow buckets
    ceiling = psi_null_ceiling_from_table(null_table, len(live_floats)) if null_table else None
    if ceiling is not None:
        source = f"simulated on the baseline at {len(live_floats)} live rows"
    else:
        floor = psi_null_floor(len(baseline_floats), len(live_floats), n_buckets)
        ceiling = floor.ceiling
        source = (
            f"chi-square approximation, null mean {floor.mean:.3f} at "
            f"{len(baseline_floats)} baseline vs {len(live_floats)} live rows, {n_buckets} buckets"
        )
    if ceiling > clear_threshold:
        return PSIResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason=(
                f"PSI sampling-noise ceiling {ceiling:.3f} ({source}) exceeds the clear "
                f"threshold {clear_threshold}; a quiet window could not reliably read as "
                f"clear, so PSI is not gated on at this volume"
            ),
        )
    return psi(baseline_floats, live_floats, edges)


def _refused(reason: str, statistic: str, ceiling: float, source: str, clear: float) -> str:
    return (
        f"{statistic} sampling-noise ceiling {ceiling:.3f} ({source}) exceeds the clear "
        f"threshold {clear}; a quiet window could not reliably read as clear, so {statistic} is "
        f"not gated on at this volume{reason}"
    )


def _ks_or_below_floor(
    baseline_floats: list[float], live_floats: list[float], clear_threshold: float
) -> KSResult:
    """KS D, unless Kolmogorov's null ceiling at these two sample sizes
    exceeds the clear threshold (driftwatch.stats.floors.ks_null_ceiling).
    Same reasoning as _psi_or_below_floor; D's null falls with the square
    root of the sample size, so this bites only at a few dozen rows."""
    if not baseline_floats or not live_floats:
        return ks(baseline_floats, live_floats)
    ceiling = ks_null_ceiling(len(baseline_floats), len(live_floats))
    if ceiling > clear_threshold:
        return KSResult(
            status=StatStatus.NOT_COMPUTABLE,
            statistic=None,
            p_value=None,
            not_computable_reason=_refused(
                "",
                "KS D",
                ceiling,
                f"Kolmogorov at {len(baseline_floats)} baseline vs {len(live_floats)} live rows",
                clear_threshold,
            ),
        )
    return ks(baseline_floats, live_floats)


def _chi_square_or_below_floor(
    baseline_strs: list[str], live_strs: list[str], categories: list[str], clear_threshold: float
) -> ChiSquareResult:
    """Cramer's V, unless its chi-square null ceiling for a 2 x k table at
    these sample sizes exceeds the clear threshold
    (driftwatch.stats.floors.cramers_v_null_ceiling). A fifteen-category
    feature in a 600-row segment under a 0.07 clear threshold is the case
    that motivated this: V read about 0.1 on pure noise there."""
    if not baseline_strs or not live_strs:
        return chi_square(baseline_strs, live_strs, categories)
    n_categories = len(categories) + 1  # plus the "unseen" bucket
    ceiling = cramers_v_null_ceiling(len(baseline_strs), len(live_strs), n_categories)
    if ceiling > clear_threshold:
        return ChiSquareResult(
            status=StatStatus.NOT_COMPUTABLE,
            statistic=None,
            p_value=None,
            cramers_v=None,
            not_computable_reason=_refused(
                "",
                "Cramer's V",
                ceiling,
                f"chi-square with {n_categories - 1} degrees of freedom at "
                f"{len(baseline_strs)} baseline vs {len(live_strs)} live rows",
                clear_threshold,
            ),
        )
    return chi_square(baseline_strs, live_strs, categories)


def _jsd_or_below_floor(
    baseline_scores: list[float],
    live_scores: list[float],
    edges: list[float],
    clear_threshold: float,
    null_table: dict[str, list[float]] | None = None,
) -> JSDResult:
    """Prediction-score JSD, unless its null ceiling at these sample sizes
    exceeds the clear threshold: from the baseline's simulated table when
    it carries one, else driftwatch.stats.floors.jsd_null_ceiling."""
    if not baseline_scores or not live_scores or not edges:
        return jsd(baseline_scores, live_scores, edges)
    ceiling = psi_null_ceiling_from_table(null_table, len(live_scores)) if null_table else None
    if ceiling is not None:
        source = f"simulated on the baseline at {len(live_scores)} live rows"
    else:
        ceiling = jsd_null_ceiling(len(baseline_scores), len(live_scores), len(edges) + 1)
        source = (
            f"chi-square approximation at {len(baseline_scores)} baseline vs "
            f"{len(live_scores)} live rows, {len(edges) + 1} buckets"
        )
    if ceiling > clear_threshold:
        return JSDResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason=_refused("", "JSD", ceiling, source, clear_threshold),
        )
    return jsd(baseline_scores, live_scores, edges)


def _evaluate_prediction_score(
    window: EvaluationWindow,
    profile: Profile,
    binning: dict[str, Any],
    baseline_records: list[BaselineRecord],
    predictions: list[Prediction],
    *,
    segment_dimension: str | None,
    segment_value: str | None,
) -> list[DriftResult]:
    edges = binning.get("prediction_score", {}).get("edges", [])
    baseline_scores = [
        r.prediction_score for r in baseline_records if r.prediction_score is not None
    ]
    live_scores = [p.prediction_score for p in predictions if p.prediction_score is not None]

    jsd_result = _jsd_or_below_floor(
        baseline_scores,
        live_scores,
        edges,
        profile.drift_tests.prediction_score.jsd_clear_threshold,
        prediction_score_null_floor_table(binning, segment_dimension, segment_value),
    )
    return [
        _make_drift_result(
            window,
            PREDICTION_SCORE_FEATURE_NAME,
            TestMethod.JSD,
            segment_dimension,
            segment_value,
            len(baseline_scores),
            len(live_scores),
            status=_map_status(jsd_result.status),
            statistic=jsd_result.value,
            is_significant=_is_significant(
                jsd_result.status,
                jsd_result.value,
                profile.drift_tests.prediction_score.jsd_threshold,
            ),
            not_computable_reason=jsd_result.not_computable_reason,
        )
    ]


def _evaluate_all_features(
    window: EvaluationWindow,
    model_config: ModelConfig,
    profile: Profile,
    binning: dict[str, Any],
    baseline_records: list[BaselineRecord],
    predictions: list[Prediction],
) -> list[DriftResult]:
    results: list[DriftResult] = []

    for feature in model_config.schema_.features:
        results.extend(
            _evaluate_feature(
                window,
                feature,
                profile,
                binning,
                baseline_records,
                predictions,
                segment_dimension=None,
                segment_value=None,
            )
        )
    results.extend(
        _evaluate_prediction_score(
            window,
            profile,
            binning,
            baseline_records,
            predictions,
            segment_dimension=None,
            segment_value=None,
        )
    )

    for dimension in model_config.segments.dimensions:
        segment_values: list[str] = sorted(
            {
                str(prediction.segment_values[dimension])
                for prediction in predictions
                if prediction.segment_values.get(dimension) is not None
            }
        )
        for segment_value in segment_values:
            live_subset = [
                p for p in predictions if p.segment_values.get(dimension) == segment_value
            ]
            baseline_subset = [
                r for r in baseline_records if r.segment_values.get(dimension) == segment_value
            ]
            if len(live_subset) < model_config.segments.min_segment_size:
                reason = (
                    f"segment {dimension}={segment_value!r} has {len(live_subset)} live "
                    f"predictions, below configured minimum of "
                    f"{model_config.segments.min_segment_size}"
                )
                for feature in model_config.schema_.features:
                    if feature.name == dimension:
                        continue
                    for method in _configured_methods(feature, profile):
                        results.append(
                            _make_drift_result(
                                window,
                                feature.name,
                                TestMethod(method),
                                dimension,
                                segment_value,
                                len(baseline_subset),
                                len(live_subset),
                                status=MetricStatus.NOT_COMPUTABLE,
                                not_computable_reason=reason,
                            )
                        )
                results.append(
                    _make_drift_result(
                        window,
                        PREDICTION_SCORE_FEATURE_NAME,
                        TestMethod(profile.drift_tests.prediction_score.method),
                        dimension,
                        segment_value,
                        len(baseline_subset),
                        len(live_subset),
                        status=MetricStatus.NOT_COMPUTABLE,
                        not_computable_reason=reason,
                    )
                )
                continue

            for feature in model_config.schema_.features:
                if feature.name == dimension:
                    # Every row in the region='EU' segment has region == 'EU'
                    # by construction, so drift on the segment-defining
                    # feature inside its own segment can only ever be "one
                    # category": not a signal, not even a not_computable
                    # one. No row is written, and the dashboard renders the
                    # slot as not_configured -- which it is.
                    continue
                results.extend(
                    _evaluate_feature(
                        window,
                        feature,
                        profile,
                        binning,
                        baseline_subset,
                        live_subset,
                        segment_dimension=dimension,
                        segment_value=segment_value,
                    )
                )
            results.extend(
                _evaluate_prediction_score(
                    window,
                    profile,
                    binning,
                    baseline_subset,
                    live_subset,
                    segment_dimension=dimension,
                    segment_value=segment_value,
                )
            )

    return results


def _apply_benjamini_hochberg(drift_results: list[DriftResult], alpha: float) -> None:
    """Corrects p-values across every p-value-bearing test (KS, chi-square)
    produced by THIS window's evaluation -- global and all segments together,
    since they're all part of the same "single model, single window
    evaluation" per driftwatch.stats.correction.benjamini_hochberg's scope.
    Purely informational: is_significant was already decided by effect size
    before this runs, and is never revised here."""
    candidates = [
        result
        for result in drift_results
        if result.status == MetricStatus.COMPUTED and result.p_value is not None
    ]
    if not candidates:
        return
    p_values = [cast(float, result.p_value) for result in candidates]
    correction = benjamini_hochberg(p_values, alpha)
    for result, adjusted in zip(candidates, correction.adjusted_p_values, strict=True):
        result.corrected_p_value = adjusted
