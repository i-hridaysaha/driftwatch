from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

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
from driftwatch.stats.chi_square import chi_square
from driftwatch.stats.correction import benjamini_hochberg
from driftwatch.stats.jsd import jsd
from driftwatch.stats.ks import ks
from driftwatch.stats.psi import psi
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
      window: driftwatch.api.routes.labels calls
      recompute_performance_for_window directly, independent of this
      function, every time a new label arrives for a prediction in this
      window -- indefinitely, with no watermark of its own. Calling this
      function again for an already-evaluated window does NOT pick up new
      labels; that happens automatically via label ingestion, not by
      re-calling evaluate_window.

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
                psi_result = psi(baseline_floats, live_floats, edges)
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
                ks_result = ks(baseline_floats, live_floats)
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
            chi_result = chi_square(baseline_strs, live_strs, categories)
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

    jsd_result = jsd(baseline_scores, live_scores, edges)
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
