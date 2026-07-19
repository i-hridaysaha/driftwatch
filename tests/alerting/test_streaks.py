from driftwatch.alerting.streaks import (
    Streak,
    classify_drift,
    classify_performance,
    current_streak,
)
from driftwatch.db.models import DriftResult, MetricStatus, PerformanceResult, TestMethod


def _drift_result(
    statistic: float | None, status: MetricStatus = MetricStatus.COMPUTED
) -> DriftResult:
    return DriftResult(
        evaluation_window_id=1,
        feature_name="age",
        test_method=TestMethod.PSI,
        status=status,
        statistic=statistic,
        n_baseline=0,
        n_live=0,
    )


def _performance_result(
    value: float | None, status: MetricStatus = MetricStatus.COMPUTED
) -> PerformanceResult:
    return PerformanceResult(
        evaluation_window_id=1,
        metric_name="pr_auc",
        status=status,
        metric_value=value,
        n_labeled=0,
    )


def test_classify_drift_breach_dead_zone_clear() -> None:
    fire, clear = 0.3, 0.2
    assert classify_drift(_drift_result(0.5), fire, clear) == "breach"
    assert classify_drift(_drift_result(0.3), fire, clear) == "breach"
    assert classify_drift(_drift_result(0.25), fire, clear) == "dead_zone"
    assert classify_drift(_drift_result(0.2), fire, clear) == "clear"
    assert classify_drift(_drift_result(0.05), fire, clear) == "clear"


def test_classify_drift_not_computable() -> None:
    result = _drift_result(None, status=MetricStatus.NOT_COMPUTABLE)

    assert classify_drift(result, fire_threshold=0.3, clear_threshold=0.2) == "not_computable"


def test_classify_performance_direction_below() -> None:
    # e.g. precision: low value is bad
    assert (
        classify_performance(
            _performance_result(0.2), "below", fire_threshold=0.3, clear_threshold=0.4
        )
        == "breach"
    )
    assert (
        classify_performance(
            _performance_result(0.35), "below", fire_threshold=0.3, clear_threshold=0.4
        )
        == "dead_zone"
    )
    assert (
        classify_performance(
            _performance_result(0.5), "below", fire_threshold=0.3, clear_threshold=0.4
        )
        == "clear"
    )


def test_classify_performance_direction_above() -> None:
    # e.g. RMSE: high value is bad
    assert (
        classify_performance(
            _performance_result(60.0), "above", fire_threshold=50.0, clear_threshold=40.0
        )
        == "breach"
    )
    assert (
        classify_performance(
            _performance_result(45.0), "above", fire_threshold=50.0, clear_threshold=40.0
        )
        == "dead_zone"
    )
    assert (
        classify_performance(
            _performance_result(30.0), "above", fire_threshold=50.0, clear_threshold=40.0
        )
        == "clear"
    )


def test_classify_performance_not_computable() -> None:
    result = _performance_result(None, status=MetricStatus.NOT_COMPUTABLE)

    assert classify_performance(result, "below", fire_threshold=0.3, clear_threshold=0.4) == (
        "not_computable"
    )


def test_current_streak_empty_input() -> None:
    assert current_streak([]) == Streak(kind=None, length=0)


def test_current_streak_counts_only_the_unbroken_run() -> None:
    # most-recent-first: breach, breach, clear, breach, breach, breach
    # -- the streak is 2 breaches, the clear two windows back interrupts it
    result = current_streak(["breach", "breach", "clear", "breach", "breach", "breach"])

    assert result == Streak(kind="breach", length=2)


def test_current_streak_dead_zone_breaks_a_breach_streak() -> None:
    # a boundary-hugging statistic drifting into the dead zone must not let
    # the breach streak keep accumulating -- this is what stops flapping
    result = current_streak(["dead_zone", "breach", "breach", "breach"])

    assert result == Streak(kind="dead_zone", length=1)


def test_current_streak_all_same_classification() -> None:
    result = current_streak(["clear", "clear", "clear"])

    assert result == Streak(kind="clear", length=3)


def test_current_streak_saturates_at_provided_history_length() -> None:
    # if the caller only fetched N rows, the streak can never be reported as
    # longer than N even if the true streak extends further back
    result = current_streak(["breach"] * 5)

    assert result.length == 5
