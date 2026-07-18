import math

from driftwatch.stats.psi import EPSILON, psi
from driftwatch.stats.result import StatStatus

EDGES = [1.0, 2.0, 3.0]  # 2 interior bins -> 4 buckets: under, (1,2], (2,3], over


def test_identical_distributions_give_zero() -> None:
    values = [0.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5, 2.5, 100.0]

    result = psi(values, values, EDGES)

    assert result.status == StatStatus.COMPUTED
    assert result.value == 0.0


def test_known_answer_case() -> None:
    # baseline: 1 in bucket0, 4 in bucket1, 4 in bucket2, 1 in bucket3 -> [0.1, 0.4, 0.4, 0.1]
    baseline = [0.0] * 1 + [1.5] * 4 + [2.5] * 4 + [100.0] * 1
    # live: 2 in bucket0, 3 in bucket1, 3 in bucket2, 2 in bucket3 -> [0.2, 0.3, 0.3, 0.2]
    live = [0.0] * 2 + [1.5] * 3 + [2.5] * 3 + [100.0] * 2

    result = psi(baseline, live, EDGES)

    # PSI = sum((live - baseline) * ln(live / baseline)) per bucket, computed directly
    # from the formula against the same proportions constructed above.
    expected = (
        (0.2 - 0.1) * math.log(0.2 / 0.1)
        + (0.3 - 0.4) * math.log(0.3 / 0.4)
        + (0.3 - 0.4) * math.log(0.3 / 0.4)
        + (0.2 - 0.1) * math.log(0.2 / 0.1)
    )
    assert result.status == StatStatus.COMPUTED
    assert result.value is not None
    assert math.isclose(result.value, expected, rel_tol=1e-9)


def test_values_outside_baseline_range_are_drift_not_errors() -> None:
    baseline = [1.5] * 10  # entirely within the interior range, no overflow at all
    live = [1000.0] * 10  # entirely in the overflow bucket -- unseen by the baseline

    result = psi(baseline, live, EDGES)

    # baseline proportion for the overflow bucket is exactly 0 (floored to EPSILON);
    # live proportion there is 1.0 -- this must produce a large, finite PSI, not a
    # crash and not a silently-ignored value. Two buckets contribute: the interior
    # bin the baseline was entirely concentrated in (baseline=1.0, live=0->EPSILON)
    # and the overflow bucket the live sample moved entirely into (baseline=0-
    # >EPSILON, live=1.0) -- both terms happen to be algebraically identical here.
    assert result.status == StatStatus.COMPUTED
    assert result.value is not None
    assert result.value > 0
    assert math.isfinite(result.value)
    interior_bin_term = (EPSILON - 1.0) * math.log(EPSILON / 1.0)
    overflow_bucket_term = (1.0 - EPSILON) * math.log(1.0 / EPSILON)
    expected = interior_bin_term + overflow_bucket_term
    assert math.isclose(result.value, expected, rel_tol=1e-9)


def test_empty_baseline_is_not_computable_not_raising() -> None:
    result = psi([], [1.5], EDGES)

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.value is None
    assert result.not_computable_reason


def test_empty_live_is_not_computable_not_raising() -> None:
    result = psi([1.5], [], EDGES)

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.value is None
    assert result.not_computable_reason


def test_empty_edges_is_not_computable_not_raising() -> None:
    result = psi([1.5], [1.5], [])

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.value is None
    assert result.not_computable_reason


def test_single_value_baseline_is_defined_not_raising() -> None:
    # a feature that was constant in the baseline (e.g. edges collapsed to [5.0, 5.0])
    degenerate_edges = [5.0, 5.0]

    same_value = psi([5.0], [5.0], degenerate_edges)
    different_value = psi([5.0], [6.0], degenerate_edges)

    assert same_value.status == StatStatus.COMPUTED
    assert same_value.value == 0.0
    assert different_value.status == StatStatus.COMPUTED
    assert different_value.value is not None
    assert different_value.value > 0  # a live value the baseline never saw is drift


def test_is_pure_and_deterministic() -> None:
    baseline = [0.5, 1.5, 2.5, 2.9]
    live = [1.1, 1.9, 2.1, 2.8]

    first = psi(baseline, live, EDGES)
    second = psi(list(reversed(baseline)), list(reversed(live)), EDGES)

    assert first == second
    assert baseline == [0.5, 1.5, 2.5, 2.9]  # inputs untouched
