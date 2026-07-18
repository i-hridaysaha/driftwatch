import math

from driftwatch.stats.jsd import jsd
from driftwatch.stats.result import StatStatus

EDGES = [1.0, 2.0, 3.0]  # 2 interior bins -> 4 buckets: under, (1,2], (2,3], over


def test_identical_distributions_give_zero() -> None:
    values = [0.5, 1.5, 2.5, 100.0]

    result = jsd(values, values, EDGES)

    assert result.status == StatStatus.COMPUTED
    assert result.value == 0.0


def test_known_answer_disjoint_distributions_give_one() -> None:
    # base-2 JS divergence for two distributions with entirely disjoint support is
    # exactly 1.0 -- the maximum possible value on this scale (see module docstring
    # for why base 2 + squaring scipy's distance, not scipy's raw output).
    baseline = [1.5] * 10  # entirely in interior bucket (1, 2]
    live = [2.5] * 10  # entirely in interior bucket (2, 3], zero overlap with baseline

    result = jsd(baseline, live, EDGES)

    assert result.status == StatStatus.COMPUTED
    assert result.value is not None
    assert math.isclose(result.value, 1.0, rel_tol=1e-9)


def test_empty_baseline_is_not_computable_not_raising() -> None:
    result = jsd([], [1.5], EDGES)

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.value is None
    assert result.not_computable_reason


def test_empty_live_is_not_computable_not_raising() -> None:
    result = jsd([1.5], [], EDGES)

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.value is None
    assert result.not_computable_reason


def test_empty_edges_is_not_computable_not_raising() -> None:
    result = jsd([1.5], [1.5], [])

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.value is None
    assert result.not_computable_reason


def test_no_epsilon_needed_for_zero_probability_buckets() -> None:
    # a bucket with zero mass in both distributions must not raise or produce nan --
    # unlike PSI, JSD's mixture-distribution formula is well-defined at zero.
    baseline = [1.5] * 5  # bucket (2,3] and both overflow buckets are empty in both
    live = [1.6] * 5

    result = jsd(baseline, live, EDGES)

    assert result.status == StatStatus.COMPUTED
    assert result.value == 0.0


def test_is_pure_and_deterministic() -> None:
    baseline = [0.5, 1.5, 2.5, 2.9]
    live = [1.1, 1.9, 2.1, 2.8]

    first = jsd(baseline, live, EDGES)
    second = jsd(list(reversed(baseline)), list(reversed(live)), EDGES)

    assert first == second
    assert baseline == [0.5, 1.5, 2.5, 2.9]
