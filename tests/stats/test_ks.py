from driftwatch.stats.ks import ks
from driftwatch.stats.result import StatStatus


def test_identical_distributions_give_zero_statistic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    result = ks(values, values)

    assert result.status == StatStatus.COMPUTED
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_known_answer_completely_disjoint_samples() -> None:
    # two equal-size samples with completely non-overlapping ranges: their empirical
    # CDFs never cross, so the KS D-statistic is exactly 1.0 -- the maximum possible.
    baseline = list(range(1, 11))
    live = list(range(11, 21))

    result = ks(baseline, live)

    assert result.status == StatStatus.COMPUTED
    assert result.statistic == 1.0


def test_empty_baseline_is_not_computable_not_raising() -> None:
    result = ks([], [1.0, 2.0])

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.statistic is None
    assert result.p_value is None
    assert result.not_computable_reason


def test_empty_live_is_not_computable_not_raising() -> None:
    result = ks([1.0, 2.0], [])

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.statistic is None
    assert result.p_value is None
    assert result.not_computable_reason


def test_single_value_samples_are_defined_not_raising() -> None:
    same = ks([5.0], [5.0])
    different = ks([5.0], [6.0])

    assert same.status == StatStatus.COMPUTED
    assert same.statistic == 0.0
    assert different.status == StatStatus.COMPUTED
    assert different.statistic == 1.0


def test_is_pure_and_deterministic() -> None:
    baseline = [1.0, 5.0, 2.0, 9.0, 3.0]
    live = [4.0, 8.0, 6.0, 2.0, 7.0]

    first = ks(baseline, live)
    second = ks(list(reversed(baseline)), list(reversed(live)))

    assert first == second
    assert baseline == [1.0, 5.0, 2.0, 9.0, 3.0]
