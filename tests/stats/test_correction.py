import math

from driftwatch.stats.correction import benjamini_hochberg


def test_empty_input_is_defined_not_raising() -> None:
    result = benjamini_hochberg([], alpha=0.05)

    assert result.adjusted_p_values == []
    assert result.is_significant == []


def test_all_p_values_at_their_threshold_are_all_significant() -> None:
    # p_(i) == i/m*alpha exactly at every rank -> every rank passes its own
    # threshold, so everything is significant with q-value == alpha throughout.
    p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

    result = benjamini_hochberg(p_values, alpha=0.05)

    assert result.is_significant == [True, True, True, True, True]
    for q in result.adjusted_p_values:
        assert math.isclose(q, 0.05, rel_tol=1e-9)


def test_only_the_smallest_p_value_survives() -> None:
    # only rank 1 (0.001 <= 0.01) satisfies its threshold; every later rank's
    # p-value is far above its own (much looser) threshold.
    p_values = [0.001, 0.2, 0.3, 0.4, 0.5]

    result = benjamini_hochberg(p_values, alpha=0.05)

    assert result.is_significant == [True, False, False, False, False]
    assert math.isclose(result.adjusted_p_values[0], 0.005, rel_tol=1e-9)


def test_step_up_procedure_rescues_an_earlier_failing_rank() -> None:
    # sorted p-values [0.001, 0.03, 0.035, 0.045] against thresholds
    # [0.0125, 0.025, 0.0375, 0.05] (m=4, alpha=0.05): rank 2 (0.03) individually
    # FAILS its own threshold (0.025), but rank 4 (0.045) passes -- BH's step-up
    # rule means every rank at or below the LARGEST passing rank is significant,
    # including the one that failed on its own. This is the property a naive
    # per-rank implementation gets wrong.
    p_values = [0.001, 0.03, 0.035, 0.045]

    result = benjamini_hochberg(p_values, alpha=0.05)

    assert result.is_significant == [True, True, True, True]
    assert math.isclose(result.adjusted_p_values[0], 0.004, rel_tol=1e-9)
    assert math.isclose(result.adjusted_p_values[1], 0.045, rel_tol=1e-9)
    assert math.isclose(result.adjusted_p_values[2], 0.045, rel_tol=1e-9)
    assert math.isclose(result.adjusted_p_values[3], 0.045, rel_tol=1e-9)


def test_output_order_matches_input_order_not_sorted_order() -> None:
    p_values = [0.045, 0.001, 0.035, 0.03]  # same values as above, shuffled

    result = benjamini_hochberg(p_values, alpha=0.05)

    assert result.is_significant == [True, True, True, True]
    assert math.isclose(result.adjusted_p_values[1], 0.004, rel_tol=1e-9)  # the 0.001 rank


def test_adjusted_p_values_never_exceed_one() -> None:
    p_values = [0.9, 0.95, 0.99]

    result = benjamini_hochberg(p_values, alpha=0.05)

    assert all(q <= 1.0 for q in result.adjusted_p_values)


def test_is_pure_and_deterministic() -> None:
    p_values = [0.01, 0.2, 0.03, 0.4, 0.005]

    first = benjamini_hochberg(p_values, alpha=0.05)
    second = benjamini_hochberg(list(p_values), alpha=0.05)

    assert first == second
    assert p_values == [0.01, 0.2, 0.03, 0.4, 0.005]
