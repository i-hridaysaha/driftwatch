import math

import numpy as np
import pytest

from driftwatch.stats.binning import compute_continuous_edges
from driftwatch.stats.psi import (
    EPSILON,
    _bucket_proportions_array,
    _psi_from_proportions,
    psi,
    psi_null_ceiling_from_table,
    psi_null_floor,
    simulate_psi_null,
)
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


def test_null_floor_matches_the_chi_square_approximation() -> None:
    """(B - 1) k for the mean and sqrt(2 (B - 1)) k for the sd, with
    k = 1/n_baseline + 1/n_live -- Yurdakul (2018)."""
    floor = psi_null_floor(n_baseline=225, n_live=75, n_buckets=12)
    k = 1 / 225 + 1 / 75
    assert floor.mean == pytest.approx(11 * k)
    assert floor.sd == pytest.approx(math.sqrt(22) * k)
    assert floor.ceiling == pytest.approx(floor.mean + 2 * floor.sd)
    # the segment geometry the case study was first published on: the
    # ceiling sits above the patient profile's 0.25 fire threshold
    assert floor.ceiling > 0.25


def test_null_floor_matches_simulation_when_bins_are_populated() -> None:
    """Empirical check with the real psi() on same-distribution samples at
    a volume where every interior bin is populated, so the asymptotic
    approximation should be close: mean within 25 percent."""
    rng = np.random.default_rng(0)
    values = []
    for _ in range(400):
        base = rng.normal(40, 12, 600).tolist()
        live = rng.normal(40, 12, 210).tolist()
        edges = compute_continuous_edges(base)
        result = psi(base, live, edges)
        assert result.value is not None
        values.append(result.value)
    empirical_mean = float(np.mean(values))
    predicted = psi_null_floor(600, 210, 12).mean
    assert abs(empirical_mean - predicted) / predicted < 0.25


def test_null_floor_shrinks_with_volume_and_is_infinite_for_empty_samples() -> None:
    small = psi_null_floor(225, 75, 12)
    large = psi_null_floor(900, 300, 12)
    assert large.ceiling < small.ceiling
    assert large.ceiling < 0.15  # the volume the segment scenario ships at
    assert math.isinf(psi_null_floor(0, 75, 12).ceiling)
    assert math.isinf(psi_null_floor(225, 75, 1).ceiling)


def test_vectorised_null_simulation_uses_the_same_psi_as_psi() -> None:
    """The simulation's bucketing and formula must be bit-for-bit psi()'s,
    or the table would guard a different statistic than the one gated on."""
    rng = np.random.default_rng(3)
    base = rng.normal(40, 12, 225)
    live = rng.normal(40, 12, 75)
    edges = compute_continuous_edges(base.tolist())
    slow = psi(base.tolist(), live.tolist(), edges).value
    fast = _psi_from_proportions(
        _bucket_proportions_array(base, edges), _bucket_proportions_array(live, edges)
    )
    assert slow == pytest.approx(fast, abs=1e-12)


def test_simulated_null_is_heavier_than_the_approximation_at_small_sizes() -> None:
    """The chi-square approximation is optimistic below about a hundred rows,
    where empty bins hit the epsilon floor; the simulation sees that. At
    the case study's original segment geometry the two are close; at thirty
    rows the simulation is far above."""
    rng = np.random.default_rng(0)
    base = rng.normal(40, 12, 225).tolist()
    edges = compute_continuous_edges(base)
    table = simulate_psi_null(base, edges, live_sizes=(30, 75, 300), draws=150)
    at_30 = psi_null_ceiling_from_table(table, 30)
    at_75 = psi_null_ceiling_from_table(table, 75)
    at_300 = psi_null_ceiling_from_table(table, 300)
    assert at_30 is not None and at_75 is not None and at_300 is not None
    assert at_30 > at_75 > at_300
    assert at_30 > 2 * psi_null_floor(225, 30, 12).ceiling * 0.8
    assert at_75 == pytest.approx(psi_null_floor(225, 75, 12).ceiling, rel=0.35)


def test_ceiling_lookup_interpolates_in_log_size_and_clamps_at_the_ends() -> None:
    table = {"sizes": [10.0, 100.0, 1000.0], "mean": [0.0, 0.0, 0.0], "ceiling": [1.0, 0.5, 0.1]}
    assert psi_null_ceiling_from_table(table, 5) == 1.0
    assert psi_null_ceiling_from_table(table, 10) == 1.0
    assert psi_null_ceiling_from_table(table, 1000) == 0.1
    assert psi_null_ceiling_from_table(table, 5000) == 0.1
    mid = psi_null_ceiling_from_table(table, 316)  # halfway between 100 and 1000 in log space
    assert mid == pytest.approx(0.3, abs=0.01)
    assert psi_null_ceiling_from_table({"sizes": [], "mean": [], "ceiling": []}, 50) is None
