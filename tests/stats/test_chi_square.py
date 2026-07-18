import math

from driftwatch.stats.chi_square import chi_square
from driftwatch.stats.result import StatStatus

CATEGORIES = ["A", "B"]


def test_identical_distributions_give_zero_statistic() -> None:
    values = ["A"] * 50 + ["B"] * 50

    result = chi_square(values, values, CATEGORIES)

    assert result.status == StatStatus.COMPUTED
    assert result.statistic is not None
    assert result.cramers_v is not None
    assert math.isclose(result.statistic, 0.0, abs_tol=1e-9)
    assert math.isclose(result.cramers_v, 0.0, abs_tol=1e-9)


def test_known_answer_case() -> None:
    # baseline: A=50, B=50 (n=100); live: A=60, B=40 (n=100).
    # Contingency table [[50, 50], [60, 40]], expected frequencies under
    # independence: row_total * col_total / grand_total.
    baseline = ["A"] * 50 + ["B"] * 50
    live = ["A"] * 60 + ["B"] * 40

    result = chi_square(baseline, live, CATEGORIES)

    expected_baseline_a = 100 * 110 / 200
    expected_baseline_b = 100 * 90 / 200
    expected_live_a = 100 * 110 / 200
    expected_live_b = 100 * 90 / 200
    expected_statistic = (
        (50 - expected_baseline_a) ** 2 / expected_baseline_a
        + (50 - expected_baseline_b) ** 2 / expected_baseline_b
        + (60 - expected_live_a) ** 2 / expected_live_a
        + (40 - expected_live_b) ** 2 / expected_live_b
    )
    assert result.status == StatStatus.COMPUTED
    assert result.statistic is not None
    assert result.cramers_v is not None
    assert math.isclose(result.statistic, expected_statistic, rel_tol=1e-9)

    expected_cramers_v = math.sqrt(expected_statistic / (200 * min(2 - 1, 2 - 1)))
    assert math.isclose(result.cramers_v, expected_cramers_v, rel_tol=1e-9)


def test_unseen_category_is_drift_not_error() -> None:
    baseline = ["A"] * 50 + ["B"] * 50
    live = ["A"] * 50 + ["C"] * 50  # "C" was never in the baseline's frozen categories

    result = chi_square(baseline, live, CATEGORIES)

    assert result.status == StatStatus.COMPUTED
    assert result.statistic is not None
    assert result.cramers_v is not None
    assert math.isfinite(result.statistic)
    assert result.cramers_v > 0


def test_empty_baseline_is_not_computable_not_raising() -> None:
    result = chi_square([], ["A"], CATEGORIES)

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.statistic is None
    assert result.p_value is None
    assert result.cramers_v is None
    assert result.not_computable_reason


def test_degenerate_single_populated_category_is_not_computable_not_raising() -> None:
    # baseline and live both entirely in category "A" -- only one column of the
    # contingency table has any data, so no meaningful chi-square exists.
    result = chi_square(["A"] * 10, ["A"] * 10, CATEGORIES)

    assert result.status == StatStatus.NOT_COMPUTABLE
    assert result.statistic is None
    assert result.not_computable_reason


def test_is_pure_and_deterministic() -> None:
    baseline = ["A", "B", "A", "B", "A"]
    live = ["B", "A", "B", "A", "B"]

    first = chi_square(baseline, live, CATEGORIES)
    second = chi_square(list(reversed(baseline)), list(reversed(live)), CATEGORIES)

    assert first == second
    assert baseline == ["A", "B", "A", "B", "A"]
