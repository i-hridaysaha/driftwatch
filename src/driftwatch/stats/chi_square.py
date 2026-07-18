import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats

from driftwatch.stats.result import StatStatus


@dataclass(frozen=True)
class ChiSquareResult:
    status: StatStatus
    statistic: float | None
    p_value: float | None
    cramers_v: float | None
    not_computable_reason: str | None = None


def _category_counts(values: Sequence[str], categories: Sequence[str]) -> list[int]:
    counts = dict.fromkeys(categories, 0)
    unseen = 0
    for value in values:
        if value in counts:
            counts[value] += 1
        else:
            unseen += 1
    return [counts[category] for category in categories] + [unseen]


def chi_square(
    baseline_values: Sequence[str], live_values: Sequence[str], categories: Sequence[str]
) -> ChiSquareResult:
    """Chi-square test of independence between a live sample and the baseline,
    plus Cramer's V as the effect-size measure alert thresholds are actually
    compared against.

    `categories` must be the baseline's own frozen category list (see
    driftwatch.stats.binning.compute_categorical_categories), computed once
    at registration. A live value not in that list is bucketed as "unseen"
    -- a new category appearing after registration is drift, not an error,
    the same principle as the continuous overflow buckets in psi().

    Cramer's V is bounded [0, 1] regardless of sample size or category
    count, unlike the raw chi-square statistic (which grows with N) or the
    p-value (which collapses toward zero at production sample sizes for the
    same reason documented in driftwatch.stats.ks) -- see
    CategoricalDriftConfig.cramers_v_threshold, which is what alerting
    actually gates on. Computed here as sqrt(chi2 / (n * min(r-1, c-1))),
    with correction=False passed to scipy's chi2_contingency: Yates'
    continuity correction is a 2x2-specific small-sample adjustment that
    doesn't generalize to the arbitrary category counts this function
    handles, so it's left off uniformly rather than applied inconsistently
    depending on how many categories a feature happens to have.

    Returns status=NOT_COMPUTABLE with a reason -- never a bare nan -- if
    either sample is empty, or if fewer than two categories have any
    observations at all (a degenerate contingency table chi-square cannot
    meaningfully be computed from). Same status/reason convention as psi()
    and driftwatch.db.models.PerformanceResult.
    """
    if not baseline_values:
        return ChiSquareResult(
            status=StatStatus.NOT_COMPUTABLE,
            statistic=None,
            p_value=None,
            cramers_v=None,
            not_computable_reason="baseline sample is empty",
        )
    if not live_values:
        return ChiSquareResult(
            status=StatStatus.NOT_COMPUTABLE,
            statistic=None,
            p_value=None,
            cramers_v=None,
            not_computable_reason="live sample is empty",
        )

    baseline_counts = _category_counts(baseline_values, categories)
    live_counts = _category_counts(live_values, categories)
    table = np.array([baseline_counts, live_counts])

    nonzero_columns = table.sum(axis=0) > 0
    table = table[:, nonzero_columns]
    if table.shape[1] < 2:
        return ChiSquareResult(
            status=StatStatus.NOT_COMPUTABLE,
            statistic=None,
            p_value=None,
            cramers_v=None,
            not_computable_reason="fewer than two categories have any observations",
        )

    statistic, p_value, _, _ = scipy_stats.chi2_contingency(table, correction=False)
    n = int(table.sum())
    rows, cols = table.shape
    cramers_v = math.sqrt(float(statistic) / (n * min(rows - 1, cols - 1)))

    return ChiSquareResult(
        status=StatStatus.COMPUTED,
        statistic=float(statistic),
        p_value=float(p_value),
        cramers_v=cramers_v,
    )
