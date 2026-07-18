from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BenjaminiHochbergResult:
    adjusted_p_values: list[float]
    is_significant: list[bool]


def benjamini_hochberg(p_values: Sequence[float], alpha: float) -> BenjaminiHochbergResult:
    """Benjamini-Hochberg step-up procedure for controlling the false discovery
    rate across multiple hypothesis tests.

    Scope, which callers must respect: this is applied across the
    feature-level p-values produced by a SINGLE model's SINGLE evaluation
    window, and nothing broader. Pooling p-values across different models,
    or across different time windows for the same model, would conflate
    unrelated hypotheses into one family and make "m" (the family size used
    in the i/m*alpha threshold) meaningless -- a model with 50 features
    evaluated over 30 windows is not one experiment with 1500 comparisons,
    it's 30 independent experiments of 50 comparisons each. Each call to
    this function is exactly one such experiment: one window's worth of
    per-feature test results, nothing else mixed in.

    Both outputs are returned in the same order as the input `p_values`, not
    sorted order. Returns two empty lists (not an exception) for an empty
    input.
    """
    m = len(p_values)
    if m == 0:
        return BenjaminiHochbergResult(adjusted_p_values=[], is_significant=[])

    order = sorted(range(m), key=lambda i: p_values[i])
    sorted_p = [p_values[i] for i in order]

    # Adjusted p-values (q-values): q_(i) = min_{j>=i} (m/j * p_(j)), computed as a
    # running minimum from the largest p-value down to the smallest, capped at 1.0.
    adjusted_sorted = [0.0] * m
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        candidate = sorted_p[i] * m / (i + 1)
        running_min = min(running_min, candidate)
        adjusted_sorted[i] = min(running_min, 1.0)

    # Significance is a step-up decision, not a per-rank one: find the LARGEST
    # rank i where sorted_p[i] <= (i+1)/m*alpha, then every p-value at or below
    # that rank is significant -- including ranks that individually failed
    # their own threshold, as long as a later (larger) rank still passed.
    largest_significant_rank = -1
    for i in range(m):
        threshold = (i + 1) / m * alpha
        if sorted_p[i] <= threshold:
            largest_significant_rank = i
    significant_sorted = [i <= largest_significant_rank for i in range(m)]

    adjusted_p_values = [0.0] * m
    is_significant = [False] * m
    for rank, original_index in enumerate(order):
        adjusted_p_values[original_index] = adjusted_sorted[rank]
        is_significant[original_index] = significant_sorted[rank]

    return BenjaminiHochbergResult(
        adjusted_p_values=adjusted_p_values, is_significant=is_significant
    )
