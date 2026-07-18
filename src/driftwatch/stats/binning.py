from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from driftwatch.config.schema import ModelSchemaSpec

DEFAULT_N_BINS = 10


def compute_continuous_edges(values: Sequence[float], n_bins: int = DEFAULT_N_BINS) -> list[float]:
    """Quantile-based bin edges spanning exactly [min(values), max(values)].

    The returned edges are frozen at baseline registration time and reused
    for every future evaluation window -- they are never recomputed from live
    data, so PSI/JSD values stay comparable across windows and across time.

    A value observed later that falls below edges[0] or above edges[-1] is,
    by construction, outside anything the baseline ever produced (the edges
    ARE the baseline's own min and max). bucket_proportions() below treats
    those as explicit overflow/underflow buckets rather than clipping them
    into the nearest real bin or raising -- a live value outside the
    baseline's range is itself a drift signal, not an error condition.
    """
    if not values:
        return []
    ordered = sorted(values)
    lo, hi = ordered[0], ordered[-1]
    if lo == hi:
        return [float(lo), float(hi)]
    quantile_points = [100 * i / n_bins for i in range(1, n_bins)]
    interior = [float(np.percentile(ordered, q)) for q in quantile_points]
    edges = [float(lo), *interior, float(hi)]
    # Heavily skewed or low-cardinality data can produce repeated quantiles;
    # collapse those into a single edge so no bin has zero width.
    deduped = [edges[0]]
    for edge in edges[1:]:
        if edge > deduped[-1]:
            deduped.append(edge)
    return deduped


def _continuous_bin_entry(values: Sequence[float], n_bins: int) -> dict[str, Any]:
    """A continuous feature's binning entry, including `effective_bins` --
    the ACTUAL number of usable bins after quantile deduplication, which can
    be far fewer than `n_bins` for skewed or low-cardinality data (e.g. a
    feature that's the same value 90% of the time collapses most of its
    quantile edges together). Recording this explicitly means a feature with
    3 effective bins is never silently treated as if it had the nominal 10 --
    anything consuming this config can check effective_bins directly instead
    of re-deriving it from len(edges) - 1 itself."""
    edges = compute_continuous_edges(values, n_bins)
    return {"type": "continuous", "edges": edges, "effective_bins": max(len(edges) - 1, 0)}


def compute_categorical_categories(values: Sequence[str]) -> list[str]:
    """The frozen set of categories observed in the baseline, sorted for determinism.

    Any category seen live but absent here is bucketed as "unseen" by the
    chi-square test -- new categories are drift, not errors, mirroring the
    continuous-feature overflow-bucket treatment above.
    """
    return sorted(set(values))


def _assign_bucket(value: float, edges: Sequence[float]) -> int:
    """Bucket 0 = below the baseline's range, len(edges) = above it,
    1..len(edges)-1 = interior bins. Both baseline endpoints (edges[0],
    edges[-1]) count as inside the range, since they were derived from the
    baseline's own data."""
    idx = int(np.digitize([value], edges, right=False)[0])
    if idx == len(edges) and value == edges[-1]:
        # numpy.digitize treats a value equal to the last edge as "above the
        # range"; the baseline's own maximum is not out-of-range drift.
        idx = len(edges) - 1
    return idx


def bucket_proportions(values: Sequence[float], edges: Sequence[float]) -> list[float]:
    """Proportion of `values` landing in each of len(edges)+1 buckets
    (underflow, interior bins, overflow), using frozen `edges`."""
    if not values or not edges:
        return []
    n_buckets = len(edges) + 1
    counts = [0] * n_buckets
    for value in values:
        counts[_assign_bucket(value, edges)] += 1
    total = len(values)
    return [count / total for count in counts]


def compute_baseline_binning(
    features: Sequence[Mapping[str, Any]],
    prediction_scores: Sequence[float | None],
    schema: ModelSchemaSpec,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, Any]:
    """Compute and freeze the binning configuration for a newly registered
    baseline. Called exactly once, at registration -- never re-derived from
    live data, so drift test results stay comparable across the baseline's
    entire lifetime."""
    feature_bins: dict[str, Any] = {}
    for feature in schema.features:
        raw_values = [record.get(feature.name) for record in features]
        present = [v for v in raw_values if v is not None]
        if feature.dtype == "continuous":
            feature_bins[feature.name] = _continuous_bin_entry([float(v) for v in present], n_bins)
        else:
            feature_bins[feature.name] = {
                "type": "categorical",
                "categories": compute_categorical_categories([str(v) for v in present]),
            }

    scores = [score for score in prediction_scores if score is not None]
    prediction_bins = _continuous_bin_entry(scores, n_bins)

    return {"features": feature_bins, "prediction_score": prediction_bins}
