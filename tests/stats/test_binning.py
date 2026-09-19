import math

import numpy as np

from driftwatch.config.schema import (
    FeatureSpec,
    LabelFieldSpec,
    ModelSchemaSpec,
    PredictionFieldSpec,
)
from driftwatch.stats.binning import (
    bucket_proportions,
    compute_baseline_binning,
    compute_categorical_categories,
    compute_continuous_edges,
    psi_null_floor_table,
)
from driftwatch.stats.psi import NULL_FLOOR_GRID


def test_continuous_edges_span_exactly_min_to_max() -> None:
    edges = compute_continuous_edges(list(range(1, 101)), n_bins=10)

    assert edges[0] == 1.0
    assert edges[-1] == 100.0
    assert len(edges) == 11  # 9 interior quantile points + the two endpoints


def test_continuous_edges_single_distinct_value_is_defined_not_raising() -> None:
    edges = compute_continuous_edges([5.0, 5.0, 5.0])

    assert edges == [5.0, 5.0]


def test_continuous_edges_empty_input_is_defined_not_raising() -> None:
    assert compute_continuous_edges([]) == []


def test_continuous_edges_deduplicates_repeated_quantiles_for_skewed_data() -> None:
    # 50 of 60 values (83%) are identical -- most decile cutoffs fall inside that
    # cluster and collapse to the same edge, so far fewer than 11 edges survive.
    skewed_values = [1.0] * 50 + [float(v) for v in range(2, 12)]

    edges = compute_continuous_edges(skewed_values, n_bins=10)

    assert len(edges) < 11
    assert len(edges) == len(set(edges))  # no duplicate edges survive dedup


def test_categorical_categories_sorted_and_deduplicated() -> None:
    categories = compute_categorical_categories(["US", "EU", "US", "APAC", "EU"])

    assert categories == ["APAC", "EU", "US"]


def test_bucket_proportions_underflow_and_overflow_are_explicit() -> None:
    edges = [1.0, 2.0, 3.0]  # 2 interior bins -> 4 total buckets: under, (1,2], (2,3], over

    proportions = bucket_proportions([0.0, 1.5, 2.5, 100.0], edges)

    assert len(proportions) == 4
    assert proportions == [0.25, 0.25, 0.25, 0.25]
    assert math.isclose(sum(proportions), 1.0)


def test_bucket_proportions_baseline_endpoints_are_interior_not_overflow() -> None:
    edges = [1.0, 2.0, 3.0]

    # values exactly at the frozen min/max must land inside the range they defined,
    # not in the overflow buckets -- they came from the baseline itself.
    proportions = bucket_proportions([1.0, 3.0], edges)

    assert proportions[0] == 0.0  # underflow empty
    assert proportions[-1] == 0.0  # overflow empty


def test_bucket_proportions_empty_input_is_defined_not_raising() -> None:
    assert bucket_proportions([], [1.0, 2.0]) == []
    assert bucket_proportions([1.0], []) == []


def test_compute_baseline_binning_all_null_feature_is_defined_not_raising() -> None:
    schema = ModelSchemaSpec(
        features=[FeatureSpec(name="age", dtype="continuous", nullable=True)],
        prediction=PredictionFieldSpec(dtype="continuous"),
        label=LabelFieldSpec(dtype="categorical"),
    )

    binning = compute_baseline_binning(
        features=[{"age": None}, {"age": None}],
        prediction_scores=[0.5, 0.6],
        schema=schema,
    )

    assert binning["features"]["age"] == {
        "type": "continuous",
        "edges": [],
        "effective_bins": 0,
        "null_floor": {"sizes": [], "mean": [], "ceiling": []},
    }


def test_effective_bins_reflects_post_dedup_count_not_nominal_n_bins() -> None:
    schema = ModelSchemaSpec(
        features=[FeatureSpec(name="skewed", dtype="continuous", nullable=False)],
        prediction=PredictionFieldSpec(dtype="continuous"),
        label=LabelFieldSpec(dtype="categorical"),
    )
    skewed_values = [1.0] * 50 + [float(v) for v in range(2, 12)]

    binning = compute_baseline_binning(
        features=[{"skewed": v} for v in skewed_values],
        prediction_scores=[0.5] * len(skewed_values),
        schema=schema,
        n_bins=10,
    )

    entry = binning["features"]["skewed"]
    assert entry["effective_bins"] == len(entry["edges"]) - 1
    assert entry["effective_bins"] < 10  # fewer than the nominal bin count requested


def test_null_floor_tables_are_simulated_globally_and_per_segment_slice() -> None:
    """Every continuous feature carries a simulated PSI null table on the
    global baseline and on each segment slice, sliced by the declared
    segment dimension (a categorical feature, never itself tabled), and
    the ceiling falls as the live size grows."""
    schema = ModelSchemaSpec(
        features=[
            FeatureSpec(name="age", dtype="continuous", nullable=False),
            FeatureSpec(name="region", dtype="categorical", nullable=False),
        ],
        prediction=PredictionFieldSpec(dtype="continuous"),
        label=LabelFieldSpec(dtype="categorical"),
    )
    rng = np.random.default_rng(0)
    records = [
        {"age": float(rng.normal(40, 12)), "region": ["EU", "US", "APAC"][i % 3]}
        for i in range(600)
    ]
    binning = compute_baseline_binning(
        features=records,
        prediction_scores=[0.5] * 600,
        schema=schema,
        segment_dimensions=["region"],
    )

    table = binning["features"]["age"]["null_floor"]
    assert table["sizes"] == [float(n) for n in NULL_FLOOR_GRID]
    assert all(a >= b for a, b in zip(table["ceiling"], table["ceiling"][1:], strict=False))
    assert set(binning["segments"]["region"]) == {"APAC", "EU", "US"}
    apac = binning["segments"]["region"]["APAC"]["features"]
    assert set(apac) == {"age"}  # region is not tabled inside its own segment
    assert psi_null_floor_table(binning, "age", "region", "APAC") is apac["age"]["null_floor"]
    assert psi_null_floor_table(binning, "age", None, None) is table
    assert psi_null_floor_table(binning, "region", None, None) is None
    # a 200-row slice is noisier than the 600-row whole at the same live size
    from driftwatch.stats.psi import psi_null_ceiling_from_table

    assert psi_null_ceiling_from_table(apac["age"]["null_floor"], 75) > (
        psi_null_ceiling_from_table(table, 75) or 0.0
    )
