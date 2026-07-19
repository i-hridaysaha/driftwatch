"""Pure data generation: generate_scenario_data(scenario) -> ScenarioData.

No I/O, no wall-clock reads (every timestamp is scenario.start_date plus a
deterministic offset), and exactly one numpy.random.Generator, seeded once
from scenario.seed, drawn from in a fixed order (baseline first, then
window 0..N-1 in order, and within each window: segments, then features in
their declared order, then prediction score, then label + delay). Same
scenario, same seed, same output -- every time, regardless of when this
runs. See tests/demo/test_generator_determinism.py, which is exactly what
proves that claim rather than just asserting it in a docstring.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from driftwatch.demo.distributions import (
    sample_bernoulli,
    sample_categories,
    sample_distribution,
    sample_high_cardinality_ids,
    sample_missing_mask,
)
from driftwatch.demo.schema import EventSpec, ScenarioConfig, ScenarioFeatureSpec
from driftwatch.durations import parse_duration


@dataclass(frozen=True)
class BaselineRecordData:
    features: dict[str, Any]
    prediction_score: float
    segment_values: dict[str, str]


@dataclass(frozen=True)
class PredictionData:
    prediction_id: str
    predicted_at: datetime
    features: dict[str, Any]
    prediction_score: float
    segment_values: dict[str, str]


@dataclass(frozen=True)
class LabelData:
    prediction_id: str
    label_value: float
    labeled_at: datetime
    delay_hours: float


@dataclass(frozen=True)
class ScenarioData:
    baseline_records: list[BaselineRecordData] = field(default_factory=list)
    predictions: list[PredictionData] = field(default_factory=list)
    labels: list[LabelData] = field(default_factory=list)


def _event_active(event: EventSpec, window: int) -> bool:
    if window < event.start_window:
        return False
    if event.duration_windows is None:
        return True
    return window < event.start_window + event.duration_windows


def _shift_reference_scale(feature: ScenarioFeatureSpec) -> float:
    """The "one unit" a mean_shift's magnitude is expressed in: std for a
    normal distribution, sigma for lognormal (applied in the same linear
    space numpy samples in either case -- not a claim that a lognormal
    shift is exactly N sigma in log-space, just a consistent, documented
    convention for how far to move the sampled values)."""
    assert feature.distribution is not None
    scale = feature.distribution.std or feature.distribution.sigma
    if scale is None:
        raise ValueError(
            f"feature {feature.name!r} has no std/sigma to express a mean_shift magnitude in"
        )
    return scale


def _sample_continuous_feature(
    rng: np.random.Generator,
    feature: ScenarioFeatureSpec,
    n: int,
    events: list[EventSpec],
    segment_values: dict[str, np.ndarray],
) -> np.ndarray:
    assert feature.distribution is not None
    values = sample_distribution(rng, feature.distribution, n)
    scale = _shift_reference_scale(feature)
    for event in events:
        if event.shift_type != "mean_shift" or event.feature != feature.name:
            continue
        if event.segment_dimension is None:
            values = values + event.magnitude * scale
        else:
            mask = segment_values[event.segment_dimension] == event.segment_value
            values = np.where(mask, values + event.magnitude * scale, values)
    return values


def _sample_segment_assignments(
    rng: np.random.Generator, scenario: ScenarioConfig, n: int
) -> dict[str, np.ndarray]:
    by_name = {f.name: f for f in scenario.features}
    assignments: dict[str, np.ndarray] = {}
    for dim in scenario.segments:
        spec = by_name[dim]
        assert spec.categories is not None
        assignments[dim] = sample_categories(rng, spec.categories, spec.weights, n)
    return assignments


def _sample_categorical_feature(
    rng: np.random.Generator,
    feature: ScenarioFeatureSpec,
    n: int,
    segment_values: dict[str, np.ndarray],
) -> np.ndarray:
    # A feature also used as a segment dimension was already sampled once
    # (segment assignment must happen before any segment-scoped mean_shift
    # can be applied to other features) -- reuse that draw rather than
    # sampling it a second time, which would both waste an RNG draw and
    # desynchronize the segment_values used for masking from the feature's
    # own recorded value.
    if feature.name in segment_values:
        return segment_values[feature.name]
    if feature.cardinality is not None:
        return sample_high_cardinality_ids(rng, feature.cardinality, n)
    assert feature.categories is not None
    return sample_categories(rng, feature.categories, feature.weights, n)


def _apply_missing(
    rng: np.random.Generator, feature: ScenarioFeatureSpec, values: np.ndarray
) -> np.ndarray:
    if feature.missing_rate <= 0:
        return values
    mask = sample_missing_mask(rng, feature.missing_rate, len(values))
    out = values.astype(object)
    out[mask] = None
    return out


def _to_native(value: Any) -> Any:
    """Numeric values (continuous features) become plain `float`; anything
    else (categorical values, high-cardinality ids) becomes plain `str`.
    Deliberately checked as "is this any numeric type" rather than "is
    this specifically a numpy scalar" -- after `_apply_missing`'s
    `.astype(object)`, a numpy float64 array's surviving elements are
    already plain Python `float`, not `np.floating`, so a check that only
    recognized numpy's own scalar types would silently fall through to
    the string branch and stringify real continuous values."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return str(value)


def _generate_baseline(
    rng: np.random.Generator, scenario: ScenarioConfig
) -> list[BaselineRecordData]:
    n = scenario.baseline_size
    segment_values = _sample_segment_assignments(rng, scenario, n)
    feature_columns: dict[str, np.ndarray] = {}
    for feature in scenario.features:
        if feature.dtype == "continuous":
            raw = _sample_continuous_feature(rng, feature, n, [], segment_values)
        else:
            raw = _sample_categorical_feature(rng, feature, n, segment_values)
        feature_columns[feature.name] = _apply_missing(rng, feature, raw)
    scores = sample_distribution(rng, scenario.prediction_distribution, n)
    scores = np.clip(scores, 0.0, 1.0)

    records = []
    for i in range(n):
        features = {name: _to_native(col[i]) for name, col in feature_columns.items()}
        seg = {dim: str(segment_values[dim][i]) for dim in scenario.segments}
        records.append(
            BaselineRecordData(
                features=features, prediction_score=float(scores[i]), segment_values=seg
            )
        )
    return records


def _generate_window(
    rng: np.random.Generator,
    scenario: ScenarioConfig,
    window_index: int,
    window_start: datetime,
    window_duration: timedelta,
    prediction_id_start: int,
) -> tuple[list[PredictionData], list[LabelData]]:
    n = scenario.predictions_per_window
    active_events = [e for e in scenario.events if _event_active(e, window_index)]

    segment_values = _sample_segment_assignments(rng, scenario, n)
    feature_columns: dict[str, np.ndarray] = {}
    for feature in scenario.features:
        if feature.dtype == "continuous":
            raw = _sample_continuous_feature(rng, feature, n, active_events, segment_values)
        else:
            raw = _sample_categorical_feature(rng, feature, n, segment_values)
        feature_columns[feature.name] = _apply_missing(rng, feature, raw)

    scores = np.clip(sample_distribution(rng, scenario.prediction_distribution, n), 0.0, 1.0)

    offsets_seconds = np.sort(rng.uniform(0, window_duration.total_seconds(), n))

    noise_rate = scenario.label_base_noise_rate
    for event in active_events:
        if event.shift_type == "label_noise":
            noise_rate = min(0.99, noise_rate + event.magnitude)

    positive_draws = sample_bernoulli(rng, scores)
    flip_draws = sample_bernoulli(rng, np.full(n, noise_rate))
    label_values = np.where(flip_draws, ~positive_draws, positive_draws)

    delay_hours = np.clip(
        rng.lognormal(np.log(scenario.label_delay.median_hours), scenario.label_delay.sigma, n),
        0.0,
        scenario.label_delay.max_hours,
    )

    predictions: list[PredictionData] = []
    labels: list[LabelData] = []
    for i in range(n):
        prediction_id = f"{scenario.name}-p{prediction_id_start + i:07d}"
        predicted_at = window_start + timedelta(seconds=float(offsets_seconds[i]))
        features = {name: _to_native(col[i]) for name, col in feature_columns.items()}
        seg = {dim: str(segment_values[dim][i]) for dim in scenario.segments}
        predictions.append(
            PredictionData(
                prediction_id=prediction_id,
                predicted_at=predicted_at,
                features=features,
                prediction_score=float(scores[i]),
                segment_values=seg,
            )
        )
        labels.append(
            LabelData(
                prediction_id=prediction_id,
                label_value=float(bool(label_values[i])),
                labeled_at=predicted_at + timedelta(hours=float(delay_hours[i])),
                delay_hours=float(delay_hours[i]),
            )
        )
    return predictions, labels


def generate_scenario_data(scenario: ScenarioConfig) -> ScenarioData:
    rng = np.random.default_rng(scenario.seed)
    window_duration = parse_duration(scenario.window)
    start_date = scenario.start_date.astimezone(UTC)

    baseline_records = _generate_baseline(rng, scenario)

    predictions: list[PredictionData] = []
    labels: list[LabelData] = []
    next_id = 0
    for window_index in range(scenario.total_windows):
        window_start = start_date + window_index * window_duration
        window_predictions, window_labels = _generate_window(
            rng, scenario, window_index, window_start, window_duration, next_id
        )
        predictions.extend(window_predictions)
        labels.extend(window_labels)
        next_id += len(window_predictions)

    return ScenarioData(baseline_records=baseline_records, predictions=predictions, labels=labels)
