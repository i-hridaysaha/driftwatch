from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from driftwatch.durations import parse_duration


class StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


class DistributionSpec(StrictModel):
    """A flat, all-optional bag rather than a discriminated union -- this
    is scenario-authoring config for an internal demo tool, not a public
    API; validity of "the right fields are set for this type" is checked
    once, here, rather than modeled as a dozen near-identical pydantic
    subclasses."""

    type: Literal["normal", "lognormal", "beta", "uniform"]
    mean: float | None = None
    std: float | None = None
    sigma: float | None = None
    a: float | None = None
    b: float | None = None
    low: float | None = None
    high: float | None = None
    loc: float = 0.0
    scale: float = 1.0

    @model_validator(mode="after")
    def _required_fields_present(self) -> "DistributionSpec":
        required: dict[str, tuple[str, ...]] = {
            "normal": ("mean", "std"),
            "lognormal": ("mean", "sigma"),
            "beta": ("a", "b"),
            "uniform": ("low", "high"),
        }
        missing = [f for f in required[self.type] if getattr(self, f) is None]
        if missing:
            raise ValueError(f"distribution type {self.type!r} requires fields {missing}")
        return self


class ScenarioFeatureSpec(StrictModel):
    name: str
    dtype: Literal["continuous", "categorical"]
    nullable: bool = False
    missing_rate: float = Field(default=0.0, ge=0, lt=1)
    # continuous
    distribution: DistributionSpec | None = None
    # categorical
    categories: list[str] | None = None
    weights: list[float] | None = None
    cardinality: int | None = None
    """Set instead of categories/weights for a high-cardinality feature --
    sampled uniformly from `cardinality` synthetic ids rather than listed
    out by hand."""

    @model_validator(mode="after")
    def _shape_matches_dtype(self) -> "ScenarioFeatureSpec":
        if self.dtype == "continuous":
            if self.distribution is None:
                raise ValueError(f"feature {self.name!r}: continuous features need `distribution`")
            if self.categories is not None or self.cardinality is not None:
                raise ValueError(
                    f"feature {self.name!r}: continuous features can't set categories/cardinality"
                )
        else:
            has_categories = self.categories is not None
            has_cardinality = self.cardinality is not None
            if has_categories == has_cardinality:
                raise ValueError(
                    f"feature {self.name!r}: categorical features need exactly one of "
                    "`categories` or `cardinality`"
                )
            if self.weights is not None and not has_categories:
                raise ValueError(
                    f"feature {self.name!r}: `weights` needs `categories`, not `cardinality`"
                )
            if (
                self.weights is not None
                and self.categories is not None
                and len(self.weights) != len(self.categories)
            ):
                raise ValueError(f"feature {self.name!r}: `weights` must match `categories` length")
            if self.missing_rate > 0 and not self.nullable:
                raise ValueError(f"feature {self.name!r}: missing_rate > 0 requires nullable: true")
        if self.dtype == "continuous" and self.missing_rate > 0 and not self.nullable:
            raise ValueError(f"feature {self.name!r}: missing_rate > 0 requires nullable: true")
        return self


class LabelDelaySpec(StrictModel):
    """Every prediction eventually gets exactly one label, arriving
    `delay_hours` after its own predicted_at, delay_hours ~
    lognormal(median=median_hours, sigma), clipped to max_hours."""

    median_hours: float = Field(gt=0)
    sigma: float = Field(gt=0)
    max_hours: float = Field(gt=0)
    early_cutoff_hours: float = Field(gt=0)
    """Labels with delay_hours <= this are inserted BEFORE the scenario's
    evaluate_range backfill call, so a window's INITIAL performance
    already reflects them; labels with a longer delay are inserted AFTER,
    triggering a genuine retroactive recompute for that window. This is
    what gives the dashboard's before/after-label-backfill split real
    content to show, deterministically, rather than hoping some labels
    happen to straddle the evaluation by chance."""


class EventSpec(StrictModel):
    feature: str | None = None
    """Required for shift_type=mean_shift (must name a continuous feature
    declared above); must be omitted for shift_type=label_noise, which
    applies to the label-generation process itself, not to any feature."""
    shift_type: Literal["mean_shift", "label_noise"]
    magnitude: float
    """mean_shift: additive shift, in units of the feature's own
    distribution std/sigma (i.e. magnitude=3 means "shift the mean by 3
    standard deviations"). label_noise: additional probability of
    flipping a label from what the (stable) score-to-label process would
    have produced, on top of the scenario's baseline label_noise_rate."""
    start_window: int = Field(ge=0)
    duration_windows: int | None = None
    """None = sustained through the end of the scenario."""
    segment_dimension: str | None = None
    segment_value: str | None = None
    """If set, the shift only applies to predictions in this segment --
    both must be set together, and only meaningful for mean_shift."""

    @model_validator(mode="after")
    def _shape_matches_shift_type(self) -> "EventSpec":
        if self.shift_type == "mean_shift" and self.feature is None:
            raise ValueError("mean_shift events require `feature`")
        if self.shift_type == "label_noise":
            if self.feature is not None:
                raise ValueError("label_noise events must not set `feature`")
            if self.segment_dimension is not None:
                raise ValueError("label_noise events must not set a segment scope")
        if (self.segment_dimension is None) != (self.segment_value is None):
            raise ValueError("segment_dimension and segment_value must be set together")
        return self


class ScenarioConfig(StrictModel):
    name: str
    description: str
    model_id: str
    profile: Literal["aggressive", "patient"]
    seed: int
    start_date: datetime
    window: str
    stable_windows: int = Field(ge=1)
    """Windows before ANY injected event may start -- every event's
    start_window must be >= this."""
    total_windows: int = Field(gt=0)
    predictions_per_window: int = Field(gt=0)
    baseline_size: int = Field(gt=0)
    prediction_type: Literal["binary_classification"]
    prediction_distribution: DistributionSpec
    features: list[ScenarioFeatureSpec]
    segments: list[str] = Field(default_factory=list)
    min_segment_size: int = Field(gt=0, default=30)
    label_base_noise_rate: float = Field(ge=0, lt=1)
    """P(label flip) even with no active label_noise event -- real label
    noise, not a knob to disable."""
    label_delay: LabelDelaySpec
    events: list[EventSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _start_date_is_utc(self) -> "ScenarioConfig":
        if self.start_date.tzinfo is None:
            raise ValueError("start_date must be timezone-aware (UTC)")
        return self

    @model_validator(mode="after")
    def _start_date_is_window_aligned(self) -> "ScenarioConfig":
        """driftwatch.scheduler.windowing.compute_window_boundaries aligns
        real evaluation windows to the Unix epoch, not to this scenario's
        start_date. The generator groups predictions into "window index W"
        as [start_date + W*duration, start_date + (W+1)*duration) -- if
        start_date itself isn't epoch-aligned for `window`, those generated
        groupings won't coincide with the real evaluated windows at all, so
        a single generator window could straddle two real ones (or vice
        versa), silently breaking the mapping between "the window an event
        was injected into" and "the window that actually gets evaluated".
        Midnight UTC is always aligned for any window duration that evenly
        divides 24h (1h, 2h, 3h, 4h, 6h, 12h, 1d)."""
        duration = parse_duration(self.window)
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        offset = (self.start_date - epoch) % duration
        if offset != timedelta(0):
            raise ValueError(
                f"start_date {self.start_date.isoformat()} is not aligned to a "
                f"{self.window} window boundary (epoch-relative remainder {offset}); "
                "use a start_date that falls exactly on one (e.g. midnight UTC)"
            )
        return self

    @model_validator(mode="after")
    def _events_reference_real_features_and_segments(self) -> "ScenarioConfig":
        feature_names = {f.name for f in self.features}
        segment_dims = set(self.segments)
        for event in self.events:
            if event.start_window < self.stable_windows:
                raise ValueError(
                    f"event on {event.feature!r} starts at window {event.start_window}, "
                    f"before stable_windows ({self.stable_windows})"
                )
            if event.start_window >= self.total_windows:
                raise ValueError(
                    f"event on {event.feature!r} starts at window {event.start_window}, "
                    f"at or past total_windows ({self.total_windows})"
                )
            if event.feature is not None and event.feature not in feature_names:
                raise ValueError(f"event references unknown feature {event.feature!r}")
            if event.segment_dimension is not None and event.segment_dimension not in segment_dims:
                raise ValueError(
                    f"event references unknown segment dimension {event.segment_dimension!r}"
                )
        return self

    @model_validator(mode="after")
    def _segments_reference_categorical_features(self) -> "ScenarioConfig":
        by_name = {f.name: f for f in self.features}
        for dim in self.segments:
            if dim not in by_name:
                raise ValueError(f"segment dimension {dim!r} is not a declared feature")
            if by_name[dim].dtype != "categorical":
                raise ValueError(f"segment dimension {dim!r} must be a categorical feature")
        return self
