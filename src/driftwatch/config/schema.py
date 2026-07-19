from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from driftwatch.durations import parse_duration


class StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


class EvaluationConfig(StrictModel):
    window: str
    min_window_size: int = Field(gt=0)
    watermark: str
    """How long after window_end to wait before a window's DRIFT statistics
    become eligible for their one-time, never-recomputed evaluation -- see
    driftwatch.scheduler.windowing for the full reasoning on why drift is
    computed once on a fixed delay rather than retroactively re-evaluated on
    late-arriving predictions. Governs drift only: performance metrics for
    the same window are a separate, indefinitely-revisable lifecycle driven
    by label arrival, not by this watermark -- see
    EvaluationWindow.label_watermark and
    driftwatch.evaluation.performance.recompute_performance_for_window."""

    @field_validator("window", "watermark")
    @classmethod
    def _must_be_parseable_duration(cls, value: str) -> str:
        parse_duration(value)  # raises ValueError with a clear message if malformed
        return value


class ContinuousDriftConfig(StrictModel):
    """Alert thresholds here are effect sizes, not significance levels. At
    production window sizes the p-value from a KS test collapses toward zero
    regardless of whether the shift is operationally meaningful, so alerting
    is gated on the KS D-statistic itself (bounded [0, 1]) -- the p-value is
    still computed and reported, but never drives the decision. See
    driftwatch.stats.ks for the full reasoning.

    Each test also has a *_clear_threshold, strictly below its fire
    threshold: hysteresis. A statistic sitting right at the fire threshold
    would otherwise flap open/resolved/open every window on pure noise --
    see driftwatch.alerting.streaks for the three-way breach/dead_zone/clear
    classification this pair of thresholds drives.
    """

    methods: list[Literal["psi", "ks"]]
    psi_threshold: float = Field(gt=0)
    psi_clear_threshold: float = Field(gt=0)
    ks_statistic_threshold: float = Field(gt=0, lt=1)
    ks_statistic_clear_threshold: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _clear_below_fire(self) -> "ContinuousDriftConfig":
        if self.psi_clear_threshold >= self.psi_threshold:
            raise ValueError("psi_clear_threshold must be < psi_threshold (hysteresis gap)")
        if self.ks_statistic_clear_threshold >= self.ks_statistic_threshold:
            raise ValueError(
                "ks_statistic_clear_threshold must be < ks_statistic_threshold (hysteresis gap)"
            )
        return self


class CategoricalDriftConfig(StrictModel):
    """Same effect-size-over-significance principle as ContinuousDriftConfig,
    applied to the categorical case: alerting is gated on Cramer's V (bounded
    [0, 1]), not the chi-square p-value, which has the identical collapse-to-
    zero problem at scale as KS's p-value. See driftwatch.stats.chi_square.
    cramers_v_clear_threshold provides the same hysteresis gap as the
    continuous case.
    """

    methods: list[Literal["chi_square"]]
    cramers_v_threshold: float = Field(gt=0, lt=1)
    cramers_v_clear_threshold: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _clear_below_fire(self) -> "CategoricalDriftConfig":
        if self.cramers_v_clear_threshold >= self.cramers_v_threshold:
            raise ValueError(
                "cramers_v_clear_threshold must be < cramers_v_threshold (hysteresis gap)"
            )
        return self


class PredictionScoreDriftConfig(StrictModel):
    method: Literal["jsd"]
    jsd_threshold: float = Field(gt=0)
    jsd_clear_threshold: float = Field(gt=0)

    @model_validator(mode="after")
    def _clear_below_fire(self) -> "PredictionScoreDriftConfig":
        if self.jsd_clear_threshold >= self.jsd_threshold:
            raise ValueError("jsd_clear_threshold must be < jsd_threshold (hysteresis gap)")
        return self


class DriftTestsConfig(StrictModel):
    continuous: ContinuousDriftConfig
    categorical: CategoricalDriftConfig
    prediction_score: PredictionScoreDriftConfig


class MultipleComparisonCorrectionConfig(StrictModel):
    method: Literal["benjamini_hochberg"]
    fdr_alpha: float = Field(gt=0, lt=1)


_MIN_HISTORY_SCAN_MARGIN = 2
"""history_scan_windows must exceed the largest configured persistence count
by at least this many windows. Not just >=: a scan bound pinned exactly at
the minimum required has zero slack for future persistence-count tweaks to
silently outrun it again, which is the exact failure this validation exists
to catch."""


class AlertingConfig(StrictModel):
    """An alert is a stateful entity (open -> escalated -> resolved), not an
    event per window -- see driftwatch.alerting.engine. These four counts are
    all persistence gates in the same sense, applied at different lifecycle
    transitions:

    - fire_persistence_windows: consecutive breaching windows before a new
      alert opens.
    - escalate_persistence_windows: consecutive breaching windows (continuing
      the same streak) before an already-open alert escalates. Must be >=
      fire_persistence_windows.
    - resolve_persistence_windows: consecutive CLEAR windows (see
      *_clear_threshold on the drift-test configs) before an open/escalated
      alert resolves -- symmetric with fire_persistence_windows by design,
      though not required to be numerically equal.
    - not_computable_persistence_windows: consecutive not_computable windows
      before a signal that's persistently uncomputable raises its own alert.
      Silence about a broken feature is exactly the failure mode this
      service exists to prevent, so this is never inferred from the drift
      counts above -- it's its own explicit gate.
    """

    fire_persistence_windows: int = Field(ge=1)
    escalate_persistence_windows: int = Field(ge=1)
    resolve_persistence_windows: int = Field(ge=1)
    not_computable_persistence_windows: int = Field(ge=1)
    history_scan_windows: int = Field(ge=1)
    """How many past windows driftwatch.alerting.engine rescans (per signal)
    to reconstruct its current streak from DriftResult/PerformanceResult
    history -- see driftwatch.alerting.streaks.current_streak. There is no
    persisted streak counter; every run recomputes the streak from this much
    trailing history, so if this bound were smaller than a configured
    persistence count, that gate could never be satisfied and alerts for
    this profile would never fire, escalate, or resolve -- silently, since
    nothing else in the system would surface the mismatch. Validated below
    to exceed the largest persistence count by _MIN_HISTORY_SCAN_MARGIN."""
    max_alerts_per_run: int = Field(ge=1)
    """If more new alerts would open in a single evaluation run than this,
    they're rolled up into one aggregate alert instead -- see
    driftwatch.alerting.engine. Caps notification-storm risk when many
    segments breach at once without hiding that it happened."""

    @model_validator(mode="after")
    def _escalate_not_below_fire(self) -> "AlertingConfig":
        if self.escalate_persistence_windows < self.fire_persistence_windows:
            raise ValueError("escalate_persistence_windows must be >= fire_persistence_windows")
        return self

    @model_validator(mode="after")
    def _history_scan_covers_persistence_with_margin(self) -> "AlertingConfig":
        largest = max(
            self.fire_persistence_windows,
            self.escalate_persistence_windows,
            self.resolve_persistence_windows,
            self.not_computable_persistence_windows,
        )
        required = largest + _MIN_HISTORY_SCAN_MARGIN
        if self.history_scan_windows < required:
            raise ValueError(
                f"history_scan_windows ({self.history_scan_windows}) must be >= the "
                f"largest configured persistence count ({largest}) + "
                f"_MIN_HISTORY_SCAN_MARGIN ({_MIN_HISTORY_SCAN_MARGIN}) = {required}; "
                "otherwise a streak can never reach the required persistence and "
                "alerts under this profile would never fire, escalate, or resolve"
            )
        return self


class MetricSpec(StrictModel):
    name: str
    params: dict[str, float | int | str] = Field(default_factory=dict)
    alert_direction: Literal["below", "above"] | None = None
    """"below": alert fires when the metric drops to/below fire_threshold
    (e.g. precision). "above": fires when it rises to/above (e.g. RMSE).
    None (the default) means this metric is informational only -- not every
    configured metric needs an alert lifecycle."""
    fire_threshold: float | None = None
    clear_threshold: float | None = None

    @model_validator(mode="after")
    def _alert_config_consistent(self) -> "MetricSpec":
        fields = (self.alert_direction, self.fire_threshold, self.clear_threshold)
        if any(f is not None for f in fields) and not all(f is not None for f in fields):
            raise ValueError(
                "alert_direction, fire_threshold, and clear_threshold must all be set "
                "together, or all omitted"
            )
        # by this point alert_direction set implies fire_threshold/clear_threshold are too
        if self.alert_direction == "below" and not self.clear_threshold > self.fire_threshold:  # type: ignore[operator]
            raise ValueError(
                "for alert_direction='below', clear_threshold must be > fire_threshold "
                "(hysteresis: must recover further before clearing)"
            )
        if self.alert_direction == "above" and not self.clear_threshold < self.fire_threshold:  # type: ignore[operator]
            raise ValueError(
                "for alert_direction='above', clear_threshold must be < fire_threshold"
            )
        return self


class Profile(StrictModel):
    name: str
    description: str
    evaluation: EvaluationConfig
    drift_tests: DriftTestsConfig
    multiple_comparison_correction: MultipleComparisonCorrectionConfig
    alerting: AlertingConfig
    performance_metrics: list[MetricSpec]


class FeatureSpec(StrictModel):
    name: str
    dtype: Literal["continuous", "categorical"]
    nullable: bool


class PredictionFieldSpec(StrictModel):
    dtype: Literal["continuous", "categorical"]


class LabelFieldSpec(StrictModel):
    dtype: Literal["continuous", "categorical"]


class ModelSchemaSpec(StrictModel):
    features: list[FeatureSpec]
    prediction: PredictionFieldSpec
    label: LabelFieldSpec


class SegmentsSpec(StrictModel):
    dimensions: list[str] = Field(default_factory=list)
    min_segment_size: int = Field(gt=0, default=1)


class ModelConfig(StrictModel):
    model_config = {"extra": "forbid", "populate_by_name": True, "protected_namespaces": ()}

    model_id: str
    profile: str
    prediction_type: Literal["regression", "binary_classification", "multiclass_classification"]
    schema_: ModelSchemaSpec = Field(alias="schema")
    segments: SegmentsSpec = Field(default_factory=SegmentsSpec)
