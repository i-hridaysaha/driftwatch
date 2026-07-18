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
    """

    methods: list[Literal["psi", "ks"]]
    psi_threshold: float = Field(gt=0)
    ks_statistic_threshold: float = Field(gt=0, lt=1)


class CategoricalDriftConfig(StrictModel):
    """Same effect-size-over-significance principle as ContinuousDriftConfig,
    applied to the categorical case: alerting is gated on Cramer's V (bounded
    [0, 1]), not the chi-square p-value, which has the identical collapse-to-
    zero problem at scale as KS's p-value. See driftwatch.stats.chi_square.
    """

    methods: list[Literal["chi_square"]]
    cramers_v_threshold: float = Field(gt=0, lt=1)


class PredictionScoreDriftConfig(StrictModel):
    method: Literal["jsd"]
    jsd_threshold: float = Field(gt=0)


class DriftTestsConfig(StrictModel):
    continuous: ContinuousDriftConfig
    categorical: CategoricalDriftConfig
    prediction_score: PredictionScoreDriftConfig


class MultipleComparisonCorrectionConfig(StrictModel):
    method: Literal["benjamini_hochberg"]
    fdr_alpha: float = Field(gt=0, lt=1)


class SeverityThresholds(StrictModel):
    warning: int = Field(ge=1)
    critical: int = Field(ge=1)

    @model_validator(mode="after")
    def critical_not_below_warning(self) -> "SeverityThresholds":
        if self.critical < self.warning:
            raise ValueError("critical threshold must be >= warning threshold")
        return self


class AlertingConfig(StrictModel):
    persistence_windows: int = Field(ge=1)
    severity_thresholds: SeverityThresholds


class MetricSpec(StrictModel):
    name: str
    params: dict[str, float | int | str] = Field(default_factory=dict)


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
