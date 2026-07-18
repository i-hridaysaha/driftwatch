from typing import Literal

from pydantic import BaseModel, Field, model_validator


class StrictModel(BaseModel):
    model_config = {"extra": "forbid"}


class EvaluationConfig(StrictModel):
    window: str
    min_window_size: int = Field(gt=0)


class ContinuousDriftConfig(StrictModel):
    methods: list[Literal["psi", "ks"]]
    psi_threshold: float = Field(gt=0)
    ks_alpha: float = Field(gt=0, lt=1)


class CategoricalDriftConfig(StrictModel):
    methods: list[Literal["chi_square"]]
    chi_square_alpha: float = Field(gt=0, lt=1)


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
