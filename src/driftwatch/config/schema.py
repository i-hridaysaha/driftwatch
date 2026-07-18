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


class Profile(StrictModel):
    name: str
    description: str
    evaluation: EvaluationConfig
    drift_tests: DriftTestsConfig
    multiple_comparison_correction: MultipleComparisonCorrectionConfig
    alerting: AlertingConfig
