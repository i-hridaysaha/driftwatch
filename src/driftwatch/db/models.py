import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from driftwatch.db.base import Base


class TestMethod(enum.StrEnum):
    PSI = "psi"
    KS = "ks"
    CHI_SQUARE = "chi_square"
    JSD = "jsd"


class Model(Base):
    __tablename__ = "models"

    model_id: Mapped[str] = mapped_column(String, primary_key=True)
    first_registered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")


class Baseline(Base):
    __tablename__ = "baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.model_id"), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    binning_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index(
            "ix_one_active_baseline_per_model",
            "model_id",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )


class BaselineRecord(Base):
    __tablename__ = "baseline_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    baseline_id: Mapped[int] = mapped_column(ForeignKey("baselines.id"), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prediction_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    segment_values: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )

    __table_args__ = (Index("ix_baseline_records_baseline_id", "baseline_id"),)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String, nullable=False)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.model_id"), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prediction_value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    prediction_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    segment_values: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default="{}"
    )
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("model_id", "prediction_id", name="uq_predictions_model_prediction"),
        Index("ix_predictions_model_predicted_at", "model_id", "predicted_at"),
        Index("ix_predictions_segment_values", "segment_values", postgresql_using="gin"),
    )


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[str] = mapped_column(String, nullable=False)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.model_id"), nullable=False)
    label_value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    labeled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["model_id", "prediction_id"],
            ["predictions.model_id", "predictions.prediction_id"],
        ),
        Index("ix_labels_model_prediction", "model_id", "prediction_id"),
    )


class EvaluationWindow(Base):
    __tablename__ = "evaluation_windows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.model_id"), nullable=False)
    baseline_id: Mapped[int] = mapped_column(ForeignKey("baselines.id"), nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    config_hash: Mapped[str] = mapped_column(String, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    performance_computed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    n_predictions: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_labels: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    __table_args__ = (
        UniqueConstraint(
            "model_id", "window_start", "window_end", name="uq_evaluation_windows_model_range"
        ),
    )


class DriftResult(Base):
    __tablename__ = "drift_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_window_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_windows.id"), nullable=False
    )
    feature_name: Mapped[str] = mapped_column(String, nullable=False)
    segment_dimension: Mapped[str | None] = mapped_column(String, nullable=True)
    segment_value: Mapped[str | None] = mapped_column(String, nullable=True)
    test_method: Mapped[TestMethod] = mapped_column(
        Enum(TestMethod, name="test_method"), nullable=False
    )
    statistic: Mapped[float] = mapped_column(Float, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    corrected_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_significant: Mapped[bool] = mapped_column(Boolean, nullable=False)
    n_baseline: Mapped[int] = mapped_column(Integer, nullable=False)
    n_live: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_drift_results_window", "evaluation_window_id"),
        Index("ix_drift_results_window_feature", "evaluation_window_id", "feature_name"),
    )


class PerformanceResult(Base):
    __tablename__ = "performance_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_window_id: Mapped[int] = mapped_column(
        ForeignKey("evaluation_windows.id"), nullable=False
    )
    segment_dimension: Mapped[str | None] = mapped_column(String, nullable=True)
    segment_value: Mapped[str | None] = mapped_column(String, nullable=True)
    metric_name: Mapped[str] = mapped_column(String, nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    n_labeled: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_retroactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    __table_args__ = (Index("ix_performance_results_window", "evaluation_window_id"),)
