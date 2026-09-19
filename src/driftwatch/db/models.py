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


class MetricStatus(enum.StrEnum):
    COMPUTED = "computed"
    NOT_COMPUTABLE = "not_computable"


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
    payload_hash: Mapped[str] = mapped_column(String, nullable=False)

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
    label_value: Mapped[Any] = mapped_column(JSONB, nullable=False)
    labeled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String, nullable=False)
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
    label_watermark: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    n_predictions: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    n_labels: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    performance_stale: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    """Set by label ingestion, in the same transaction as the labels, when a
    batch touches this window; cleared by the scheduler when it claims the
    window for a performance recompute (driftwatch.scheduler.jobs
    .recompute_stale_windows). A label landing during the recompute sets it
    again, so the window is simply recomputed once more on the next tick.
    This is what keeps a month-long label backfill out of the request path:
    the request flags, the scheduler does the work."""
    late_prediction_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    """Predictions ingested after this window's drift was already evaluated,
    whose predicted_at falls inside [window_start, window_end) anyway.
    Permanently excluded from this window's drift stats (see
    driftwatch.scheduler.windowing.is_drift_watermark_elapsed) -- tracked
    here, and logged at ingestion time, so a rising rate is visible as a
    data pipeline problem rather than silently absorbed."""

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
    status: Mapped[MetricStatus] = mapped_column(
        Enum(MetricStatus, name="metric_status"), nullable=False
    )
    statistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    corrected_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_significant: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    not_computable_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    n_baseline: Mapped[int] = mapped_column(Integer, nullable=False)
    n_live: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        Index("ix_drift_results_window", "evaluation_window_id"),
        Index("ix_drift_results_window_feature", "evaluation_window_id", "feature_name"),
        Index(
            "uq_drift_results_window_feature_segment_method",
            "evaluation_window_id",
            "feature_name",
            "test_method",
            text("COALESCE(segment_dimension, '')"),
            text("COALESCE(segment_value, '')"),
            unique=True,
        ),
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
    status: Mapped[MetricStatus] = mapped_column(
        Enum(MetricStatus, name="metric_status"), nullable=False
    )
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    not_computable_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    n_labeled: Mapped[int] = mapped_column(Integer, nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_retroactive: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")

    __table_args__ = (Index("ix_performance_results_window", "evaluation_window_id"),)


class AlertKind(enum.StrEnum):
    DRIFT = "drift"
    PERFORMANCE = "performance"
    NOT_COMPUTABLE = "not_computable"


class AlertStatus(enum.StrEnum):
    OPEN = "open"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class Alert(Base):
    """A stateful entity with a lifecycle (open -> escalated -> resolved),
    not an event per window -- driftwatch.alerting.engine recomputes the
    relevant streak from recent DriftResult/PerformanceResult history on
    every evaluation and updates ONE row per signal identity, rather than
    inserting a new row every time a window confirms the signal is still
    active. See driftwatch.alerting.streaks for how the streak itself is
    computed.

    Identity (what makes two rows "the same recurring signal"): model_id +
    kind + signal_name + feature_name + segment_dimension + segment_value.
    Enforced by uq_alerts_active_identity below, scoped to non-resolved,
    non-aggregate rows only -- a RESOLVED alert doesn't block a fresh row
    when the same signal fires again later; that's a new episode, kept as
    its own row for history.
    """

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.model_id"), nullable=False)
    kind: Mapped[AlertKind] = mapped_column(Enum(AlertKind, name="alert_kind"), nullable=False)
    signal_name: Mapped[str] = mapped_column(String, nullable=False)
    """test_method value (e.g. "psi") for kind=drift/not_computable on a
    drift test; metric_name (e.g. "pr_auc") for kind=performance or
    kind=not_computable on a performance metric."""
    feature_name: Mapped[str | None] = mapped_column(String, nullable=True)
    """Feature name for drift/not_computable-on-a-drift-test alerts (or the
    PREDICTION_SCORE_FEATURE_NAME sentinel). Null for performance alerts,
    which aren't feature-scoped."""
    segment_dimension: Mapped[str | None] = mapped_column(String, nullable=True)
    segment_value: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus, name="alert_status"), nullable=False
    )
    consecutive_breaching_windows: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    consecutive_clear_windows: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )

    # Evidence snapshot, frozen at the moment this row transitions to OPEN.
    # Never updated afterward -- constraint: if config changes later, the
    # alert must still explain why it fired under the rules in effect then.
    evidence_statistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_baseline_id: Mapped[int | None] = mapped_column(
        ForeignKey("baselines.id"), nullable=True
    )
    evidence_config_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    evidence_window_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_windows.id"), nullable=True
    )
    evidence_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    """not_computable_reason from the triggering row, for kind=not_computable."""

    # A second evidence snapshot, frozen once at the moment this row
    # transitions OPEN -> ESCALATED (never touched again after that, same
    # freeze-once discipline as the open-time evidence above). An alert that
    # opened at 0.11 and escalated at 0.42 should still be able to say so --
    # without this, escalated_at is just a timestamp with no statistic behind
    # it. Null until (unless) the alert actually escalates.
    escalation_evidence_statistic: Mapped[float | None] = mapped_column(Float, nullable=True)
    escalation_evidence_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    escalation_evidence_baseline_id: Mapped[int | None] = mapped_column(
        ForeignKey("baselines.id"), nullable=True
    )
    escalation_evidence_config_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    escalation_evidence_window_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_windows.id"), nullable=True
    )

    # Currency tracking -- mutable, updated every run that confirms this signal
    # is still active. This is the "updated last-seen" from a sustained streak,
    # as opposed to the frozen evidence above.
    first_opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_window_id: Mapped[int | None] = mapped_column(
        ForeignKey("evaluation_windows.id"), nullable=True
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    escalated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Notification idempotency: only re-notify when status changes from what
    # was last notified, never on steady-state continuation of an open alert.
    last_notified_status: Mapped[AlertStatus | None] = mapped_column(
        Enum(AlertStatus, name="alert_status"), nullable=True
    )
    last_notified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Delivery-failure currency, not evidence: overwritten on every failed
    # attempt, cleared implicitly by nothing (a persistently-failing channel
    # keeps this populated). A raising/hanging notification channel must
    # never fail the evaluation run or roll back this row's transition --
    # see driftwatch.alerting.notifications.notify_if_needed, which catches
    # per-channel and records the failure here instead of propagating.
    last_notification_error: Mapped[str | None] = mapped_column(String, nullable=True)
    last_notification_error_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notification_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    """Failed delivery rounds for the CURRENT status. A failed round leaves
    last_notified_status behind the real status, so the scheduler retries
    the transition on its next tick (driftwatch.scheduler.jobs
    .retry_pending_notifications), up to MAX_NOTIFICATION_ATTEMPTS; a
    successful round resets this to 0. An alert that opens while the
    webhook is down therefore still reaches someone once it is back, rather
    than being logged once and forgotten."""

    # Aggregation: when more new alerts would open in one run than
    # AlertingConfig.max_alerts_per_run, they're rolled into one row like this
    # instead, recording how many and which.
    is_aggregate: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    aggregated_signal_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    aggregated_signals: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_alerts_model", "model_id"),
        Index(
            "uq_alerts_active_identity",
            "model_id",
            "kind",
            "signal_name",
            text("COALESCE(feature_name, '')"),
            text("COALESCE(segment_dimension, '')"),
            text("COALESCE(segment_value, '')"),
            unique=True,
            # Postgres stores the enum member NAME (uppercase), not the Python
            # value -- see TestMethod's precedent in the initial migration.
            postgresql_where=text("status != 'RESOLVED' AND NOT is_aggregate"),
        ),
    )
