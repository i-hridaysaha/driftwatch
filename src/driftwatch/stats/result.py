import enum


class StatStatus(enum.StrEnum):
    """Mirrors driftwatch.db.models.MetricStatus's status/reason shape by
    convention, not by import -- driftwatch.stats is pure and has no
    dependency on the DB layer. Phase 5's evaluation pipeline is responsible
    for mapping this onto DriftResult's own status column, the same way it
    already maps MetricStatus onto PerformanceResult, so the system tells one
    consistent story about values that can't be computed rather than two."""

    COMPUTED = "computed"
    NOT_COMPUTABLE = "not_computable"
