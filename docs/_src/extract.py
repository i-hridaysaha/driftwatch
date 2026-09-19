"""Pull the per-window traces and alert rows the demo page charts, as JSON.

Everything here is read from the database that five `driftwatch demo` runs
wrote (`clean`, then the other four with `--no-reset`); nothing is typed in.
Run from the repository root, with the same DATABASE_URL the runs used:

    uv run python docs/_src/extract.py > docs/_src/data.json
"""

import json
import sys
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.db.models import (
    Alert,
    DriftResult,
    EvaluationWindow,
    Label,
    PerformanceResult,
    Prediction,
)
from driftwatch.db.session import SessionLocal

SCENARIOS = {
    "clean": "demo-clean",
    "covariate_shift": "demo-covariate-shift",
    "segment_isolated": "demo-segment-isolated",
    "concept_drift": "demo-concept-drift",
    "burst_shift": "demo-burst-shift",
}


def iso(dt: datetime | None) -> str | None:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def windows(session: Session, model_id: str) -> list[EvaluationWindow]:
    return list(
        session.scalars(
            select(EvaluationWindow)
            .where(EvaluationWindow.model_id == model_id)
            .order_by(EvaluationWindow.window_start)
        )
    )


def drift_trace(
    session: Session,
    wins: list[EvaluationWindow],
    feature: str,
    method: str,
    seg_dim: str | None = None,
    seg_val: str | None = None,
) -> list[dict | None]:
    ids = {w.id: i for i, w in enumerate(wins)}
    q = select(DriftResult).where(
        DriftResult.evaluation_window_id.in_(list(ids)),
        DriftResult.feature_name == feature,
        DriftResult.test_method == method,
    )
    q = (
        q.where(DriftResult.segment_dimension.is_(None))
        if seg_dim is None
        else q.where(
            DriftResult.segment_dimension == seg_dim, DriftResult.segment_value == seg_val
        )
    )
    out: list[dict | None] = [None] * len(wins)
    for r in session.scalars(q):
        out[ids[r.evaluation_window_id]] = {
            "i": ids[r.evaluation_window_id],
            "stat": None if r.statistic is None else round(r.statistic, 4),
            "status": str(r.status),
            "n_live": r.n_live,
            "n_baseline": r.n_baseline,
            "reason": r.not_computable_reason,
        }
    return out


def perf_trace(
    session: Session, wins: list[EvaluationWindow], retro: bool
) -> list[dict | None]:
    ids = {w.id: i for i, w in enumerate(wins)}
    q = select(PerformanceResult).where(
        PerformanceResult.evaluation_window_id.in_(list(ids)),
        PerformanceResult.metric_name == "pr_auc",
        PerformanceResult.segment_dimension.is_(None),
        PerformanceResult.is_retroactive.is_(retro),
    )
    out: list[dict | None] = [None] * len(wins)
    for r in session.scalars(q):
        out[ids[r.evaluation_window_id]] = {
            "i": ids[r.evaluation_window_id],
            "value": None if r.metric_value is None else round(r.metric_value, 4),
            "status": str(r.status),
            "n_labeled": r.n_labeled,
            "reason": r.not_computable_reason,
        }
    return out


def alerts(session: Session, model_id: str, wins: list[EvaluationWindow]) -> list[dict]:
    """One row per Alert, with its evidence windows as indexes into `wins`.

    A resolved alert is never touched again, so its last_seen window is the
    one that resolved it (driftwatch.alerting.engine).
    """
    ids = {w.id: i for i, w in enumerate(wins)}
    rows = []
    for a in session.scalars(select(Alert).where(Alert.model_id == model_id).order_by(Alert.id)):
        segment = (
            None if a.segment_dimension is None else f"{a.segment_dimension}={a.segment_value}"
        )
        rows.append(
            {
                "id": a.id,
                "kind": str(a.kind),
                "signal": a.signal_name,
                "feature": a.feature_name,
                "segment": segment,
                "status": str(a.status),
                "opened_window": ids.get(a.evidence_window_id),
                "escalated_window": ids.get(a.escalation_evidence_window_id),
                "resolved_window": (
                    ids.get(a.last_seen_window_id) if str(a.status) == "resolved" else None
                ),
                "last_seen_window": ids.get(a.last_seen_window_id),
                "reason": a.evidence_reason,
                "breach_streak": a.consecutive_breaching_windows,
                "clear_streak": a.consecutive_clear_windows,
                "opened_at": iso(a.first_opened_at),
                "escalated_at": iso(a.escalated_at),
                "resolved_at": iso(a.resolved_at),
                "evidence_stat": (
                    None if a.evidence_statistic is None else round(a.evidence_statistic, 4)
                ),
                "evidence_threshold": a.evidence_threshold,
                "escalation_stat": (
                    None
                    if a.escalation_evidence_statistic is None
                    else round(a.escalation_evidence_statistic, 4)
                ),
                "notified": None if a.last_notified_status is None else str(a.last_notified_status),
                "attempts": a.notification_attempts,
            }
        )
    return rows


def main() -> None:
    out: dict = {"scenarios": {}, "totals": {}}
    with SessionLocal() as session:
        for name, model_id in SCENARIOS.items():
            wins = windows(session, model_id)
            entry: dict = {
                "model_id": model_id,
                "n_windows": len(wins),
                "window_start": iso(wins[0].window_start),
                "window_end": iso(wins[-1].window_end),
                "n_predictions": sum(w.n_predictions for w in wins),
                "n_labels": sum(w.n_labels for w in wins),
                "age_psi_global": drift_trace(session, wins, "age", "psi"),
                "age_ks_global": drift_trace(session, wins, "age", "ks"),
                "alerts": alerts(session, model_id, wins),
            }
            if name == "segment_isolated":
                for seg in ("APAC", "EU", "US"):
                    entry[f"age_psi_{seg}"] = drift_trace(
                        session, wins, "age", "psi", "region", seg
                    )
                    entry[f"age_ks_{seg}"] = drift_trace(session, wins, "age", "ks", "region", seg)
            if name == "concept_drift":
                entry["pr_auc_initial"] = perf_trace(session, wins, retro=False)
                entry["pr_auc_final"] = perf_trace(session, wins, retro=True)
                entry["jsd_global"] = drift_trace(session, wins, "__prediction_score__", "jsd")
            out["scenarios"][name] = entry
        out["totals"] = {
            "predictions": session.query(Prediction).count(),
            "labels": session.query(Label).count(),
            "windows": session.query(EvaluationWindow).count(),
            "drift_results": session.query(DriftResult).count(),
            "performance_results": session.query(PerformanceResult).count(),
            "alerts": session.query(Alert).count(),
        }
    json.dump(out, sys.stdout, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    main()
