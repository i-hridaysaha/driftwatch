"""The I/O side of demo data loading: everything generator.py and
modelconfig.py can't do because they're pure. build_scenario() is the one
function backing `driftwatch demo <scenario>` -- reset the database, write
the derived model config, register the baseline, ingest predictions,
ingest labels on their staggered two-phase schedule, and evaluate the full
window range, in that order.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import text, update
from sqlalchemy.orm import Session

from driftwatch.alerting.notifications import default_channels
from driftwatch.cli import evaluate_range
from driftwatch.config.loader import load_model_config, load_profile
from driftwatch.db.models import (
    Baseline,
    BaselineRecord,
    EvaluationWindow,
    Label,
    Model,
    Prediction,
)
from driftwatch.db.session import SessionLocal
from driftwatch.demo.generator import (
    LabelData,
    PredictionData,
    ScenarioData,
    generate_scenario_data,
)
from driftwatch.demo.loader import load_scenario
from driftwatch.demo.modelconfig import write_model_config_yaml
from driftwatch.demo.schema import ScenarioConfig
from driftwatch.durations import parse_duration
from driftwatch.hashing import compute_payload_hash
from driftwatch.scheduler.jobs import recompute_stale_windows
from driftwatch.scheduler.windowing import EPOCH
from driftwatch.stats.binning import compute_baseline_binning

_APP_TABLES = (
    "alerts",
    "performance_results",
    "drift_results",
    "evaluation_windows",
    "labels",
    "predictions",
    "baseline_records",
    "baselines",
    "models",
)


@dataclass(frozen=True)
class BuildSummary:
    scenario_name: str
    model_id: str
    baseline_records: int
    predictions: int
    labels_early: int
    labels_late: int
    windows_evaluated: int
    windows_skipped: int


def reset_database(session: Session) -> None:
    """Wipes every row this service owns -- schema/migrations are left
    alone (that's `alembic upgrade head`'s job, run separately); this is
    "empty database" in the sense of "no data", the starting point
    constraint 8 asks for. CASCADE + one statement across all tables lets
    Postgres resolve FK order itself rather than this needing to know the
    dependency graph by hand."""
    session.execute(text(f"TRUNCATE TABLE {', '.join(_APP_TABLES)} RESTART IDENTITY CASCADE"))
    session.commit()


def _register_baseline(session: Session, scenario: ScenarioConfig, data: ScenarioData) -> Baseline:
    session.add(Model(model_id=scenario.model_id))
    session.flush()

    model_config = load_model_config(scenario.model_id)
    features = [r.features for r in data.baseline_records]
    scores = [r.prediction_score for r in data.baseline_records]
    binning = compute_baseline_binning(
        features, scores, model_config.schema_, segment_dimensions=model_config.segments.dimensions
    )

    baseline = Baseline(model_id=scenario.model_id, is_active=True, binning_config=binning)
    session.add(baseline)
    session.flush()
    for record in data.baseline_records:
        session.add(
            BaselineRecord(
                baseline_id=baseline.id,
                features=record.features,
                prediction_score=record.prediction_score,
                segment_values=record.segment_values,
            )
        )
    session.commit()
    return baseline


def _prediction_payload_hash(record: PredictionData) -> str:
    return compute_payload_hash(
        {
            "predicted_at": record.predicted_at.isoformat(),
            "features": record.features,
            "prediction_value": {"score": record.prediction_score},
            "prediction_score": record.prediction_score,
        }
    )


def _insert_predictions(
    session: Session, scenario: ScenarioConfig, predictions: list[PredictionData]
) -> None:
    for record in predictions:
        session.add(
            Prediction(
                prediction_id=record.prediction_id,
                model_id=scenario.model_id,
                predicted_at=record.predicted_at,
                features=record.features,
                prediction_value={"score": record.prediction_score},
                prediction_score=record.prediction_score,
                segment_values=record.segment_values,
                payload_hash=_prediction_payload_hash(record),
            )
        )
    session.commit()


def _label_payload_hash(record: LabelData) -> str:
    return compute_payload_hash(
        {"label_value": record.label_value, "labeled_at": record.labeled_at.isoformat()}
    )


def _insert_labels(session: Session, scenario: ScenarioConfig, labels: Iterable[LabelData]) -> None:
    for record in labels:
        session.add(
            Label(
                prediction_id=record.prediction_id,
                model_id=scenario.model_id,
                label_value=record.label_value,
                labeled_at=record.labeled_at,
                payload_hash=_label_payload_hash(record),
            )
        )
    session.commit()


def _window_start_for(window_duration: timedelta, predicted_at: datetime) -> datetime:
    """The window_start `predicted_at` falls into, computed the exact same
    way driftwatch.scheduler.windowing.compute_window_boundaries does:
    epoch-aligned, not aligned to this scenario's own start_date. A
    scenario's start_date is usually also epoch-aligned (e.g. midnight
    UTC for hourly/daily windows), but this must never assume that -- it
    reuses the real EPOCH constant so the two can never silently
    disagree."""
    index = (predicted_at - EPOCH) // window_duration
    return EPOCH + index * window_duration


def _recompute_touched_windows(
    session: Session,
    scenario: ScenarioConfig,
    labels: list[LabelData],
    predicted_at_by_id: dict[str, datetime],
) -> None:
    """Mirrors exactly what driftwatch.api.routes.labels.ingest_labels does
    for late-arriving labels -- flag every EvaluationWindow touched by this
    batch as performance_stale -- and then what the scheduler's next tick
    does: claim and recompute each flagged window once, via the same
    recompute_stale_windows the scheduler runs. No HTTP layer, since this
    is inserting thousands of labels at once, not one ingestion request at
    a time. Window lookup is index arithmetic against the same epoch-aligned
    boundaries the real windows use, not a query per label."""
    window_duration = parse_duration(scenario.window)
    windows_by_start = {
        w.window_start: w.id
        for w in session.query(EvaluationWindow).filter(
            EvaluationWindow.model_id == scenario.model_id
        )
    }
    touched_window_ids: set[int] = set()
    for record in labels:
        predicted_at = predicted_at_by_id[record.prediction_id]
        window_start = _window_start_for(window_duration, predicted_at)
        window_id = windows_by_start.get(window_start)
        if window_id is not None:
            touched_window_ids.add(window_id)

    if touched_window_ids:
        session.execute(
            update(EvaluationWindow)
            .where(EvaluationWindow.id.in_(touched_window_ids))
            .values(performance_stale=True)
        )
    session.commit()
    recompute_stale_windows(session, default_channels())


def _check_window_matches_profile(scenario: ScenarioConfig) -> None:
    """evaluate_range (via driftwatch.cli) groups predictions into windows
    using the CHOSEN PROFILE's evaluation.window, not this scenario's own
    `window` field -- the generator groups predictions into windows using
    scenario.window. If the two ever disagree, the generator's "window
    index N" (which every event's start_window and every verification
    assertion is expressed in) stops corresponding to what evaluate_range
    actually evaluates -- e.g. an aggressive-profile scenario declaring
    window: 1d against a profile whose real window is 1h would silently
    evaluate 24x as many real windows as the scenario thinks it generated.
    This must fail loudly at build time, not produce a quietly wrong
    demo dataset."""
    profile = load_profile(scenario.profile)
    if profile.evaluation.window != scenario.window:
        raise ValueError(
            f"scenario {scenario.name!r} declares window={scenario.window!r}, but its "
            f"profile {scenario.profile!r} evaluates on window={profile.evaluation.window!r} -- "
            "these must match exactly, or the generator's window grouping won't correspond "
            "to what evaluate_range actually evaluates"
        )


def build_scenario(
    scenario_name: str, *, reset: bool = True, seed: int | None = None
) -> BuildSummary:
    """Generate, load and evaluate one scenario. `seed` overrides the
    scenario's own seed so CI can verify every scenario's claims at more
    than the one seed it ships with; nothing else about the scenario
    changes, so the verifier's checks apply unchanged."""
    scenario = load_scenario(scenario_name)
    if seed is not None:
        scenario = scenario.model_copy(update={"seed": seed})
    _check_window_matches_profile(scenario)
    data = generate_scenario_data(scenario)

    session = SessionLocal()
    try:
        if reset:
            reset_database(session)

        write_model_config_yaml(scenario)

        _register_baseline(session, scenario, data)
        _insert_predictions(session, scenario, data.predictions)

        early_labels = [
            label
            for label in data.labels
            if label.delay_hours <= scenario.label_delay.early_cutoff_hours
        ]
        late_labels = [
            label
            for label in data.labels
            if label.delay_hours > scenario.label_delay.early_cutoff_hours
        ]
        _insert_labels(session, scenario, early_labels)

        window_duration = parse_duration(scenario.window)
        range_start = scenario.start_date
        range_end = scenario.start_date + scenario.total_windows * window_duration
        evaluated, skipped = evaluate_range(session, scenario.model_id, range_start, range_end)

        predicted_at_by_id = {p.prediction_id: p.predicted_at for p in data.predictions}
        _insert_labels(session, scenario, late_labels)
        _recompute_touched_windows(session, scenario, late_labels, predicted_at_by_id)

        return BuildSummary(
            scenario_name=scenario_name,
            model_id=scenario.model_id,
            baseline_records=len(data.baseline_records),
            predictions=len(data.predictions),
            labels_early=len(early_labels),
            labels_late=len(late_labels),
            windows_evaluated=evaluated,
            windows_skipped=skipped,
        )
    finally:
        session.close()
