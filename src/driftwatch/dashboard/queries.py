"""Every function here reads from drift_results / performance_results /
alerts / evaluation_windows -- tables that are already one-row-per-window
(or one-row-per-signal-identity) aggregates produced by the evaluation
pipeline, not raw predictions or labels. Nothing in this module ever
queries the predictions or labels tables: ranking, filtering, and
window-function reductions (e.g. "first vs latest performance revision")
happen in SQL over those small aggregate tables, so the dashboard stays
responsive regardless of how many raw prediction rows fed into them.

Where a function reconciles SQL results against an "expected" grid (e.g.
every configured feature x test method, or every known segment value) to
surface not_configured gaps, that expected grid comes from static model
config (a file read) or a small SQL DISTINCT -- never from raw prediction
volume -- and the reconciliation itself is plain pandas over a few dozen
rows at most.
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, aliased

from driftwatch.alerting.reporting import find_suppressed_episodes
from driftwatch.alerting.streaks import StreakKind, classify_drift, classify_performance
from driftwatch.config.schema import ModelConfig, Profile
from driftwatch.db.models import (
    Alert,
    DriftResult,
    EvaluationWindow,
    MetricStatus,
    Model,
    PerformanceResult,
    TestMethod,
)
from driftwatch.evaluation.drift import PREDICTION_SCORE_FEATURE_NAME
from driftwatch.scheduler.windowing import compute_window_boundaries


def date_range_to_datetimes(start: date, end: date) -> tuple[datetime, datetime]:
    """Half-open bound for comparison against window_start/window_end.
    `end` is treated as inclusive of that whole calendar day, matching what
    a user picking "end date" in the UI expects."""
    start_dt = datetime.combine(start, time.min, tzinfo=UTC)
    end_dt = datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC)
    return start_dt, end_dt


def _expected_window_grid(
    start_dt: datetime, end_dt: datetime, window_duration: timedelta
) -> pd.DataFrame:
    """The windows that SHOULD exist over [start_dt, end_dt) for a profile
    configured with `window_duration` -- the same deterministic, epoch-
    aligned boundaries driftwatch.scheduler.windowing.compute_window_boundaries
    produces for the real scheduler/CLI backfill, reused here as the single
    source of truth for "what windows were ever supposed to be evaluated."

    Reconciling actual EvaluationWindow rows against this grid (rather than
    just listing whatever EvaluationWindow rows happen to exist) is what
    lets a window that was NEVER evaluated at all (scheduler downtime, a
    backfill gap) be told apart from a window that was evaluated and found
    quiet. Without this, the two are visually identical: a line drawn only
    from the rows that exist connects straight across a genuinely missing
    window with no break and no marker -- indistinguishable from
    uninterrupted quiet monitoring, which is the most dangerous confusion
    a monitoring dashboard can produce."""
    boundaries = compute_window_boundaries(start_dt, end_dt, window_duration)
    return pd.DataFrame(boundaries, columns=["window_start", "window_end"])


def _label_state(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `state` column, one of:
    - 'computed' / 'not_computable': a DriftResult/PerformanceResult row
      exists, with that status.
    - 'not_configured': the EvaluationWindow exists, but no result row for
      this exact signal does (not in the profile's configured methods, or
      -- for a segment -- zero live predictions fell into it that window).
    - 'missing': no EvaluationWindow exists at all for this expected time
      slot -- only detected when the frame carries a `window_id` column
      from a merge against _expected_window_grid; call sites with no such
      grid (fetch_feature_attribution's single already-selected window)
      never have a missing state to report, since there's nothing to
      reconcile against.
    Every combination this function is given a chance to see gets an
    explicit state, never a silently dropped row."""
    df = df.copy()
    if df.empty:
        df["state"] = pd.Series(dtype="object")
        return df
    has_window_grid = "window_id" in df.columns

    def label(row: pd.Series) -> str:
        if has_window_grid and pd.isna(row["window_id"]):
            return "missing"
        status = row.get("status")
        if pd.isna(status):
            return "not_configured"
        return "computed" if status == MetricStatus.COMPUTED else "not_computable"

    df["state"] = df.apply(label, axis=1)
    if "reason" in df.columns:
        df.loc[df["state"] == "missing", "reason"] = (
            "not evaluated -- no window exists for this time slot"
        )
    return df


def drift_thresholds(profile: Profile, test_method: TestMethod) -> tuple[float, float]:
    """(fire_threshold, clear_threshold) for a drift test method. Mirrors
    driftwatch.alerting.engine._drift_thresholds -- kept as a small
    separate copy rather than importing that private helper across module
    boundaries; the lookup itself is six lines of config-shape mapping,
    not logic worth sharing a symbol for."""
    continuous = profile.drift_tests.continuous
    categorical = profile.drift_tests.categorical
    prediction_score = profile.drift_tests.prediction_score
    if test_method == TestMethod.PSI:
        return continuous.psi_threshold, continuous.psi_clear_threshold
    if test_method == TestMethod.KS:
        return continuous.ks_statistic_threshold, continuous.ks_statistic_clear_threshold
    if test_method == TestMethod.CHI_SQUARE:
        return categorical.cramers_v_threshold, categorical.cramers_v_clear_threshold
    return prediction_score.jsd_threshold, prediction_score.jsd_clear_threshold


def list_models(session: Session) -> list[str]:
    """Only models with at least one evaluated window -- a registered
    model with no evaluation history yet has nothing any view here could
    show."""
    stmt = (
        select(Model.model_id)
        .join(EvaluationWindow, EvaluationWindow.model_id == Model.model_id)
        .distinct()
        .order_by(Model.model_id)
    )
    return list(session.scalars(stmt))


def default_time_range(session: Session, model_id: str) -> tuple[date, date] | None:
    """Derived from the data present in evaluation_windows -- never
    datetime.now() (constraint: default range must be reproducible from
    the database's own contents, not from when the dashboard happens to be
    opened)."""
    row = session.execute(
        select(
            func.min(EvaluationWindow.window_start), func.max(EvaluationWindow.window_end)
        ).where(EvaluationWindow.model_id == model_id)
    ).one()
    if row[0] is None:
        return None
    return row[0].date(), row[1].date()


def list_windows(
    session: Session, model_id: str, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    stmt = (
        select(
            EvaluationWindow.id.label("window_id"),
            EvaluationWindow.window_start,
            EvaluationWindow.window_end,
        )
        .where(
            EvaluationWindow.model_id == model_id,
            EvaluationWindow.window_end > start_dt,
            EvaluationWindow.window_start < end_dt,
        )
        .order_by(EvaluationWindow.window_end)
    )
    return pd.DataFrame(session.execute(stmt).mappings().all())


def fetch_drift_timeline(
    session: Session,
    model_id: str,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str | None,
    segment_value: str | None,
    start_dt: datetime,
    end_dt: datetime,
    window_duration: timedelta,
) -> pd.DataFrame:
    """The EXPECTED window grid over [start_dt, end_dt) (see
    _expected_window_grid), LEFT JOINed against whatever EvaluationWindow
    rows actually exist, in turn LEFT JOINed against this exact (feature,
    test_method, segment) signal's DriftResult row. Three ways a window can
    come up empty, each rendered as its own state: 'missing' (no
    EvaluationWindow at all -- the window was never evaluated), vs.
    'not_configured' (the window was evaluated, but not this signal -- not
    in the profile's configured methods for this feature's dtype, or a
    segment with zero live predictions that window), vs. 'not_computable'
    (an attempt was recorded, with a reason). A window that was simply
    never evaluated must never look identical to one that was evaluated
    and found quiet."""
    join_condition = and_(
        DriftResult.evaluation_window_id == EvaluationWindow.id,
        DriftResult.feature_name == feature_name,
        DriftResult.test_method == test_method,
        DriftResult.segment_dimension == segment_dimension,
        DriftResult.segment_value == segment_value,
    )
    stmt = (
        select(
            EvaluationWindow.id.label("window_id"),
            EvaluationWindow.window_start,
            EvaluationWindow.window_end,
            DriftResult.status,
            DriftResult.statistic.label("value"),
            DriftResult.is_significant,
            DriftResult.not_computable_reason.label("reason"),
        )
        .select_from(EvaluationWindow)
        .outerjoin(DriftResult, join_condition)
        .where(
            EvaluationWindow.model_id == model_id,
            EvaluationWindow.window_end > start_dt,
            EvaluationWindow.window_start < end_dt,
        )
        .order_by(EvaluationWindow.window_end)
    )
    actual = pd.DataFrame(session.execute(stmt).mappings().all())
    if actual.empty:
        actual = pd.DataFrame(
            columns=[
                "window_id",
                "window_start",
                "window_end",
                "status",
                "value",
                "is_significant",
                "reason",
            ]
        )

    expected = _expected_window_grid(start_dt, end_dt, window_duration)
    if expected.empty:
        return _label_state(actual)

    merged = expected.merge(actual, on=["window_start", "window_end"], how="left")
    return _label_state(merged)


def _expected_drift_signals(
    model_config: ModelConfig, profile: Profile
) -> list[tuple[str, TestMethod]]:
    signals: list[tuple[str, TestMethod]] = []
    for feature in model_config.schema_.features:
        methods = (
            profile.drift_tests.continuous.methods
            if feature.dtype == "continuous"
            else profile.drift_tests.categorical.methods
        )
        signals.extend((feature.name, TestMethod(method)) for method in methods)
    signals.append(
        (PREDICTION_SCORE_FEATURE_NAME, TestMethod(profile.drift_tests.prediction_score.method))
    )
    return signals


def fetch_feature_attribution(
    session: Session, model_config: ModelConfig, profile: Profile, window_id: int
) -> pd.DataFrame:
    """Global (non-segment) drift signals for one window, ranked by
    value / fire_threshold so effect sizes on different scales (PSI, KS
    D-statistic, Cramer's V, JSD) are comparable on one axis -- "how close
    to its own fire threshold". The expected (feature, test_method) grid
    comes from config, not the database (a methods list can legally be
    configured empty, which is exactly a 'not_configured' case for every
    feature of that dtype); actual results for this one window_id are a
    single small SQL fetch, reconciled against that grid in pandas."""
    expected = _expected_drift_signals(model_config, profile)
    stmt = select(
        DriftResult.feature_name,
        DriftResult.test_method,
        DriftResult.status,
        DriftResult.statistic.label("value"),
        DriftResult.is_significant,
        DriftResult.not_computable_reason.label("reason"),
    ).where(
        DriftResult.evaluation_window_id == window_id,
        DriftResult.segment_dimension.is_(None),
    )
    actual = pd.DataFrame(session.execute(stmt).mappings().all())
    expected_df = pd.DataFrame(expected, columns=["feature_name", "test_method"])

    if actual.empty:
        merged = expected_df.copy()
        for col in ("status", "value", "is_significant", "reason"):
            merged[col] = None
    else:
        merged = expected_df.merge(actual, on=["feature_name", "test_method"], how="left")

    merged = _label_state(merged)
    merged["fire_threshold"] = merged["test_method"].apply(
        lambda tm: drift_thresholds(profile, tm)[0]
    )
    merged["ratio"] = merged["value"] / merged["fire_threshold"]
    return merged.sort_values("ratio", ascending=False, na_position="last").reset_index(drop=True)


def known_segment_values(
    session: Session,
    model_id: str,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str,
) -> list[str]:
    """Every segment value this exact signal has ever recorded a
    DriftResult for -- segment values are data-driven (whoever is present
    in live predictions that window), not static config, so this is a SQL
    DISTINCT rather than something read from a YAML file."""
    stmt = (
        select(DriftResult.segment_value)
        .join(EvaluationWindow, DriftResult.evaluation_window_id == EvaluationWindow.id)
        .where(
            EvaluationWindow.model_id == model_id,
            DriftResult.feature_name == feature_name,
            DriftResult.test_method == test_method,
            DriftResult.segment_dimension == segment_dimension,
        )
        .distinct()
    )
    return sorted(v for v in session.scalars(stmt) if v is not None)


def fetch_segment_view(
    session: Session,
    model_id: str,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str,
    start_dt: datetime,
    end_dt: datetime,
    window_duration: timedelta,
) -> pd.DataFrame:
    """'Global' plus every segment value of `segment_dimension` that has
    ever appeared for this signal (a SQL DISTINCT over the signal's full
    history, not just the visible range, so a segment column doesn't
    appear/disappear as the date range changes), rendered over the same
    time range side by side -- so a passing global aggregate and a
    breaching individual segment show up together in one frame.

    The window axis is the EXPECTED grid (_expected_window_grid), not just
    whatever EvaluationWindow rows exist -- otherwise a window that was
    never evaluated at all is indistinguishable per-segment from one that
    was evaluated and found quiet, same as fetch_drift_timeline."""
    segment_values = known_segment_values(
        session, model_id, feature_name, test_method, segment_dimension
    )
    segments = ["Global", *segment_values]

    columns = [
        "segment",
        "window_id",
        "window_start",
        "window_end",
        "state",
        "value",
        "is_significant",
        "reason",
    ]
    expected = _expected_window_grid(start_dt, end_dt, window_duration)
    if expected.empty:
        return pd.DataFrame(columns=columns)

    windows_df = list_windows(session, model_id, start_dt, end_dt)

    stmt = (
        select(
            EvaluationWindow.id.label("window_id"),
            DriftResult.segment_dimension,
            DriftResult.segment_value,
            DriftResult.status,
            DriftResult.statistic.label("value"),
            DriftResult.is_significant,
            DriftResult.not_computable_reason.label("reason"),
        )
        .select_from(EvaluationWindow)
        .join(
            DriftResult,
            and_(
                DriftResult.evaluation_window_id == EvaluationWindow.id,
                DriftResult.feature_name == feature_name,
                DriftResult.test_method == test_method,
                or_(
                    DriftResult.segment_dimension.is_(None),
                    DriftResult.segment_dimension == segment_dimension,
                ),
            ),
        )
        .where(
            EvaluationWindow.model_id == model_id,
            EvaluationWindow.window_end > start_dt,
            EvaluationWindow.window_start < end_dt,
        )
    )
    actual = pd.DataFrame(session.execute(stmt).mappings().all())
    if actual.empty:
        actual = pd.DataFrame(
            columns=["window_id", "segment", "status", "value", "is_significant", "reason"]
        )
    else:
        actual["segment"] = actual.apply(
            lambda r: "Global" if pd.isna(r["segment_dimension"]) else r["segment_value"], axis=1
        )
        actual = actual[["window_id", "segment", "status", "value", "is_significant", "reason"]]

    # expected windows x segments, first reconciled against whichever
    # windows actually exist (bringing in window_id -- NaN where the
    # window is genuinely missing), then against the per-segment results.
    base_grid = (
        expected.assign(_key=1)
        .merge(pd.DataFrame({"segment": segments, "_key": 1}), on="_key")
        .drop(columns="_key")
    )
    base_grid = base_grid.merge(windows_df, on=["window_start", "window_end"], how="left")
    merged = base_grid.merge(actual, on=["window_id", "segment"], how="left")
    return _label_state(merged)


def fetch_performance_timeline(
    session: Session,
    model_id: str,
    metric_name: str,
    segment_dimension: str | None,
    segment_value: str | None,
    start_dt: datetime,
    end_dt: datetime,
    window_duration: timedelta,
) -> pd.DataFrame:
    """Each window's FIRST-computed PerformanceResult (the initial,
    pre-label-backfill snapshot) and its LATEST (after however many
    retroactive recomputes label arrival has triggered since), picked out
    with SQL row_number() window functions ranking a window's revisions by
    computed_at -- not by loading every revision into pandas and reducing
    there. A window whose initial and latest values match had no
    meaningful retroactive change; one where they differ is exactly what
    the label-backfill annotation is for.

    The window axis is the EXPECTED grid (_expected_window_grid), not just
    whatever EvaluationWindow rows exist, for the same reason as
    fetch_drift_timeline: a window never evaluated at all must not look
    identical to one evaluated and found quiet.

    Ties on computed_at are broken by id: Postgres's now() is transaction-
    start time, so two revisions of the same window recomputed in the same
    transaction (the ordinary case for one scheduler tick recomputing the
    several windows a label batch flagged) can share an identical
    computed_at -- the same
    reasoning driftwatch.evaluation.performance.recompute_performance_for_window
    already applies to labels sharing received_at. Without the id
    tiebreaker, ROW_NUMBER()'s ordering among tied rows is unspecified."""
    first_rank = func.row_number().over(
        partition_by=PerformanceResult.evaluation_window_id,
        order_by=(PerformanceResult.computed_at.asc(), PerformanceResult.id.asc()),
    )
    last_rank = func.row_number().over(
        partition_by=PerformanceResult.evaluation_window_id,
        order_by=(PerformanceResult.computed_at.desc(), PerformanceResult.id.desc()),
    )
    ranked = (
        select(
            PerformanceResult.evaluation_window_id,
            PerformanceResult.status,
            PerformanceResult.metric_value,
            PerformanceResult.not_computable_reason,
            first_rank.label("rn_first"),
            last_rank.label("rn_last"),
        )
        .where(
            PerformanceResult.metric_name == metric_name,
            PerformanceResult.segment_dimension == segment_dimension,
            PerformanceResult.segment_value == segment_value,
        )
        .subquery()
    )
    stmt = (
        select(
            EvaluationWindow.id.label("window_id"),
            EvaluationWindow.window_start,
            EvaluationWindow.window_end,
            ranked.c.status,
            ranked.c.metric_value,
            ranked.c.not_computable_reason,
            ranked.c.rn_first,
            ranked.c.rn_last,
        )
        .select_from(EvaluationWindow)
        .outerjoin(ranked, ranked.c.evaluation_window_id == EvaluationWindow.id)
        .where(
            EvaluationWindow.model_id == model_id,
            EvaluationWindow.window_end > start_dt,
            EvaluationWindow.window_start < end_dt,
        )
        .order_by(EvaluationWindow.window_end)
    )
    raw = pd.DataFrame(session.execute(stmt).mappings().all())
    if raw.empty:
        raw = pd.DataFrame(
            columns=[
                "window_id",
                "window_start",
                "window_end",
                "status",
                "metric_value",
                "not_computable_reason",
                "rn_first",
                "rn_last",
            ]
        )

    columns = ["window_id", "window_start", "window_end", "revision", "status", "value", "reason"]
    expected = _expected_window_grid(start_dt, end_dt, window_duration)
    if expected.empty:
        return pd.DataFrame(columns=[*columns, "state"])
    merged_windows = expected.merge(raw, on=["window_start", "window_end"], how="left")

    rows: list[dict[str, Any]] = []
    for _, row in merged_windows.iterrows():
        base = {
            "window_id": row["window_id"],
            "window_start": row["window_start"],
            "window_end": row["window_end"],
        }
        # No rn_first covers two distinct cases, told apart by window_id in
        # _label_state: the window itself is missing (window_id NaN -- no
        # EvaluationWindow at all), or the window exists but no
        # PerformanceResult row does for this metric (not_configured).
        if pd.isna(row["rn_first"]):
            empty = {"status": None, "value": None, "reason": None}
            rows.append({**base, "revision": "initial", **empty})
            rows.append({**base, "revision": "latest", **empty})
            continue
        if row["rn_first"] == 1:
            rows.append(
                {
                    **base,
                    "revision": "initial",
                    "status": row["status"],
                    "value": row["metric_value"],
                    "reason": row["not_computable_reason"],
                }
            )
        if row["rn_last"] == 1:
            rows.append(
                {
                    **base,
                    "revision": "latest",
                    "status": row["status"],
                    "value": row["metric_value"],
                    "reason": row["not_computable_reason"],
                }
            )
    return _label_state(pd.DataFrame(rows))


def fetch_alert_history(
    session: Session, model_id: str, start_dt: datetime, end_dt: datetime
) -> pd.DataFrame:
    """Alert rows already ARE the aggregate, stateful representation --
    this is a direct filtered read, no reduction needed.

    Every timestamp returned here is DATA time -- an evaluation window's
    own window_start/window_end -- never Alert.first_opened_at/
    escalated_at/resolved_at/last_seen_at. Those four columns are
    wall-clock evaluation-run time, correct for what they're actually for
    (the engine's re-notify-on-transition currency tracking), but showing
    them in this table would be actively misleading: a historical backfill
    evaluates a whole date range of windows in one fast real-time burst, so
    every alert's wall-clock timestamps would show today's date sitting
    next to a data event from months earlier, with nothing in the table to
    tell a reader the two apart. Showing data time only (sourced from
    evidence_window_id / escalation_evidence_window_id / last_seen_window_id)
    avoids that confusion outright, and keeps this table on the same time
    axis as every chart around it."""
    open_window = aliased(EvaluationWindow)
    seen_window = aliased(EvaluationWindow)
    escalation_window = aliased(EvaluationWindow)
    stmt = (
        select(
            Alert.id,
            Alert.kind,
            Alert.signal_name,
            Alert.feature_name,
            Alert.segment_dimension,
            Alert.segment_value,
            Alert.status,
            Alert.is_aggregate,
            Alert.aggregated_signal_count,
            Alert.evidence_statistic,
            Alert.evidence_threshold,
            Alert.escalation_evidence_statistic,
            Alert.escalation_evidence_threshold,
            open_window.window_start.label("opened_window_start"),
            open_window.window_end.label("opened_window_end"),
            escalation_window.window_end.label("escalated_window_end"),
            seen_window.window_end.label("last_seen_window_end"),
        )
        .select_from(Alert)
        .join(open_window, Alert.evidence_window_id == open_window.id)
        .join(seen_window, Alert.last_seen_window_id == seen_window.id)
        .outerjoin(escalation_window, Alert.escalation_evidence_window_id == escalation_window.id)
        .where(
            Alert.model_id == model_id,
            open_window.window_start < end_dt,
            seen_window.window_end > start_dt,
        )
        .order_by(open_window.window_start.desc())
    )
    return pd.DataFrame(session.execute(stmt).mappings().all())


def _drift_history_in_range(
    session: Session,
    model_id: str,
    feature_name: str,
    test_method: TestMethod,
    segment_dimension: str | None,
    segment_value: str | None,
    start_dt: datetime,
    end_dt: datetime,
) -> list[tuple[int, DriftResult]]:
    """(window_id, DriftResult) pairs, oldest-first, for exactly this
    signal identity within [start_dt, end_dt). Same identity shape as
    driftwatch.alerting.streaks.fetch_drift_history, but date-bounded (for
    a reporting view over an explicit range) instead of count-bounded (for
    the engine's persistence rescan), and ascending since suppressed-
    episode detection walks history forward, not backward from "now" like
    the engine's own streak recompute does."""
    stmt = (
        select(EvaluationWindow.id, DriftResult)
        .join(EvaluationWindow, DriftResult.evaluation_window_id == EvaluationWindow.id)
        .where(
            EvaluationWindow.model_id == model_id,
            DriftResult.feature_name == feature_name,
            DriftResult.test_method == test_method,
            DriftResult.segment_dimension == segment_dimension,
            DriftResult.segment_value == segment_value,
            EvaluationWindow.window_end > start_dt,
            EvaluationWindow.window_start < end_dt,
        )
        .order_by(EvaluationWindow.window_end.asc())
    )
    return [(row[0], row[1]) for row in session.execute(stmt).all()]


def _performance_history_in_range(
    session: Session,
    model_id: str,
    metric_name: str,
    segment_dimension: str | None,
    segment_value: str | None,
    start_dt: datetime,
    end_dt: datetime,
) -> list[tuple[int, PerformanceResult]]:
    """Latest-per-window PerformanceResult, oldest-window-first, within
    range -- same dedup rule as driftwatch.alerting.streaks
    .fetch_performance_history (a window recomputed N times after label
    backfill must contribute exactly one entry), just date-bounded and
    ascending instead of count-bounded and descending."""
    stmt = (
        select(EvaluationWindow.id, PerformanceResult)
        .join(EvaluationWindow, PerformanceResult.evaluation_window_id == EvaluationWindow.id)
        .where(
            EvaluationWindow.model_id == model_id,
            PerformanceResult.metric_name == metric_name,
            PerformanceResult.segment_dimension == segment_dimension,
            PerformanceResult.segment_value == segment_value,
            EvaluationWindow.window_end > start_dt,
            EvaluationWindow.window_start < end_dt,
        )
        .order_by(EvaluationWindow.window_end.asc(), PerformanceResult.computed_at.desc())
    )
    latest_per_window: dict[int, PerformanceResult] = {}
    order: list[int] = []
    for window_id, result in session.execute(stmt).all():
        if window_id not in latest_per_window:
            order.append(window_id)
        latest_per_window[window_id] = result
    return [(window_id, latest_per_window[window_id]) for window_id in order]


def fetch_suppressed_episodes(
    session: Session,
    model_id: str,
    model_config: ModelConfig,
    profile: Profile,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    """Every streak (drift or performance, breach or not_computable) that
    reached at least one window but never reached the persistence count
    its profile requires to open a real Alert -- the complement of
    fetch_alert_history. Reuses classify_drift/classify_performance from
    driftwatch.alerting.streaks (the exact functions the real alerting
    engine uses) so this can never silently drift from what the engine
    would actually have decided; the run-length grouping itself is
    driftwatch.alerting.reporting.find_suppressed_episodes, also pure and
    already unit-tested there."""
    rows: list[dict[str, Any]] = []

    drift_identities_stmt = (
        select(
            DriftResult.feature_name,
            DriftResult.test_method,
            DriftResult.segment_dimension,
            DriftResult.segment_value,
        )
        .join(EvaluationWindow, DriftResult.evaluation_window_id == EvaluationWindow.id)
        .where(
            EvaluationWindow.model_id == model_id,
            EvaluationWindow.window_end > start_dt,
            EvaluationWindow.window_start < end_dt,
        )
        .distinct()
    )
    drift_gates: tuple[tuple[StreakKind, int], ...] = (
        ("breach", profile.alerting.fire_persistence_windows),
        ("not_computable", profile.alerting.not_computable_persistence_windows),
    )
    for feature_name, test_method, segment_dimension, segment_value in session.execute(
        drift_identities_stmt
    ).all():
        drift_history = _drift_history_in_range(
            session,
            model_id,
            feature_name,
            test_method,
            segment_dimension,
            segment_value,
            start_dt,
            end_dt,
        )
        fire_threshold, clear_threshold = drift_thresholds(profile, test_method)
        drift_classified = [
            (window_id, classify_drift(result, fire_threshold, clear_threshold), result.statistic)
            for window_id, result in drift_history
        ]
        for target_kind, persistence in drift_gates:
            for episode in find_suppressed_episodes(drift_classified, persistence, target_kind):
                rows.append(
                    {
                        "kind": "drift",
                        "signal_name": test_method.value,
                        "feature_name": feature_name,
                        "segment_dimension": segment_dimension,
                        "segment_value": segment_value,
                        "episode_kind": episode.kind,
                        "start_window_id": episode.start_window_id,
                        "end_window_id": episode.end_window_id,
                        "length": episode.length,
                        "persistence_required": persistence,
                        "peak_value": episode.peak_value,
                    }
                )

    metric_specs = {
        spec.name: spec for spec in profile.performance_metrics if spec.alert_direction is not None
    }
    if metric_specs:
        perf_identities_stmt = (
            select(
                PerformanceResult.metric_name,
                PerformanceResult.segment_dimension,
                PerformanceResult.segment_value,
            )
            .join(EvaluationWindow, PerformanceResult.evaluation_window_id == EvaluationWindow.id)
            .where(
                EvaluationWindow.model_id == model_id,
                EvaluationWindow.window_end > start_dt,
                EvaluationWindow.window_start < end_dt,
                PerformanceResult.metric_name.in_(metric_specs.keys()),
            )
            .distinct()
        )
        perf_gates: tuple[tuple[StreakKind, int], ...] = (
            ("breach", profile.alerting.fire_persistence_windows),
            ("not_computable", profile.alerting.not_computable_persistence_windows),
        )
        for metric_name, segment_dimension, segment_value in session.execute(
            perf_identities_stmt
        ).all():
            spec = metric_specs[metric_name]
            assert spec.alert_direction is not None
            assert spec.fire_threshold is not None
            assert spec.clear_threshold is not None
            perf_history = _performance_history_in_range(
                session, model_id, metric_name, segment_dimension, segment_value, start_dt, end_dt
            )
            perf_classified = [
                (
                    window_id,
                    classify_performance(
                        result, spec.alert_direction, spec.fire_threshold, spec.clear_threshold
                    ),
                    result.metric_value,
                )
                for window_id, result in perf_history
            ]
            for target_kind, persistence in perf_gates:
                for episode in find_suppressed_episodes(perf_classified, persistence, target_kind):
                    rows.append(
                        {
                            "kind": "performance",
                            "signal_name": metric_name,
                            "feature_name": None,
                            "segment_dimension": segment_dimension,
                            "segment_value": segment_value,
                            "episode_kind": episode.kind,
                            "start_window_id": episode.start_window_id,
                            "end_window_id": episode.end_window_id,
                            "length": episode.length,
                            "persistence_required": persistence,
                            "peak_value": episode.peak_value,
                        }
                    )

    return pd.DataFrame(rows)
