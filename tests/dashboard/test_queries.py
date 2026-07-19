from datetime import UTC, date, datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from driftwatch.config.schema import ModelConfig, Profile
from driftwatch.dashboard import queries
from driftwatch.db.models import (
    Alert,
    AlertKind,
    AlertStatus,
    Baseline,
    DriftResult,
    EvaluationWindow,
    MetricStatus,
    Model,
    PerformanceResult,
    TestMethod,
)

MODEL_ID = "example-model"


def _setup_model(db_session: Session) -> Baseline:
    db_session.add(Model(model_id=MODEL_ID))
    db_session.flush()
    baseline = Baseline(model_id=MODEL_ID, is_active=True, binning_config={})
    db_session.add(baseline)
    db_session.flush()
    return baseline


def _make_window(
    db_session: Session, baseline: Baseline, index: int, config_hash: str = "hash"
) -> EvaluationWindow:
    window_start = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(hours=index)
    window = EvaluationWindow(
        model_id=MODEL_ID,
        baseline_id=baseline.id,
        window_start=window_start,
        window_end=window_start + timedelta(hours=1),
        config_hash=config_hash,
        evaluated_at=datetime.now(UTC),
    )
    db_session.add(window)
    db_session.flush()
    return window


def _add_drift_result(
    db_session: Session,
    window: EvaluationWindow,
    *,
    feature_name: str = "age",
    test_method: TestMethod = TestMethod.PSI,
    statistic: float | None = None,
    status: MetricStatus = MetricStatus.COMPUTED,
    reason: str | None = None,
    segment_dimension: str | None = None,
    segment_value: str | None = None,
    is_significant: bool | None = None,
) -> DriftResult:
    result = DriftResult(
        evaluation_window_id=window.id,
        feature_name=feature_name,
        test_method=test_method,
        segment_dimension=segment_dimension,
        segment_value=segment_value,
        status=status,
        statistic=statistic,
        is_significant=is_significant,
        not_computable_reason=reason,
        n_baseline=10,
        n_live=10,
    )
    db_session.add(result)
    db_session.flush()
    return result


def _add_performance_result(
    db_session: Session,
    window: EvaluationWindow,
    *,
    metric_name: str = "pr_auc",
    value: float | None = None,
    status: MetricStatus = MetricStatus.COMPUTED,
    is_retroactive: bool = False,
    reason: str | None = None,
    computed_at: datetime | None = None,
) -> PerformanceResult:
    result = PerformanceResult(
        evaluation_window_id=window.id,
        metric_name=metric_name,
        status=status,
        metric_value=value,
        not_computable_reason=reason,
        n_labeled=10,
        is_retroactive=is_retroactive,
        **({"computed_at": computed_at} if computed_at is not None else {}),
    )
    db_session.add(result)
    db_session.flush()
    return result


def _make_model_config() -> ModelConfig:
    return ModelConfig.model_validate(
        {
            "model_id": MODEL_ID,
            "profile": "test",
            "prediction_type": "binary_classification",
            "schema": {
                "features": [
                    {"name": "age", "dtype": "continuous", "nullable": False},
                    {"name": "region", "dtype": "categorical", "nullable": False},
                ],
                "prediction": {"dtype": "continuous"},
                "label": {"dtype": "continuous"},
            },
        }
    )


def _make_profile() -> Profile:
    return Profile.model_validate(
        {
            "name": "test",
            "description": "test",
            "evaluation": {"window": "1h", "min_window_size": 1, "watermark": "1m"},
            "drift_tests": {
                "continuous": {
                    "methods": ["psi", "ks"],
                    "psi_threshold": 0.2,
                    "psi_clear_threshold": 0.1,
                    "ks_statistic_threshold": 0.2,
                    "ks_statistic_clear_threshold": 0.1,
                },
                "categorical": {
                    "methods": ["chi_square"],
                    "cramers_v_threshold": 0.2,
                    "cramers_v_clear_threshold": 0.1,
                },
                "prediction_score": {
                    "method": "jsd",
                    "jsd_threshold": 0.2,
                    "jsd_clear_threshold": 0.1,
                },
            },
            "multiple_comparison_correction": {"method": "benjamini_hochberg", "fdr_alpha": 0.05},
            "alerting": {
                "fire_persistence_windows": 2,
                "escalate_persistence_windows": 4,
                "resolve_persistence_windows": 3,
                "not_computable_persistence_windows": 2,
                "history_scan_windows": 6,
                "max_alerts_per_run": 3,
            },
            "performance_metrics": [
                {
                    "name": "pr_auc",
                    "alert_direction": "below",
                    "fire_threshold": 0.5,
                    "clear_threshold": 0.6,
                }
            ],
        }
    )


_FULL_RANGE = queries.date_range_to_datetimes(date(2025, 12, 31), date(2026, 1, 10))
WINDOW_DURATION = timedelta(hours=1)  # matches _make_window's hourly spacing


def test_date_range_to_datetimes_is_half_open_inclusive_of_end_day() -> None:
    start_dt, end_dt = queries.date_range_to_datetimes(date(2026, 1, 1), date(2026, 1, 5))

    assert start_dt == datetime(2026, 1, 1, tzinfo=UTC)
    assert end_dt == datetime(2026, 1, 6, tzinfo=UTC)


def test_list_models_only_includes_models_with_evaluated_windows(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    db_session.add(Model(model_id="no-windows-model"))
    db_session.flush()
    _make_window(db_session, baseline, 0)

    assert queries.list_models(db_session) == [MODEL_ID]


def test_default_time_range_matches_min_max_evaluation_windows(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    _make_window(db_session, baseline, 0)
    last = _make_window(db_session, baseline, 10)

    result = queries.default_time_range(db_session, MODEL_ID)

    assert result == (date(2026, 1, 1), last.window_end.date())


def test_default_time_range_none_when_no_windows(db_session: Session) -> None:
    db_session.add(Model(model_id=MODEL_ID))
    db_session.flush()

    assert queries.default_time_range(db_session, MODEL_ID) is None


def test_fetch_drift_timeline_distinguishes_all_four_states(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    w0 = _make_window(db_session, baseline, 0)
    w1 = _make_window(db_session, baseline, 1)
    w2 = _make_window(db_session, baseline, 2)
    # index 3 deliberately never created -- a genuinely missing window
    # (scheduler downtime / backfill gap), not merely a quiet one

    _add_drift_result(db_session, w0, statistic=0.5, is_significant=True)
    _add_drift_result(db_session, w1, status=MetricStatus.NOT_COMPUTABLE, reason="too few rows")
    # w2: no DriftResult row at all for this signal -> not_configured

    start_dt, end_dt = queries.date_range_to_datetimes(date(2026, 1, 1), date(2026, 1, 1))
    df = queries.fetch_drift_timeline(
        db_session,
        MODEL_ID,
        "age",
        TestMethod.PSI,
        None,
        None,
        start_dt,
        end_dt,
        WINDOW_DURATION,
    )

    by_start = df.set_index("window_start")
    assert by_start.loc[w0.window_start, "state"] == "computed"
    assert by_start.loc[w0.window_start, "value"] == 0.5
    assert by_start.loc[w1.window_start, "state"] == "not_computable"
    assert by_start.loc[w1.window_start, "reason"] == "too few rows"
    assert by_start.loc[w2.window_start, "state"] == "not_configured"

    missing_slot_start = w0.window_start + timedelta(hours=3)
    assert by_start.loc[missing_slot_start, "state"] == "missing"
    assert pd.isna(by_start.loc[missing_slot_start, "window_id"])
    assert "not evaluated" in by_start.loc[missing_slot_start, "reason"]


def test_fetch_feature_attribution_ranks_and_labels_states(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    window = _make_window(db_session, baseline, 0)
    model_config = _make_model_config()
    profile = _make_profile()

    _add_drift_result(
        db_session, window, feature_name="age", test_method=TestMethod.PSI, statistic=0.3
    )
    _add_drift_result(
        db_session,
        window,
        feature_name="region",
        test_method=TestMethod.CHI_SQUARE,
        status=MetricStatus.NOT_COMPUTABLE,
        reason="below min segment size",
    )
    _add_drift_result(
        db_session,
        window,
        feature_name="__prediction_score__",
        test_method=TestMethod.JSD,
        statistic=0.05,
    )
    # age/ks: no row at all -> not_configured

    df = queries.fetch_feature_attribution(db_session, model_config, profile, window.id)

    assert len(df) == 4  # age/psi, age/ks, region/chi_square, prediction_score/jsd
    by_signal = df.set_index(["feature_name", "test_method"])
    assert by_signal.loc[("age", TestMethod.PSI), "state"] == "computed"
    assert by_signal.loc[("age", TestMethod.KS), "state"] == "not_configured"
    assert by_signal.loc[("region", TestMethod.CHI_SQUARE), "state"] == "not_computable"
    assert by_signal.loc[("region", TestMethod.CHI_SQUARE), "reason"] == "below min segment size"

    # ranked descending by statistic/fire_threshold: age/psi (0.3/0.2=1.5) ranks
    # above prediction_score/jsd (0.05/0.2=0.25)
    computed = df[df["state"] == "computed"].reset_index(drop=True)
    assert computed.iloc[0]["feature_name"] == "age"
    assert computed.iloc[0]["test_method"] == TestMethod.PSI


def test_fetch_segment_view_shows_global_and_segments_side_by_side(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    w0 = _make_window(db_session, baseline, 0)
    w1 = _make_window(db_session, baseline, 1)
    # index 2 deliberately never created -- a genuinely missing window

    _add_drift_result(db_session, w0, statistic=0.05)  # global, clear
    _add_drift_result(
        db_session,
        w0,
        statistic=0.9,
        segment_dimension="region",
        segment_value="EU",
        is_significant=True,
    )
    _add_drift_result(db_session, w1, statistic=0.04)
    # w1: EU segment has no row this window -> not_configured

    start_dt, end_dt = queries.date_range_to_datetimes(date(2026, 1, 1), date(2026, 1, 1))
    df = queries.fetch_segment_view(
        db_session, MODEL_ID, "age", TestMethod.PSI, "region", start_dt, end_dt, WINDOW_DURATION
    )

    assert set(df["segment"]) == {"Global", "EU"}
    at_w0 = df[df["window_start"] == w0.window_start].set_index("segment")
    assert at_w0.loc["Global", "state"] == "computed"
    assert at_w0.loc["EU", "state"] == "computed"
    assert at_w0.loc["EU", "value"] == 0.9

    at_w1 = df[df["window_start"] == w1.window_start].set_index("segment")
    assert at_w1.loc["Global", "state"] == "computed"
    assert at_w1.loc["EU", "state"] == "not_configured"

    missing_slot_start = w0.window_start + timedelta(hours=2)
    at_missing = df[df["window_start"] == missing_slot_start].set_index("segment")
    assert at_missing.loc["Global", "state"] == "missing"
    assert at_missing.loc["EU", "state"] == "missing"


def test_fetch_performance_timeline_distinguishes_initial_and_latest(db_session: Session) -> None:
    baseline = _setup_model(db_session)
    window = _make_window(db_session, baseline, 0)

    _add_performance_result(
        db_session,
        window,
        value=0.3,
        is_retroactive=False,
        computed_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
    )
    _add_performance_result(
        db_session,
        window,
        value=0.7,
        is_retroactive=True,
        computed_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    start_dt, end_dt = queries.date_range_to_datetimes(date(2026, 1, 1), date(2026, 1, 1))
    df = queries.fetch_performance_timeline(
        db_session, MODEL_ID, "pr_auc", None, None, start_dt, end_dt, WINDOW_DURATION
    )

    at_window = df[df["window_start"] == window.window_start].set_index("revision")
    assert at_window.loc["initial", "value"] == 0.3
    assert at_window.loc["latest", "value"] == 0.7


def test_fetch_performance_timeline_not_computable_not_configured_and_missing(
    db_session: Session,
) -> None:
    baseline = _setup_model(db_session)
    w0 = _make_window(db_session, baseline, 0)
    w1 = _make_window(db_session, baseline, 1)
    # index 2 deliberately never created -- a genuinely missing window

    _add_performance_result(
        db_session, w0, status=MetricStatus.NOT_COMPUTABLE, reason="no labels yet"
    )
    # w1: no PerformanceResult row at all for this metric -> not_configured

    start_dt, end_dt = queries.date_range_to_datetimes(date(2026, 1, 1), date(2026, 1, 1))
    df = queries.fetch_performance_timeline(
        db_session, MODEL_ID, "pr_auc", None, None, start_dt, end_dt, WINDOW_DURATION
    )

    at_w0 = df[df["window_start"] == w0.window_start].set_index("revision")
    assert at_w0.loc["initial", "state"] == "not_computable"
    assert at_w0.loc["initial", "reason"] == "no labels yet"

    at_w1 = df[df["window_start"] == w1.window_start].set_index("revision")
    assert at_w1.loc["initial", "state"] == "not_configured"

    missing_slot_start = w0.window_start + timedelta(hours=2)
    at_missing = df[df["window_start"] == missing_slot_start].set_index("revision")
    assert at_missing.loc["initial", "state"] == "missing"
    assert at_missing.loc["latest", "state"] == "missing"


def test_fetch_alert_history_filters_by_evaluation_window_range(db_session: Session) -> None:
    """Filtered by the underlying EvaluationWindow's own time range, not by
    Alert.first_opened_at/last_seen_at -- a historical backfill evaluates a
    whole date range of windows in one real-time burst, which would put
    every alert's wall-clock timestamps far outside the very date range
    being displayed if those columns drove the filter instead."""
    baseline = _setup_model(db_session)
    window = _make_window(db_session, baseline, 0)  # 2026-01-01 00:00-01:00 UTC
    alert = Alert(
        model_id=MODEL_ID,
        kind=AlertKind.DRIFT,
        signal_name="psi",
        feature_name="age",
        status=AlertStatus.OPEN,
        evidence_window_id=window.id,
        last_seen_window_id=window.id,
        # wall-clock columns deliberately set far outside the window's own
        # date range, to prove they aren't what the filter uses
        first_opened_at=datetime(2026, 6, 1, tzinfo=UTC),
        last_seen_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    db_session.add(alert)
    db_session.flush()

    in_range = queries.fetch_alert_history(
        db_session, MODEL_ID, *queries.date_range_to_datetimes(date(2026, 1, 1), date(2026, 1, 5))
    )
    assert len(in_range) == 1
    assert "first_opened_at" not in in_range.columns
    assert "last_seen_at" not in in_range.columns
    assert in_range.iloc[0]["opened_window_start"] == window.window_start
    assert in_range.iloc[0]["opened_window_end"] == window.window_end
    assert in_range.iloc[0]["last_seen_window_end"] == window.window_end

    out_of_range = queries.fetch_alert_history(
        db_session, MODEL_ID, *queries.date_range_to_datetimes(date(2026, 2, 1), date(2026, 2, 5))
    )
    assert out_of_range.empty


def test_fetch_alert_history_escalated_window_end_is_data_time_and_nullable(
    db_session: Session,
) -> None:
    baseline = _setup_model(db_session)
    opened_window = _make_window(db_session, baseline, 0)
    escalated_window = _make_window(db_session, baseline, 5)

    escalated_alert = Alert(
        model_id=MODEL_ID,
        kind=AlertKind.DRIFT,
        signal_name="psi",
        feature_name="age",
        status=AlertStatus.ESCALATED,
        evidence_window_id=opened_window.id,
        escalation_evidence_window_id=escalated_window.id,
        last_seen_window_id=escalated_window.id,
    )
    still_open_alert = Alert(
        model_id=MODEL_ID,
        kind=AlertKind.DRIFT,
        signal_name="ks",
        feature_name="age",
        status=AlertStatus.OPEN,
        evidence_window_id=opened_window.id,
        last_seen_window_id=opened_window.id,
    )
    db_session.add_all([escalated_alert, still_open_alert])
    db_session.flush()

    df = queries.fetch_alert_history(
        db_session, MODEL_ID, *queries.date_range_to_datetimes(date(2026, 1, 1), date(2026, 1, 5))
    )
    by_signal = df.set_index("signal_name")
    assert by_signal.loc["psi", "escalated_window_end"] == escalated_window.window_end
    assert pd.isna(by_signal.loc["ks", "escalated_window_end"])


def test_fetch_suppressed_episodes_reports_short_run_but_not_full_persistence(
    db_session: Session,
) -> None:
    baseline = _setup_model(db_session)
    profile = _make_profile()  # fire_persistence_windows=2
    model_config = _make_model_config()

    w0 = _make_window(db_session, baseline, 0)
    _add_drift_result(
        db_session, w0, feature_name="age", test_method=TestMethod.PSI, statistic=0.5
    )  # single breaching window -- below persistence of 2

    w1 = _make_window(db_session, baseline, 1)
    _add_drift_result(
        db_session, w1, feature_name="region", test_method=TestMethod.CHI_SQUARE, statistic=0.9
    )
    w2 = _make_window(db_session, baseline, 2)
    _add_drift_result(
        db_session, w2, feature_name="region", test_method=TestMethod.CHI_SQUARE, statistic=0.95
    )  # 2 consecutive breaches -- reaches persistence of 2

    start_dt, end_dt = _FULL_RANGE
    df = queries.fetch_suppressed_episodes(
        db_session, MODEL_ID, model_config, profile, start_dt, end_dt
    )

    suppressed_signals = set(zip(df["feature_name"], df["signal_name"], strict=True))
    assert ("age", "psi") in suppressed_signals
    assert ("region", "chi_square") not in suppressed_signals
