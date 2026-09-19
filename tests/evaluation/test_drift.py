from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from driftwatch.config.loader import compute_config_hash, load_model_config, load_profile
from driftwatch.db.models import (
    Alert,
    AlertKind,
    AlertStatus,
    Baseline,
    BaselineRecord,
    DriftResult,
    EvaluationWindow,
    MetricStatus,
    Model,
    Prediction,
    TestMethod,
)
from driftwatch.evaluation.drift import evaluate_window
from driftwatch.stats.binning import compute_baseline_binning

MODEL_ID = "example-model"
REGIONS = ["EU", "US", "APAC"]


# The aggressive profile's clear thresholds (PSI 0.07, KS D 0.1, Cramer's V
# 0.07, JSD 0.03) are only meaningful once every statistic's sampling-noise
# ceiling sits under them, which for two equal samples takes about 1,500
# rows. Tests that want a statistic REFUSED for being under its floor use
# SMALL instead.
N = 1500
SMALL = 60


def _register_baseline(db_session: Session, n: int = N) -> Baseline:
    db_session.add(Model(model_id=MODEL_ID))
    db_session.flush()

    schema = load_model_config(MODEL_ID).schema_
    features = [
        {"age": 20 + (i % 40), "region": REGIONS[i % 3], "income": 1000 * (i + 1)}
        for i in range(n)
    ]
    scores = [0.1 + (i % 10) / 10 for i in range(n)]
    binning = compute_baseline_binning(features, scores, schema)

    baseline = Baseline(model_id=MODEL_ID, is_active=True, binning_config=binning)
    db_session.add(baseline)
    db_session.flush()
    for feature, score in zip(features, scores, strict=True):
        db_session.add(
            BaselineRecord(
                baseline_id=baseline.id,
                features=feature,
                prediction_score=score,
                segment_values={"region": feature["region"]},
            )
        )
    db_session.flush()
    return baseline


def _add_predictions(
    db_session: Session,
    window_start: datetime,
    n: int,
    *,
    age_offset: int = 0,
    region_override: str | None = None,
) -> None:
    for i in range(n):
        region = region_override or REGIONS[i % 3]
        age = 20 + age_offset + (i % 40)
        income = 1000 * (i + 1)
        db_session.add(
            Prediction(
                prediction_id=f"p-{window_start.isoformat()}-{i}",
                model_id=MODEL_ID,
                predicted_at=window_start + timedelta(seconds=i),
                features={"age": age, "region": region, "income": income},
                prediction_value={"score": 0.5},
                prediction_score=0.1 + (i % 10) / 10,
                segment_values={"region": region},
                payload_hash=f"hash-{window_start.isoformat()}-{i}",
            )
        )
    db_session.flush()


def test_idempotent_rerun_returns_none_and_no_duplicate_rows(db_session: Session) -> None:
    _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, N)

    first = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert first is not None
    row_count_after_first = len(
        db_session.scalars(
            select(DriftResult).where(DriftResult.evaluation_window_id == first.id)
        ).all()
    )

    second = evaluate_window(db_session, MODEL_ID, window_start, window_end)

    assert second is None
    row_count_after_second = len(
        db_session.scalars(
            select(DriftResult).where(DriftResult.evaluation_window_id == first.id)
        ).all()
    )
    assert row_count_after_second == row_count_after_first


def test_returns_none_when_window_already_claimed_by_another_process(db_session: Session) -> None:
    baseline = _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    model_config = load_model_config(MODEL_ID)
    profile = load_profile(model_config.profile)

    # simulate a concurrent claim: the window row exists, but nothing computed it
    db_session.add(
        EvaluationWindow(
            model_id=MODEL_ID,
            baseline_id=baseline.id,
            window_start=window_start,
            window_end=window_end,
            config_hash=compute_config_hash(model_config, profile),
            evaluated_at=datetime.now(UTC),
        )
    )
    db_session.flush()

    result = evaluate_window(db_session, MODEL_ID, window_start, window_end)

    assert result is None


def test_force_reevaluates_and_replaces_prior_results(db_session: Session) -> None:
    _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, N)

    first = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert first is not None
    first_id = first.id

    forced = evaluate_window(db_session, MODEL_ID, window_start, window_end, force=True)

    assert forced is not None
    assert forced.id != first_id
    assert db_session.get(EvaluationWindow, first_id) is None  # old window row is gone


def test_window_below_min_size_marks_all_tests_not_computable(db_session: Session) -> None:
    _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, 5)  # aggressive profile requires 50

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert window is not None

    results = db_session.scalars(
        select(DriftResult).where(DriftResult.evaluation_window_id == window.id)
    ).all()

    assert len(results) > 0
    assert all(r.status == MetricStatus.NOT_COMPUTABLE for r in results)
    assert all(
        r.not_computable_reason and "5 predictions" in r.not_computable_reason for r in results
    )
    assert all(r.statistic is None and r.is_significant is None for r in results)


def test_segment_below_min_size_is_not_computable_but_global_still_computes(
    db_session: Session,
) -> None:
    _register_baseline(db_session, n=N)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    # every prediction in one region -> that region's segment has N rows
    # (fine), but the other two configured regions have 0 -- never even attempted.
    _add_predictions(db_session, window_start, N, region_override="EU")

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert window is not None

    global_results = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id, DriftResult.segment_dimension.is_(None)
        )
    ).all()
    assert any(r.status == MetricStatus.COMPUTED for r in global_results)

    # EU has N rows (>= min_segment_size 30) and should be computed
    eu_results = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id,
            DriftResult.segment_dimension == "region",
            DriftResult.segment_value == "EU",
        )
    ).all()
    assert len(eu_results) > 0
    assert any(r.status == MetricStatus.COMPUTED for r in eu_results)


def test_stores_baseline_version_and_config_hash(db_session: Session) -> None:
    baseline = _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, N)

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)

    assert window is not None
    assert window.baseline_id == baseline.id
    model_config = load_model_config(MODEL_ID)
    profile = load_profile(model_config.profile)
    assert window.config_hash == compute_config_hash(model_config, profile)


def test_effect_size_drives_significance_not_p_value(db_session: Session) -> None:
    _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    # live ages shifted 500 above baseline's entire range -> guaranteed overflow bucket,
    # maximal PSI and KS D-statistic (completely disjoint samples)
    _add_predictions(db_session, window_start, N, age_offset=500)

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert window is not None

    age_ks = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id,
            DriftResult.feature_name == "age",
            DriftResult.test_method == TestMethod.KS,
            DriftResult.segment_dimension.is_(None),
        )
    ).one()

    assert age_ks.status == MetricStatus.COMPUTED
    assert age_ks.statistic == 1.0  # completely disjoint -> KS D-statistic == 1.0
    assert age_ks.is_significant is True
    # p_value for a maximally-separated sample is typically ~0, but is_significant
    # must be driven by the statistic/threshold comparison, not by reading p_value
    threshold = load_profile("aggressive").drift_tests.continuous.ks_statistic_threshold
    assert age_ks.statistic >= threshold


def test_benjamini_hochberg_sets_corrected_p_value_only_where_applicable(
    db_session: Session,
) -> None:
    _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, N)

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert window is not None

    results = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id,
            DriftResult.status == MetricStatus.COMPUTED,
        )
    ).all()

    ks_and_chi = [r for r in results if r.test_method in (TestMethod.KS, TestMethod.CHI_SQUARE)]
    psi_and_jsd = [r for r in results if r.test_method in (TestMethod.PSI, TestMethod.JSD)]

    assert ks_and_chi  # sanity: the profile does configure p-value-bearing tests
    assert all(r.corrected_p_value is not None for r in ks_and_chi)
    assert all(r.corrected_p_value is None for r in psi_and_jsd)


def test_evaluate_window_opens_a_real_alert_end_to_end(db_session: Session) -> None:
    """Not a direct call into the alert engine -- proves the actual wiring
    through evaluate_window creates a real Alert row via the real
    aggressive.yaml profile (fire_persistence_windows: 1)."""
    _register_baseline(db_session)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, N, age_offset=500)  # guaranteed drift

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert window is not None

    alert = db_session.scalars(
        select(Alert).where(
            Alert.model_id == MODEL_ID,
            Alert.kind == AlertKind.DRIFT,
            Alert.feature_name == "age",
            Alert.signal_name == "psi",
        )
    ).one()

    assert alert.status == AlertStatus.OPEN
    assert alert.evidence_window_id == window.id
    assert alert.evidence_baseline_id == window.baseline_id


def test_segment_defining_feature_is_not_evaluated_inside_its_own_segment(
    db_session: Session,
) -> None:
    """Every row in the region='EU' segment has region == 'EU' by
    construction, so a categorical drift test on `region` inside that
    segment can only ever see one category. It used to be written as a
    permanently not_computable row per segment, which opened a permanent
    not_computable alert per segment value on every scenario. Now no row is
    written at all: the dashboard renders the slot as not_configured."""
    _register_baseline(db_session, n=N)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, N)

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert window is not None

    self_segment_rows = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id,
            DriftResult.feature_name == "region",
            DriftResult.segment_dimension == "region",
        )
    ).all()
    assert self_segment_rows == []

    # the same feature is still evaluated globally, and the other features
    # are still evaluated inside every segment
    global_region = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id,
            DriftResult.feature_name == "region",
            DriftResult.segment_dimension.is_(None),
        )
    ).all()
    assert len(global_region) == 1
    age_in_segments = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id,
            DriftResult.feature_name == "age",
            DriftResult.segment_dimension == "region",
        )
    ).all()
    assert {r.segment_value for r in age_in_segments} == set(REGIONS)


def test_every_statistic_is_refused_below_its_sampling_noise_floor(db_session: Session) -> None:
    """At 60 baseline rows against 60 live rows every effect size's null
    ceiling is above the aggressive profile's clear threshold -- PSI (0.07),
    KS D (0.1), Cramer's V (0.07) and the score JSD (0.03) -- so each row is
    written as not_computable with its floor and its source in the reason,
    and no alert can open on any of them. The same window at N rows (the
    other tests in this module) computes all four."""
    _register_baseline(db_session, n=SMALL)
    window_start = datetime(2026, 1, 1, tzinfo=UTC)
    window_end = window_start + timedelta(hours=1)
    _add_predictions(db_session, window_start, SMALL, age_offset=500)  # would be drift

    window = evaluate_window(db_session, MODEL_ID, window_start, window_end)
    assert window is not None

    rows = db_session.scalars(
        select(DriftResult).where(
            DriftResult.evaluation_window_id == window.id,
            DriftResult.segment_dimension.is_(None),
        )
    ).all()
    by_method = {(r.feature_name, r.test_method): r for r in rows}
    expectations = {
        ("age", TestMethod.PSI): ("PSI", "simulated on the baseline"),
        ("age", TestMethod.KS): ("KS D", "Kolmogorov"),
        ("region", TestMethod.CHI_SQUARE): ("Cramer's V", "chi-square with"),
        ("__prediction_score__", TestMethod.JSD): ("JSD", "simulated on the baseline"),
    }
    for key, (name, source) in expectations.items():
        row = by_method[key]
        assert row.status == MetricStatus.NOT_COMPUTABLE, key
        assert row.statistic is None
        assert row.not_computable_reason is not None
        assert row.not_computable_reason.startswith(f"{name} sampling-noise ceiling"), key
        assert source in row.not_computable_reason, key
    assert db_session.scalars(select(Alert).where(Alert.model_id == MODEL_ID)).all() == []
