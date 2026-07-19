from collections.abc import MutableMapping
from datetime import date

import pandas as pd
import streamlit as st
from sqlalchemy.orm import Session

from driftwatch.config.loader import ModelConfigNotFoundError, load_model_config, load_profile
from driftwatch.config.schema import ModelConfig, Profile
from driftwatch.dashboard import charts, queries, state
from driftwatch.dashboard.db import read_only_session
from driftwatch.db.models import TestMethod
from driftwatch.durations import parse_duration
from driftwatch.evaluation.drift import PREDICTION_SCORE_FEATURE_NAME

st.set_page_config(page_title="driftwatch", layout="wide")

VIEWS = {
    "drift_timeline": "Drift timeline",
    "feature_attribution": "Feature attribution",
    "segment_view": "Segment view",
    "performance_timeline": "Performance over time",
    "alert_history": "Alert history",
}


def _enum_col(series: pd.Series) -> pd.Series:
    return series.map(lambda v: getattr(v, "value", v))


def _load_config(model_id: str) -> tuple[ModelConfig, Profile] | None:
    """Config files, not the database -- read-only regardless. Errors are
    re-raised with a message that names the model only, never the
    filesystem path the loader looked at (constraint: no internal paths
    rendered in the UI)."""
    try:
        model_config = load_model_config(model_id)
    except ModelConfigNotFoundError:
        st.error(f"No config found for model {model_id!r}.")
        return None
    profile = load_profile(model_config.profile)
    return model_config, profile


def _feature_options(model_config: ModelConfig) -> list[tuple[str, str]]:
    """(feature_name, dtype) pairs, including the prediction-score pseudo
    feature -- every signal the drift timeline / segment view can plot."""
    options: list[tuple[str, str]] = [(f.name, f.dtype) for f in model_config.schema_.features]
    options.append((PREDICTION_SCORE_FEATURE_NAME, "prediction_score"))
    return options


def _methods_for(dtype: str, profile: Profile) -> list[TestMethod]:
    if dtype == "continuous":
        return [TestMethod(m) for m in profile.drift_tests.continuous.methods]
    if dtype == "categorical":
        return [TestMethod(m) for m in profile.drift_tests.categorical.methods]
    return [TestMethod(profile.drift_tests.prediction_score.method)]


def _pick_feature_and_method(
    params: MutableMapping[str, str],
    model_config: ModelConfig,
    profile: Profile,
) -> tuple[str, str, TestMethod] | None:
    """Shared feature+method picker for drift timeline and segment view.
    Returns (feature_name, dtype, test_method), or None if the chosen
    feature's dtype has no configured test method to show."""
    features = _feature_options(model_config)
    feature_names = [f[0] for f in features]
    default_feature = state.get_str(params, "feature", feature_names[0])
    if default_feature not in feature_names:
        default_feature = feature_names[0]
    feature_name = st.selectbox(
        "Feature", feature_names, index=feature_names.index(default_feature)
    )
    state.set_str(params, "feature", feature_name)

    dtype = dict(features)[feature_name]
    methods = _methods_for(dtype, profile)
    if not methods:
        st.warning(f"No drift test is configured for {feature_name!r} in this profile.")
        return None
    method_values = [m.value for m in methods]
    default_method = state.get_str(params, "test_method", method_values[0])
    if default_method not in method_values:
        default_method = method_values[0]
    test_method = TestMethod(
        st.selectbox("Test method", method_values, index=method_values.index(default_method))
    )
    state.set_str(params, "test_method", test_method.value)
    return feature_name, dtype, test_method


def _render_drift_timeline(
    session: Session,
    params: MutableMapping[str, str],
    model_id: str,
    model_config: ModelConfig,
    profile: Profile,
    start: date,
    end: date,
) -> None:
    picked = _pick_feature_and_method(params, model_config, profile)
    if picked is None:
        return
    feature_name, _dtype, test_method = picked

    segment_dimension = None
    segment_value = None
    dimensions = model_config.segments.dimensions
    if dimensions:
        scope_options = ["Global", *dimensions]
        default_scope = state.get_str(params, "scope", "Global")
        if default_scope not in scope_options:
            default_scope = "Global"
        scope = st.selectbox("Scope", scope_options, index=scope_options.index(default_scope))
        state.set_str(params, "scope", scope)
        if scope != "Global":
            segment_dimension = scope
            known_values = queries.known_segment_values(
                session, model_id, feature_name, test_method, segment_dimension
            )
            if not known_values:
                st.info(f"No segment values seen yet for {segment_dimension!r}.")
                return
            default_value = state.get_str(params, "segment_value", known_values[0])
            if default_value not in known_values:
                default_value = known_values[0]
            segment_value = st.selectbox(
                "Segment value", known_values, index=known_values.index(default_value)
            )
            state.set_str(params, "segment_value", segment_value)

    start_dt, end_dt = queries.date_range_to_datetimes(start, end)
    window_duration = parse_duration(profile.evaluation.window)
    df = queries.fetch_drift_timeline(
        session,
        model_id,
        feature_name,
        test_method,
        segment_dimension,
        segment_value,
        start_dt,
        end_dt,
        window_duration,
    )
    if df.empty:
        st.info("No evaluated windows in this range.")
        return
    fire_threshold, clear_threshold = queries.drift_thresholds(profile, test_method)
    title = f"{feature_name} / {test_method.value}"
    if segment_value:
        title += f" ({segment_dimension}={segment_value})"
    chart = charts.drift_timeline_chart(df, fire_threshold, clear_threshold, title=title)
    st.altair_chart(chart, width="content")


def _render_feature_attribution(
    session: Session,
    params: MutableMapping[str, str],
    model_id: str,
    model_config: ModelConfig,
    profile: Profile,
    start: date,
    end: date,
) -> None:
    start_dt, end_dt = queries.date_range_to_datetimes(start, end)
    windows_df = queries.list_windows(session, model_id, start_dt, end_dt)
    if windows_df.empty:
        st.info("No evaluated windows in this range.")
        return
    windows_df = windows_df.sort_values("window_end").reset_index(drop=True)
    labels = [ts.isoformat() for ts in windows_df["window_end"]]
    window_ids = list(windows_df["window_id"])
    default_window_id = state.get_optional_int(params, "window_id")
    default_index = len(labels) - 1
    if default_window_id is not None and default_window_id in window_ids:
        default_index = window_ids.index(default_window_id)
    chosen_label = st.selectbox("Window (by end time)", labels, index=default_index)
    window_id = int(window_ids[labels.index(chosen_label)])
    state.set_str(params, "window_id", str(window_id))

    df = queries.fetch_feature_attribution(session, model_config, profile, window_id)
    chart = charts.feature_attribution_chart(df)
    st.altair_chart(chart, width="content")


def _render_segment_view(
    session: Session,
    params: MutableMapping[str, str],
    model_id: str,
    model_config: ModelConfig,
    profile: Profile,
    start: date,
    end: date,
) -> None:
    dimensions = model_config.segments.dimensions
    if not dimensions:
        st.info("This model has no configured segment dimensions.")
        return
    default_dim = state.get_str(params, "segment_dimension", dimensions[0])
    if default_dim not in dimensions:
        default_dim = dimensions[0]
    segment_dimension = st.selectbox(
        "Segment dimension", dimensions, index=dimensions.index(default_dim)
    )
    state.set_str(params, "segment_dimension", segment_dimension)

    picked = _pick_feature_and_method(params, model_config, profile)
    if picked is None:
        return
    feature_name, _dtype, test_method = picked

    start_dt, end_dt = queries.date_range_to_datetimes(start, end)
    window_duration = parse_duration(profile.evaluation.window)
    df = queries.fetch_segment_view(
        session,
        model_id,
        feature_name,
        test_method,
        segment_dimension,
        start_dt,
        end_dt,
        window_duration,
    )
    if df.empty:
        st.info("No evaluated windows in this range.")
        return
    fire_threshold, clear_threshold = queries.drift_thresholds(profile, test_method)
    segments = sorted(df["segment"].unique(), key=lambda s: (s != "Global", s))
    chart = charts.segment_view_chart(df, fire_threshold, clear_threshold, segments)
    st.altair_chart(chart, width="content")


def _render_performance_timeline(
    session: Session,
    params: MutableMapping[str, str],
    model_id: str,
    model_config: ModelConfig,
    profile: Profile,
    start: date,
    end: date,
) -> None:
    metric_names = [spec.name for spec in profile.performance_metrics]
    if not metric_names:
        st.info("This profile configures no performance metrics.")
        return
    default_metric = state.get_str(params, "metric_name", metric_names[0])
    if default_metric not in metric_names:
        default_metric = metric_names[0]
    metric_name = st.selectbox("Metric", metric_names, index=metric_names.index(default_metric))
    state.set_str(params, "metric_name", metric_name)

    spec = next(s for s in profile.performance_metrics if s.name == metric_name)
    if spec.alert_direction is None or spec.fire_threshold is None or spec.clear_threshold is None:
        st.info(
            f"{metric_name!r} is informational only in this profile "
            "(no alert thresholds configured)."
        )
        return

    start_dt, end_dt = queries.date_range_to_datetimes(start, end)
    window_duration = parse_duration(profile.evaluation.window)
    df = queries.fetch_performance_timeline(
        session, model_id, metric_name, None, None, start_dt, end_dt, window_duration
    )
    if df.empty:
        st.info("No evaluated windows in this range.")
        return
    chart = charts.performance_timeline_chart(
        df, spec.fire_threshold, spec.clear_threshold, spec.alert_direction, title=metric_name
    )
    st.altair_chart(chart, width="content")


def _render_alert_history(session: Session, model_id: str, start: date, end: date) -> None:
    start_dt, end_dt = queries.date_range_to_datetimes(start, end)

    st.subheader("Fired / escalated / resolved")
    alerts_df = queries.fetch_alert_history(session, model_id, start_dt, end_dt)
    if alerts_df.empty:
        st.info("No alerts touched in this range.")
    else:
        display = alerts_df.copy()
        display["kind"] = _enum_col(display["kind"])
        display["status"] = _enum_col(display["status"])
        st.dataframe(display, width="stretch", hide_index=True)

    st.subheader("Suppressed (never reached persistence)")
    loaded = _load_config(model_id)
    if loaded is None:
        return
    model_config, profile = loaded
    suppressed_df = queries.fetch_suppressed_episodes(
        session, model_id, model_config, profile, start_dt, end_dt
    )
    if suppressed_df.empty:
        st.info("No suppressed episodes in this range.")
    else:
        st.dataframe(suppressed_df, width="stretch", hide_index=True)


def main() -> None:
    st.title("driftwatch")
    st.caption(
        "Read-only monitoring dashboard. No evaluations, alerts, or config are changed here."
    )

    params = st.query_params

    with read_only_session() as session:
        models = queries.list_models(session)
        if not models:
            st.info("No models with evaluated windows yet.")
            return

        default_model = state.get_str(params, "model", models[0])
        if default_model not in models:
            default_model = models[0]
        model_id = st.sidebar.selectbox("Model", models, index=models.index(default_model))
        state.set_str(params, "model", model_id)

        default_range = queries.default_time_range(session, model_id)
        if default_range is None:
            st.info(f"Model {model_id!r} has no evaluated windows yet.")
            return

        start, end = state.resolve_time_range(params, default_range)
        picked_start = st.sidebar.date_input(
            "Start", value=start, min_value=default_range[0], max_value=default_range[1]
        )
        picked_end = st.sidebar.date_input(
            "End", value=end, min_value=default_range[0], max_value=default_range[1]
        )
        if isinstance(picked_start, date):
            start = picked_start
        if isinstance(picked_end, date):
            end = picked_end
        state.set_str(params, "start", state.format_date_param(start))
        state.set_str(params, "end", state.format_date_param(end))

        view_labels = list(VIEWS.values())
        view_keys = list(VIEWS.keys())
        default_view = state.get_str(params, "view", view_keys[0])
        if default_view not in view_keys:
            default_view = view_keys[0]
        chosen_label = st.sidebar.radio("View", view_labels, index=view_keys.index(default_view))
        view = view_keys[view_labels.index(chosen_label)]
        state.set_str(params, "view", view)

        st.header(VIEWS[view])

        if view == "alert_history":
            _render_alert_history(session, model_id, start, end)
            return

        loaded = _load_config(model_id)
        if loaded is None:
            return
        model_config, profile = loaded

        if view == "drift_timeline":
            _render_drift_timeline(session, params, model_id, model_config, profile, start, end)
        elif view == "feature_attribution":
            _render_feature_attribution(
                session, params, model_id, model_config, profile, start, end
            )
        elif view == "segment_view":
            _render_segment_view(session, params, model_id, model_config, profile, start, end)
        elif view == "performance_timeline":
            _render_performance_timeline(
                session, params, model_id, model_config, profile, start, end
            )


main()
