"""Chart builders. Every function here is pure: DataFrame + parameters in,
an alt.Chart out -- no Streamlit calls, no randomness, no wall-clock reads.
Fixed widths/heights, an explicit sorted color domain (see
driftwatch.dashboard.colors), and a fixed y-domain computed once from the
data handed in (never re-derived per facet) are what make "same data
produces a pixel-identical chart on every run" true, including for the
faceted segment view where every facet must share one axis to be visually
comparable.

Rendering convention for the three drift/performance states: a 'computed'
value plots at its real y; 'not_computable' and 'not_configured' have no
numeric value, so each gets its own fixed lane BELOW a dashed zero-line,
distinctly colored and shaped, with the reason (if any) in the tooltip --
never a silent gap in the line.
"""

from collections.abc import Iterable
from typing import Any

import altair as alt
import pandas as pd

from driftwatch.dashboard.colors import stable_color

# Altair's fluent chaining API (.properties/.facet/.configure_*/.encode on an
# already-composed chart) isn't stub-precise about which of Chart/LayerChart/
# FacetChart each step returns, and several steps resolve to Any in Altair's
# own stubs. Every builder below is typed Any at the boundary rather than
# fighting that with casts -- the real contract ("takes a DataFrame, returns
# something st.altair_chart can render") is documented, not encoded in the
# type.
AltChart = Any

FIXED_WIDTH = 560
FIXED_HEIGHT = 240
FACET_WIDTH = 190
FACET_HEIGHT = 220
BAR_ROW_HEIGHT = 26

_STATE_COLORS = {
    "breach": "#E45756",
    "normal": "#4C78A8",
    "not_computable": "#F58518",
    "not_configured": "#BAB0AC",
    "missing": "#000000",
}
_STATE_LABELS = {
    "breach": "breach",
    "normal": "computed",
    "not_computable": "not computable",
    "not_configured": "not configured",
    "missing": "not evaluated",
}
_STATE_DOMAIN = ["breach", "normal", "not_computable", "not_configured", "missing"]
_STATE_SHAPES = {
    "breach": "circle",
    "normal": "circle",
    "not_computable": "triangle-up",
    "not_configured": "diamond",
    "missing": "cross",
}
"""'missing' (no EvaluationWindow at all -- the window was never
evaluated, e.g. scheduler downtime or a backfill gap) is deliberately the
most visually alarming state here: black, cross-shaped, unmistakable from
'not_configured' (a grey diamond -- the window WAS evaluated, just not
this particular signal). Confusing the two is the most dangerous mistake
a monitoring dashboard can make -- a gap in monitoring itself must never
look like confirmed-quiet data."""


def _state_color_scale() -> alt.Scale:
    return alt.Scale(domain=_STATE_DOMAIN, range=[_STATE_COLORS[s] for s in _STATE_DOMAIN])


def _state_shape_scale() -> alt.Scale:
    return alt.Scale(domain=_STATE_DOMAIN, range=[_STATE_SHAPES[s] for s in _STATE_DOMAIN])


def _utc_x(field: str, **kwargs: object) -> alt.X:
    """Every window_start/window_end value in this project is UTC (see
    driftwatch.db.models -- all DateTime columns are timezone(True), always
    written via datetime.now(UTC) or UTC-aware inputs). Vega-Lite's default
    temporal scale renders axis ticks and positions in the VIEWER'S browser
    timezone, not the data's own timezone -- left unset, the exact same
    chart would render with different x-axis tick labels (and potentially
    different pixel positions for points near a DST boundary) depending on
    where it's viewed, which breaks "same data produces a pixel-identical
    chart on every run" (this project's own determinism requirement) and
    would misrepresent when a window actually was. scale=utc pins it."""
    return alt.X(field, scale=alt.Scale(type="utc"), **kwargs)  # type: ignore[arg-type]


def _utc_tooltip(field: str, **kwargs: object) -> alt.Tooltip:
    """See _utc_x -- tooltips format temporal values in local time by
    default too; the "utc:" format-string prefix (D3/Vega-Lite's own
    convention for this, not a separate formatType) pins it to UTC."""
    return alt.Tooltip(field, format="utc:%Y-%m-%d %H:%M", **kwargs)  # type: ignore[arg-type]


def _breach(value: float, fire_threshold: float, direction: str) -> bool:
    return value >= fire_threshold if direction == "above" else value <= fire_threshold


def _with_plot_columns(
    df: pd.DataFrame,
    value_col: str,
    fire_threshold: float,
    clear_threshold: float,
    direction: str = "above",
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Adds `state_detail` (breach/normal/not_computable/not_configured/
    missing, used for color+shape) and `plot_y` (the real value for
    computed rows, a fixed sentinel lane below a y=0 rule for the other
    three states). Returns the augmented frame and the (y_min, y_max)
    domain to pin on every chart built from it -- computed once, here,
    from the full frame handed in (the caller passes the FULL multi-facet
    frame for the segment view, not a per-facet slice), so faceted charts
    share one comparable scale."""
    df = df.copy()
    computed_values = df.loc[df["state"] == "computed", value_col].dropna()
    candidates = [fire_threshold, clear_threshold]
    if not computed_values.empty:
        candidates.append(float(computed_values.max()))
    y_top = max(candidates) * 1.15
    lane_not_computable = -0.12 * y_top
    lane_not_configured = -0.24 * y_top
    lane_missing = -0.36 * y_top
    y_bottom = -0.44 * y_top

    def state_detail(row: pd.Series) -> str:
        if row["state"] != "computed":
            return str(row["state"])
        value = row[value_col]
        if pd.isna(value):
            return "normal"
        return "breach" if _breach(float(value), fire_threshold, direction) else "normal"

    def plot_y(row: pd.Series) -> float:
        if row["state"] == "computed" and not pd.isna(row[value_col]):
            return float(row[value_col])
        if row["state"] == "not_computable":
            return lane_not_computable
        if row["state"] == "not_configured":
            return lane_not_configured
        return lane_missing

    df["state_detail"] = df.apply(state_detail, axis=1)
    df["plot_y"] = df.apply(plot_y, axis=1)
    return df, (y_bottom, y_top)


def _finalize(chart: AltChart) -> AltChart:
    """Light theme, legible at ~600px: white background, dark-on-light
    axis/legend/title text, thin neutral gridlines, no view border."""
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#333333", titleColor="#333333", gridColor="#eeeeee", domainColor="#999999"
        )
        .configure_legend(labelColor="#333333", titleColor="#333333")
        .configure_title(color="#111111")
    )


def _threshold_rules(
    fire_threshold: float, clear_threshold: float, x_max: pd.Timestamp
) -> AltChart:
    """Reference lines for fire/clear, labelled with their value. The
    label's x position is DATA-driven (pinned to the rightmost timestamp
    actually in view, via the same temporal scale as everything else in
    the chart) rather than a raw pixel offset -- Vega-Lite's declared
    `width` is the plot's nominal size, not its actual post-axis-label
    drawable width, so a hardcoded "width - N" pixel position is not
    reliably inside the plot and was clipping the label text."""
    lines = pd.DataFrame(
        [
            {"y": fire_threshold, "x": x_max, "label": f"fire {fire_threshold:g}", "kind": "fire"},
            {
                "y": clear_threshold,
                "x": x_max,
                "label": f"clear {clear_threshold:g}",
                "kind": "clear",
            },
        ]
    )
    color = alt.Color(
        "kind:N",
        scale=alt.Scale(domain=["fire", "clear"], range=[_STATE_COLORS["breach"], "#888888"]),
        legend=None,
    )
    rule = (
        alt.Chart(lines)
        .mark_rule(strokeDash=[4, 3])
        .encode(y="y:Q", color=color, strokeWidth=alt.value(1.5))
    )
    # Fixed opposite-side offsets (label above its line for fire, below for
    # clear) rather than one shared dy -- when the two thresholds are close
    # together relative to the y-domain (as when observed statistics run
    # far past the fire line), a single shared offset puts both labels at
    # nearly the same pixel position and they overlap into unreadable text.
    fire_text = (
        alt.Chart(lines[lines["kind"] == "fire"])
        .mark_text(align="right", dx=-4, dy=-6, fontSize=10)
        .encode(x=_utc_x("x:T"), y="y:Q", text="label:N", color=color)
    )
    clear_text = (
        alt.Chart(lines[lines["kind"] == "clear"])
        .mark_text(align="right", dx=-4, dy=12, fontSize=10)
        .encode(x=_utc_x("x:T"), y="y:Q", text="label:N", color=color)
    )
    return alt.layer(rule, fire_text, clear_text)


def _zero_rule() -> AltChart:
    return (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(strokeDash=[1, 2], color="#bbbbbb")
        .encode(y="y:Q")
    )


def _timeline_layer(
    df: pd.DataFrame,
    x_field: str,
    tooltip: list[alt.Tooltip],
    point_size: int = 60,
    *,
    attach_data: bool = True,
) -> AltChart:
    """attach_data=False is required inside a faceted chart: a layer that
    carries its own explicit dataframe is NOT filtered per facet by
    Vega-Lite, so every facet cell would render every segment's points
    overlaid on top of each other instead of just its own. Passing no data
    here lets the layer inherit the already-faceted subset from the parent
    facet operator's own `data=` argument instead."""
    base = alt.Chart(df) if attach_data else alt.Chart()
    line = (
        base.transform_filter(alt.datum.state == "computed")
        .mark_line(color=_STATE_COLORS["normal"], strokeWidth=1.5)
        .encode(x=_utc_x(f"{x_field}:T"), y=alt.Y("plot_y:Q", title=None))
    )
    points = base.mark_point(filled=True, size=point_size, opacity=0.9).encode(
        x=_utc_x(f"{x_field}:T", title="window"),
        y=alt.Y("plot_y:Q", title="value"),
        color=alt.Color(
            "state_detail:N",
            scale=_state_color_scale(),
            title="state",
            legend=alt.Legend(
                labelExpr=" ".join(
                    f"datum.value == '{k}' ? '{v}' :" for k, v in _STATE_LABELS.items()
                )
                + " ''"
            ),
        ),
        shape=alt.Shape("state_detail:N", scale=_state_shape_scale(), legend=None),
        tooltip=tooltip,
    )
    return alt.layer(line, points)


def drift_timeline_chart(
    df: pd.DataFrame,
    fire_threshold: float,
    clear_threshold: float,
    *,
    width: int = FIXED_WIDTH,
    height: int = FIXED_HEIGHT,
    title: str = "",
) -> AltChart:
    """One feature/test_method signal over time: computed values as a
    connected line, threshold reference lines labelled with their value,
    breach points highlighted, not_computable/not_configured rendered in
    their own lane below the zero line with the reason on hover."""
    plotted, (y_min, y_max) = _with_plot_columns(df, "value", fire_threshold, clear_threshold)
    tooltip = [
        _utc_tooltip("window_end:T", title="window"),
        alt.Tooltip("state:N", title="state"),
        alt.Tooltip("value:Q", title="value", format=".4f"),
        alt.Tooltip("reason:N", title="reason"),
    ]
    layers = alt.layer(
        _zero_rule(),
        _threshold_rules(fire_threshold, clear_threshold, plotted["window_end"].max()),
        _timeline_layer(plotted, "window_end", tooltip),
    ).resolve_scale(y="shared", color="independent", shape="independent")
    chart = layers.properties(width=width, height=height, title=title, background="white").encode(
        y=alt.Y(scale=alt.Scale(domain=[y_min, y_max]))
    )
    return _finalize(chart)


def segment_view_chart(
    df: pd.DataFrame,
    fire_threshold: float,
    clear_threshold: float,
    segments: Iterable[str],
    *,
    width: int = FACET_WIDTH,
    height: int = FACET_HEIGHT,
) -> AltChart:
    """Global and every segment value side by side in one frame, sharing
    one y-domain (computed once over the whole `df`, not per facet) so a
    passing global aggregate and a breaching segment are directly
    comparable at a glance."""
    plotted, (y_min, y_max) = _with_plot_columns(df, "value", fire_threshold, clear_threshold)
    tooltip = [
        alt.Tooltip("segment:N", title="segment"),
        _utc_tooltip("window_end:T", title="window"),
        alt.Tooltip("state:N", title="state"),
        alt.Tooltip("value:Q", title="value", format=".4f"),
        alt.Tooltip("reason:N", title="reason"),
    ]
    layers = (
        alt.layer(
            _zero_rule(),
            _threshold_rules(fire_threshold, clear_threshold, plotted["window_end"].max()),
            _timeline_layer(plotted, "window_end", tooltip, point_size=40, attach_data=False),
        )
        .resolve_scale(color="independent", shape="independent")
        .encode(y=alt.Y(scale=alt.Scale(domain=[y_min, y_max])))
    )
    # Explicit data= here: the zero/threshold-rule layers carry their own
    # small constant dataframes (repeated identically in every facet cell,
    # which is exactly what a reference line should do), so Altair can't
    # infer one shared top-level dataset for the whole layered spec on its
    # own -- facet() needs `plotted` named explicitly instead. The data
    # layer itself (_timeline_layer) is built with attach_data=False above
    # for exactly the same reason in reverse: it must NOT carry its own
    # dataframe, so it inherits the correctly-faceted per-segment subset
    # from this facet() call instead of re-rendering all segments' points
    # in every cell.
    faceted = layers.properties(width=width, height=height).facet(
        data=plotted, column=alt.Column("segment:N", sort=list(segments), title=None)
    )
    return _finalize(faceted)


def feature_attribution_chart(
    df: pd.DataFrame, *, width: int = FIXED_WIDTH, row_height: int = BAR_ROW_HEIGHT
) -> AltChart:
    """Ranked per-feature/test_method drift for one window, on a common
    'statistic / fire_threshold' axis so PSI, KS, Cramer's V, and JSD --
    each on its own native scale -- are directly comparable as 'how close
    to its own breach line'. not_computable/not_configured signals still
    get a row (a fixed small negative bar with the reason on hover), never
    silently omitted from the ranking."""
    plotted = df.copy()
    plotted["signal"] = plotted["feature_name"] + " / " + plotted["test_method"].astype(str)

    def state_detail(row: pd.Series) -> str:
        if row["state"] != "computed":
            return str(row["state"])
        return "breach" if bool(row.get("is_significant")) else "normal"

    plotted["state_detail"] = plotted.apply(state_detail, axis=1)
    plotted["plot_ratio"] = plotted.apply(
        lambda r: r["ratio"] if r["state"] == "computed" and not pd.isna(r["ratio"]) else -0.15,
        axis=1,
    )
    order = plotted.sort_values("ratio", ascending=False, na_position="last")["signal"].tolist()
    height = max(row_height * len(plotted), row_height * 2)

    tooltip = [
        alt.Tooltip("signal:N", title="signal"),
        alt.Tooltip("state:N", title="state"),
        alt.Tooltip("value:Q", title="value", format=".4f"),
        alt.Tooltip("fire_threshold:Q", title="fire threshold", format=".4f"),
        alt.Tooltip("reason:N", title="reason"),
    ]
    base = alt.Chart(plotted)
    bars = base.mark_bar().encode(
        x=alt.X("plot_ratio:Q", title="value / fire threshold"),
        y=alt.Y("signal:N", sort=order, title=None),
        color=alt.Color("state_detail:N", scale=_state_color_scale(), title="state"),
        tooltip=tooltip,
    )
    fire_rule = (
        alt.Chart(pd.DataFrame({"x": [1.0]}))
        .mark_rule(strokeDash=[4, 3], color=_STATE_COLORS["breach"])
        .encode(x="x:Q")
    )
    chart = alt.layer(bars, fire_rule).properties(width=width, height=height, background="white")
    return _finalize(chart)


def performance_timeline_chart(
    df: pd.DataFrame,
    fire_threshold: float,
    clear_threshold: float,
    direction: str,
    *,
    width: int = FIXED_WIDTH,
    height: int = FIXED_HEIGHT,
    title: str = "",
) -> AltChart:
    """Performance metric over time, split into 'initial' (the first
    computed value for a window, before any labels backfilled after it)
    and 'latest' (after however many retroactive recomputes since) --
    dashed vs solid stroke on the same axis, so a window whose metric
    changed after label backfill is visible as the two lines diverging at
    that window, not just as a hidden revision count."""
    plotted, (y_min, y_max) = _with_plot_columns(
        df, "value", fire_threshold, clear_threshold, direction
    )
    tooltip = [
        alt.Tooltip("revision:N", title="revision"),
        _utc_tooltip("window_end:T", title="window"),
        alt.Tooltip("state:N", title="state"),
        alt.Tooltip("value:Q", title="value", format=".4f"),
        alt.Tooltip("reason:N", title="reason"),
    ]
    base = alt.Chart(plotted)
    lines = (
        base.transform_filter(alt.datum.state == "computed")
        .mark_line(strokeWidth=1.5)
        .encode(
            x=_utc_x("window_end:T", title="window"),
            y=alt.Y("plot_y:Q", title="value"),
            color=alt.value(_STATE_COLORS["normal"]),
            strokeDash=alt.StrokeDash(
                "revision:N",
                scale=alt.Scale(domain=["initial", "latest"], range=[[4, 3], [1, 0]]),
                title="revision",
            ),
        )
    )
    points = base.mark_point(filled=True, size=55, opacity=0.9).encode(
        x=_utc_x("window_end:T"),
        y=alt.Y("plot_y:Q"),
        color=alt.Color("state_detail:N", scale=_state_color_scale(), title="state"),
        shape=alt.Shape("state_detail:N", scale=_state_shape_scale(), legend=None),
        tooltip=tooltip,
    )
    layers = (
        alt.layer(
            _zero_rule(),
            _threshold_rules(fire_threshold, clear_threshold, plotted["window_end"].max()),
            lines,
            points,
        )
        .resolve_scale(color="independent", shape="independent")
        .encode(y=alt.Y(scale=alt.Scale(domain=[y_min, y_max])))
    )
    chart = layers.properties(width=width, height=height, title=title, background="white")
    return _finalize(chart)


__all__ = [
    "drift_timeline_chart",
    "segment_view_chart",
    "feature_attribution_chart",
    "performance_timeline_chart",
    "stable_color",
]
