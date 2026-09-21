"""Case-study figures for DriftWatch, drawn from the repository's own data.

    uv run --with matplotlib python figures/_src/figures.py

Every plotted number is computed here from the pure scenario generator and
the repository's own statistics (`driftwatch.demo.generator`,
`driftwatch.stats`), the same code the evaluation pipeline runs, so the
segment, alert and performance charts are the scenarios' actual per-window
traces and not drawings of them. The three performance numbers the metrics
card quotes are recomputed the same way and checked, at render time,
against the values `driftwatch demo-verify concept_drift` reported from the
database; those pinned values are the only typed numbers in this file. The
test count is read from `pytest --collect-only` at render time.

Written with the portfolio's figure kit (`theme.py`, copied from
`05 FIGURE SYSTEM.md`): PNG at 200 DPI, one emphasised block per diagram,
arrows above nodes, no em dashes and no arrow glyphs in figure text.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections import defaultdict
from dataclasses import dataclass

os.environ.setdefault("CONFIGS_DIR", "configs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from theme import (  # noqa: E402
    BAND,
    BAND2,
    BLUE_E,
    BLUE_T,
    BORDER_D,
    C_GRAY,
    CARD,
    GREEN,
    GREEN_E,
    GREEN_SOFT,
    GREEN_T,
    HERO_ASPECT,
    INK,
    MUTE,
    PAPER,
    RED,
    SAND_E,
    SAND_T,
    SURFACE,
    VOLT,
    VOLT_DEEP,
    VOLT_EDGE,
    VOLT_SOFT,
    VOLT_TXT,
    Z_ARROW,
    Z_BAND,
    Z_LABEL,
    Z_NODE,
    Z_NODE_TEXT,
    Z_PANEL,
    alabel,
    arrow,
    canvas,
    oarrow,
    rbox,
    safe_box,
    save,
    title_block,
    track,
    use_theme,
)

from driftwatch.config.loader import load_profile  # noqa: E402
from driftwatch.demo.generator import generate_scenario_data  # noqa: E402
from driftwatch.demo.loader import load_scenario  # noqa: E402
from driftwatch.metrics.binary import pr_auc  # noqa: E402
from driftwatch.stats.binning import compute_continuous_edges  # noqa: E402
from driftwatch.stats.psi import psi  # noqa: E402

use_theme()
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)

KICKER = "ML MONITORING : DELAYED GROUND TRUTH"
TAGLINE = "Catch decay before a stakeholder does."
BANNER_TAGLINE = "Separating data drift from performance decay, and alerting only on what persists"

# ---- numbers that come from the database, pinned with their provenance ----
# `driftwatch demo-verify concept_drift`, quoted verbatim in README.md. The
# concept trace below recomputes all three from the generator and refuses to
# draw if they disagree, so the figures and the verifier can never drift apart.
#   PR-AUC healthy before label_noise, degraded after: early_mean=0.874, late_mean=0.387
#   degradation only visible after backfill: 60 windows initially not_computable
VERIFIED_PR_AUC_BEFORE, VERIFIED_PR_AUC_AFTER, VERIFIED_NOT_COMPUTABLE = 0.874, 0.387, 60
# figures/_src/sweeps.py, `clean`: forty clean seeds at the shipped segment geometry
CLEAN_SEEDS, CLEAN_SPURIOUS_OPENS = 40, 0
# patient.yaml, the profile every shipped scenario runs under
PROFILE = load_profile("patient")
PSI_FIRE = PROFILE.drift_tests.continuous.psi_threshold
PSI_CLEAR = PROFILE.drift_tests.continuous.psi_clear_threshold
FIRE_PERSISTENCE = PROFILE.alerting.fire_persistence_windows
ESCALATE_PERSISTENCE = PROFILE.alerting.escalate_persistence_windows
RESOLVE_PERSISTENCE = PROFILE.alerting.resolve_persistence_windows
MIN_LABELS = PROFILE.evaluation.min_window_size
PR_AUC_SPEC = next(m for m in PROFILE.performance_metrics if m.name == "pr_auc")
PR_AUC_FIRE, PR_AUC_CLEAR = PR_AUC_SPEC.fire_threshold, PR_AUC_SPEC.clear_threshold
assert PR_AUC_FIRE is not None and PR_AUC_CLEAR is not None


def P(name: str) -> str:
    return os.path.join(OUT, name)


def test_counts() -> tuple[int, int]:
    """(tests collected, test files) from pytest itself, so the card never
    carries a typed count."""
    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    n_tests = sum(1 for line in collected.splitlines() if "::" in line)
    n_files = len({line.split("::")[0] for line in collected.splitlines() if "::" in line})
    return n_tests, n_files


# =====================================================================
# the segment scenario, recomputed window by window from the generator
# =====================================================================
@dataclass(frozen=True)
class SegmentTrace:
    windows: list[int]
    global_psi: list[float]
    segment_psi: list[float]
    shift_windows: tuple[int, int]  # [start, end) of the injected event
    share: float
    predictions_per_window: int
    segment_rows: float  # mean live rows per window in the segment


def segment_trace(feature: str = "age", segment: str = "APAC") -> SegmentTrace:
    scenario = load_scenario("segment_isolated")
    data = generate_scenario_data(scenario)
    t0 = scenario.start_date
    base_all = [
        r.features[feature] for r in data.baseline_records if r.features.get(feature) is not None
    ]
    edges = compute_continuous_edges(base_all)
    base_seg = [
        r.features[feature]
        for r in data.baseline_records
        if r.segment_values.get("region") == segment and r.features.get(feature) is not None
    ]
    by_window: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"all": [], "seg": []})
    for p in data.predictions:
        v = p.features.get(feature)
        if v is None:
            continue
        w = int((p.predicted_at - t0).total_seconds() // 3600)
        by_window[w]["all"].append(v)
        if p.segment_values.get("region") == segment:
            by_window[w]["seg"].append(v)
    windows = sorted(by_window)
    g = [psi(base_all, by_window[w]["all"], edges).value or 0.0 for w in windows]
    s = [psi(base_seg, by_window[w]["seg"], edges).value or 0.0 for w in windows]
    event = scenario.events[0]
    region = next(f for f in scenario.features if f.name == "region")
    share = region.weights[region.categories.index(segment)]
    return SegmentTrace(
        windows=windows,
        global_psi=g,
        segment_psi=s,
        shift_windows=(event.start_window, event.start_window + (event.duration_windows or 0)),
        share=share,
        predictions_per_window=scenario.predictions_per_window,
        segment_rows=float(np.mean([len(by_window[w]["seg"]) for w in windows])),
    )


@dataclass(frozen=True)
class Lifecycle:
    opened: int | None
    escalated: int | None
    resolved: int | None
    state: list[str]      # per window: quiet, counting, open, escalated, resolved
    breach_run: list[int]  # consecutive breaching windows ending here


def lifecycle(
    values: list[float],
    fire: float = PSI_FIRE,
    clear: float = PSI_CLEAR,
    direction: str = "above",
) -> Lifecycle:
    """Where the patient profile's state machine opens, escalates and
    resolves on this trace: the same three-way classification and streak
    rule the engine uses (`alerting/streaks.py`). A value in the dead zone
    between the two thresholds breaks whichever streak was building."""
    opened = escalated = resolved = None
    run_b = run_c = 0
    status = None
    states: list[str] = []
    runs: list[int] = []
    for i, v in enumerate(values):
        if direction == "above":
            kind = "B" if v >= fire else ("c" if v <= clear else "d")
        else:
            kind = "B" if v <= fire else ("c" if v >= clear else "d")
        run_b = run_b + 1 if kind == "B" else 0
        run_c = run_c + 1 if kind == "c" else 0
        if status is None and run_b >= FIRE_PERSISTENCE:
            status, opened = "open", i
        elif status == "open" and run_b >= ESCALATE_PERSISTENCE:
            status, escalated = "escalated", i
        elif status in ("open", "escalated") and run_c >= RESOLVE_PERSISTENCE:
            status, resolved = "resolved", i
        if status is None:
            states.append("counting" if run_b else "quiet")
        else:
            states.append(status)
        runs.append(run_b)
    return Lifecycle(opened, escalated, resolved, states, runs)


# =====================================================================
# the covariate scenario: one global feature, a one-window blip and then
# a sustained shift, recomputed window by window from the generator
# =====================================================================
@dataclass(frozen=True)
class CovariateTrace:
    windows: list[int]
    psi: list[float]
    blip_windows: tuple[int, int]  # [start, end) of the one-window event
    shift_start: int  # the sustained event runs to the end of the scenario
    predictions_per_window: int


def covariate_trace(feature: str = "age") -> CovariateTrace:
    scenario = load_scenario("covariate_shift")
    data = generate_scenario_data(scenario)
    t0 = scenario.start_date
    base = [
        r.features[feature] for r in data.baseline_records if r.features.get(feature) is not None
    ]
    edges = compute_continuous_edges(base)
    by_window: dict[int, list[float]] = defaultdict(list)
    for p in data.predictions:
        v = p.features.get(feature)
        if v is None:
            continue
        by_window[int((p.predicted_at - t0).total_seconds() // 3600)].append(v)
    windows = sorted(by_window)
    values = [psi(base, by_window[w], edges).value or 0.0 for w in windows]
    events = [e for e in scenario.events if e.feature == feature]
    blip = next(e for e in events if e.duration_windows)
    shift = next(e for e in events if not e.duration_windows)
    return CovariateTrace(
        windows=windows,
        psi=values,
        blip_windows=(blip.start_window, blip.start_window + blip.duration_windows),
        shift_start=shift.start_window,
        predictions_per_window=scenario.predictions_per_window,
    )


# =====================================================================
# the concept scenario: PR-AUC per window once every label is in, and how
# many labels each window held when it was first evaluated
# =====================================================================
@dataclass(frozen=True)
class ConceptTrace:
    windows: list[int]
    pr_auc: list[float]  # the latest revision, every label present
    labels_at_first: list[int]  # labels landed within early_cutoff_hours of prediction
    predictions_per_window: int
    early_cutoff_hours: float
    event_start: int
    flip_rate: tuple[float, float]  # label noise rate before and during the event
    arrived_by: dict[int, float]  # share of all labels landed within h hours

    @property
    def before(self) -> float:
        return float(np.mean(self.pr_auc[: self.event_start]))

    @property
    def after(self) -> float:
        return float(np.mean(self.pr_auc[self.event_start :]))

    @property
    def not_computable_at_first(self) -> int:
        return sum(1 for n in self.labels_at_first if n < MIN_LABELS)


def concept_trace() -> ConceptTrace:
    """`demo/build.py` first evaluates every window with only the labels
    that landed within `early_cutoff_hours`, then inserts the rest and
    recomputes: two revisions. This recomputes both from the generator."""
    scenario = load_scenario("concept_drift")
    data = generate_scenario_data(scenario)
    t0 = scenario.start_date
    cutoff = scenario.label_delay.early_cutoff_hours
    label_by_id = {lab.prediction_id: lab for lab in data.labels}
    y: dict[int, list[float]] = defaultdict(list)
    score: dict[int, list[float]] = defaultdict(list)
    early: dict[int, int] = defaultdict(int)
    for p in data.predictions:
        w = int((p.predicted_at - t0).total_seconds() // 3600)
        lab = label_by_id[p.prediction_id]
        y[w].append(lab.label_value)
        score[w].append(p.prediction_score)
        if lab.delay_hours <= cutoff:
            early[w] += 1
    windows = sorted(y)
    event = next(e for e in scenario.events if e.shift_type == "label_noise")
    base_rate = scenario.label_base_noise_rate
    delays = np.array([lab.delay_hours for lab in data.labels])
    trace = ConceptTrace(
        windows=windows,
        pr_auc=[pr_auc(y[w], score[w], {}) for w in windows],
        labels_at_first=[early[w] for w in windows],
        predictions_per_window=scenario.predictions_per_window,
        early_cutoff_hours=cutoff,
        event_start=event.start_window,
        flip_rate=(base_rate, min(0.99, base_rate + event.magnitude)),
        arrived_by={h: float(np.mean(delays <= h)) for h in (1, 6, 12, 24, 72)},
    )
    got = (round(trace.before, 3), round(trace.after, 3), trace.not_computable_at_first)
    want = (VERIFIED_PR_AUC_BEFORE, VERIFIED_PR_AUC_AFTER, VERIFIED_NOT_COMPUTABLE)
    if got != want:
        raise SystemExit(f"concept_drift trace {got} disagrees with the verifier's {want}")
    return trace


# =====================================================================
# 00 HERO  (Wix CMS field: wordmark and tagline only, inside the safe box)
# =====================================================================
def wordmark(ax, cx, y, size, ha_gap=0.4):
    ax.text(cx, y, "Drift", fontsize=size, color="#FFFFFF", fontweight="bold",
            ha="right", va="center", zorder=Z_LABEL)
    ax.text(cx + ha_gap, y, "Watch", fontsize=size, color=VOLT_TXT, fontweight="bold",
            ha="left", va="center", zorder=Z_LABEL)


def hero(debug: bool = False) -> None:
    fig, ax = canvas(12.0, 12.0 / HERO_ASPECT, bg=BAND)
    Y = ax._Y
    rbox(ax, 0, 0, 100, Y, fc=BAND, ec="none", r=0.1, z=0)
    ax.add_patch(Circle((50, Y * 0.66), 15, color="#33211A", alpha=0.5, zorder=0))
    track(ax, 50, Y * 0.88, KICKER, size=8.5, color=VOLT_TXT, ha="center")
    wordmark(ax, 50, Y * 0.64, 34)
    ax.text(50, Y * 0.42, TAGLINE, fontsize=12, color="#D8D5CC", ha="center", va="center",
            zorder=Z_LABEL)
    safe_box(ax, debug)
    save(fig, P("00_hero.png"))


# =====================================================================
# 01 BANNER  (in-body Figure 1: the full information version)
# =====================================================================
def banner(trace: SegmentTrace, n_tests: int, n_files: int) -> None:
    fig, ax = canvas(12.8, 5.8, bg=BAND)
    Y = ax._Y
    rbox(ax, 0, 0, 100, Y, fc=BAND, ec="none", r=0.1, z=0)
    track(ax, 6, Y - 4.5, KICKER, size=8.5, color=VOLT_TXT)
    wordmark(ax, 14.4, Y - 12, 40, ha_gap=0.6)
    ax.text(6, Y - 19.5, BANNER_TAGLINE, fontsize=13, color="#D8D5CC", ha="left", va="center",
            zorder=Z_LABEL)
    stats = [
        ("5", "seeded scenarios, each verified at three seeds in CI"),
        (str(n_tests), f"tests across {n_files} files"),
        ("2", "independent clocks per window"),
        (f"{CLEAN_SPURIOUS_OPENS} of {CLEAN_SEEDS}", "clean seeds opening a spurious alert"),
    ]
    tw, th, gap = 21.4, 11.5, 1.8
    x0, y0 = 6, 2.6
    for i, (big, lab) in enumerate(stats):
        x = x0 + i * (tw + gap)
        rbox(ax, x, y0, tw, th, fc=BAND2, ec="#3A2B22", lw=1, r=2, z=Z_PANEL)
        ax.text(x + 2.4, y0 + th - 4.2, big, fontsize=18, color=VOLT_TXT,
                fontweight="bold", ha="left", va="center", zorder=Z_NODE_TEXT)
        ax.text(x + 2.4, y0 + 2.4, "\n".join(textwrap.wrap(lab.upper(), 34)), fontsize=6.6,
                color="#A9A59D", ha="left", va="bottom", zorder=Z_NODE_TEXT, linespacing=1.3)
    trace_inset(ax, trace, 68, Y - 22.5, 28, 18.5)
    save(fig, P("01_banner.png"))


def trace_inset(ax, trace: SegmentTrace, x, y, w, h) -> None:
    """The thesis with no number on it: the segment line clears the
    threshold, the blended line never leaves the floor."""
    rbox(ax, x, y, w, h, fc=BAND2, ec="#3A2B22", lw=1, r=2, z=Z_PANEL)
    ax.text(x + 2.0, y + h - 2.2, "one segment against the blend, same feature", fontsize=7.2,
            color="#A9A59D", ha="left", va="center", zorder=Z_NODE_TEXT)
    px, py, pw, ph = x + 2.6, y + 3.2, w - 5.2, h - 8.0
    top = max(trace.segment_psi) * 1.1
    xs = np.linspace(px, px + pw, len(trace.windows))
    ax.plot([px, px + pw], [py, py], color="#5A5650", lw=1.0, zorder=Z_NODE)
    fire_y = py + ph * (PSI_FIRE / top)
    ax.plot([px, px + pw], [fire_y, fire_y], color="#FFFFFF", lw=1.0, ls=(0, (2.4, 2.2)),
            zorder=Z_ARROW)
    ax.plot(xs, [py + ph * (v / top) for v in trace.global_psi], color="#8C8780", lw=1.4,
            zorder=Z_NODE_TEXT)
    ax.plot(xs, [py + ph * (v / top) for v in trace.segment_psi], color=VOLT, lw=1.8,
            zorder=Z_NODE_TEXT + 1)
    ax.text(px, py - 1.6, "the blend", fontsize=6.4, color="#FFFFFF", ha="left",
            va="center", zorder=Z_LABEL)
    ax.text(px + pw, py - 1.6, "the segment", fontsize=6.4, color=VOLT_TXT, ha="right",
            va="center", zorder=Z_LABEL)


# =====================================================================
# 02 METRICS CARD
# =====================================================================
def metrics_card(title, subtitle, tiles, headline=None, footnote=None,
                 out="02_metrics_card.png"):
    import math
    n = len(tiles)
    cols = n if n <= 3 else (2 if n == 4 else 3)
    rows = math.ceil(n / cols)
    left, right, gap, th = 4.0, 96.0, 3.0, 11.0
    top_pad = 11.5
    bot_pad = 8.0 if footnote else 3.5
    grid_h = rows * th + (rows - 1) * gap
    fig, ax = canvas(12.0, 12.0 * (top_pad + grid_h + bot_pad) / 100.0, bg=CARD)
    Y = ax._Y
    title_block(ax, 4, Y - 4.6, title, subtitle, tsize=15)
    tw = (right - left - gap * (cols - 1)) / cols
    grid_top = Y - top_pad
    for i, (value, label) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = left + c * (tw + gap)
        y = grid_top - th - r * (th + gap)
        accent = headline is not None and i == headline
        rbox(ax, x, y, tw, th, fc=VOLT_SOFT if accent else PAPER,
             ec=VOLT_EDGE if accent else BORDER_D, lw=1.6 if accent else 1.1, r=2.0, z=Z_NODE)
        vsize = 17 if accent else 19
        if len(str(value)) > 14:
            vsize = 14  # a long value never runs past the tile edge
        ax.text(x + 2.6, y + th - 4.6, str(value), fontsize=vsize, color=INK,
                fontweight="bold", ha="left", va="center", zorder=Z_NODE_TEXT)
        ax.text(x + 2.6, y + 2.4, "\n".join(textwrap.wrap(str(label).upper(), 42)),
                fontsize=7.2, color=VOLT_DEEP if accent else MUTE, ha="left", va="bottom",
                zorder=Z_NODE_TEXT, linespacing=1.3)
    if footnote:
        ax.text(4, 3.6, footnote, fontsize=7.6, color=MUTE, style="italic",
                ha="left", va="center", zorder=Z_LABEL, linespacing=1.4)
    save(fig, P(out))


# =====================================================================
# 03 ARCHITECTURE  (one store, two clocks, one engine; write-back on the
#                   right rail, the dashboard's read on the left rail)
#
# The scheduler drives both paths (`scheduler/run.py`: one tick, three
# jobs), so it sits between them and feeds each with its own arrow. Labels
# reach the performance path through the store, as the `performance_stale`
# flag label ingestion sets, never directly (`api/routes/labels.py`).
# =====================================================================
def architecture() -> None:
    fig, ax = canvas(12.6, 14.8, bg=CARD)
    Y = ax._Y
    title_block(ax, 4, Y - 5, "System architecture",
                "one append-only store, two evaluation paths that close at different times, "
                "one state machine deciding what is worth saying", tsize=16)

    BL, BR = 6.0, 94.0          # band extent; the two rails run outside it
    XL, XR = BL + 5, BR - 5     # node extent inside a band
    LX, RX = 2.6, 97.4          # left rail (dashboard read), right rail (write-back)
    C_IN = "#3F5C86"; C_DRIFT = "#8A6A1F"; C_PERF = "#3E8047"; C_OUT = "#5C6B82"

    def node(x, y, w, h, title, sub, fc=PAPER, ec=BORDER_D, ts=9.6, ss=7.8, lw=1.3):
        # sublabels sit at 7.8 point or larger: at a 720-pixel column anything
        # smaller renders near five pixels and stops reading
        rbox(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=2.0, z=Z_NODE)
        ax.text(x + w / 2, y + h - 3.0, title, fontsize=ts, color=INK, fontweight="bold",
                ha="center", va="center", zorder=Z_NODE_TEXT)
        if sub:
            ax.text(x + w / 2, y + 3.3, sub, fontsize=ss, color=MUTE, ha="center",
                    va="center", zorder=Z_NODE_TEXT, linespacing=1.25)
        return {"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2,
                "top": y + h, "bot": y, "L": x, "R": x + w}

    def band(y_bot, h, label, fc, ec):
        rbox(ax, BL, y_bot, BR - BL, h, fc=fc, ec=ec, lw=1.3, r=3, z=Z_BAND)
        # a band label never sits under an arrow
        ax.text(BL + 2.5, y_bot + h - 2.8, label, fontsize=9.0, color=INK,
                fontweight="bold", ha="left", zorder=Z_BAND + 1)

    def distribute(n, left, right, gap):
        w = (right - left - gap * (n - 1)) / n
        return [(left + i * (w + gap), w) for i in range(n)]

    nh = 9.6
    gap_b = 5.4
    h_in, h_hub, h_eval, h_alert, h_out = 28.0, 11.0, 16.4, 11.0, 16.4
    b_in = Y - 11.5 - h_in
    b_hub = b_in - gap_b - h_hub
    b_eval = b_hub - gap_b - h_eval
    b_alert = b_eval - gap_b - h_alert
    b_out = b_alert - gap_b - h_out

    # INGESTION: three writers, one endpoint under them
    band(b_in, h_in, "INGESTION : ANY MODEL, ONE SCHEMA", BLUE_T, BLUE_E)
    y_src = b_in + h_in - 5.0 - nh
    pos = distribute(3, XL, XR, 3.0)
    srcs = [
        node(pos[0][0], y_src, pos[0][1], nh, "predictions", "keyed on prediction_id"),
        node(pos[1][0], y_src, pos[1][1], nh, "labels", "arrive days or weeks later"),
        node(pos[2][0], y_src, pos[2][1], nh, "baseline registration",
             "bin edges frozen once,\nreplaces the active baseline"),
    ]
    y_ing = b_in + 2.4
    ing = node(XL, y_ing, XR - XL, nh, "FastAPI ingest",
               "schema failure refused with 422; duplicate skipped; late prediction stored "
               "and counted;\nnothing accepted is ever rewritten. A label batch flags the "
               "windows it touches and recomputes nothing", ss=8.0)
    for k in srcs:
        arrow(ax, k["cx"], k["bot"], k["cx"], ing["top"], color=C_IN, lw=1.8, ms=13)

    # POSTGRES: the one emphasised block
    hub = node(XL - 2, b_hub, XR - XL + 4, h_hub, "POSTGRES : SINGLE SOURCE OF TRUTH",
               "predictions, labels, baselines, drift and performance results, alert rows.\n"
               "Append-only, windowed by event time.",
               fc=VOLT_SOFT, ec=VOLT_EDGE, ts=10.5, ss=8.4, lw=1.8)
    # the arrow starts on the node's edge, not on the band's
    arrow(ax, ing["cx"], ing["bot"], ing["cx"], hub["top"], color=C_IN, lw=2.0, ms=15)
    alabel(ax, ing["cx"] + 1.2, (b_in + hub["top"]) / 2, "every accepted row, kept for good",
           color=C_IN, size=8.0)

    # EVALUATION: the scheduler in the middle drives both paths. Two offset
    # arrows from the store, one per clock, each with its own label; one
    # arrow out to each path. Labels never reach the performance path
    # directly: they land in the store and flag the windows they touch.
    # the two arrows from the store enter this band at its centre, so the
    # label sits at the left edge, over the drift path, where nothing enters
    band(b_eval, h_eval, "EVALUATION : TWO CLOCKS, THRESHOLDS FROM YAML PROFILES",
         SAND_T, SAND_E)
    y3 = b_eval + 2.4
    pos3 = distribute(3, XL, XR, 3.2)
    drift = node(pos3[0][0], y3, pos3[0][1], nh, "drift path",
                 "computed once at the\nwatermark, then sealed")
    sched = node(pos3[1][0], y3, pos3[1][1], nh, "scheduler",
                 "one process, three jobs a tick:\nseal drift, recompute flagged\n"
                 "performance, retry delivery")
    perf = node(pos3[2][0], y3, pos3[2][1], nh, "performance path",
                "recomputed for every flagged\nwindow, no watermark")
    dx = 3.6
    arrow(ax, sched["cx"] - dx, hub["bot"], sched["cx"] - dx, sched["top"],
          color=C_DRIFT, lw=2.0, ms=15)
    alabel(ax, sched["cx"] - dx - 1.2, (hub["bot"] + sched["top"]) / 2,
           "windows closed past the watermark", color=C_DRIFT, size=8.0, ha="right")
    arrow(ax, sched["cx"] + dx, hub["bot"], sched["cx"] + dx, sched["top"],
          color=C_PERF, lw=2.0, ms=15)
    alabel(ax, sched["cx"] + dx + 1.2, (hub["bot"] + sched["top"]) / 2,
           "windows a label batch flagged", color=C_PERF, size=8.0)
    arrow(ax, sched["L"], sched["cy"], drift["R"], drift["cy"], color=C_DRIFT, lw=2.0, ms=15)
    arrow(ax, sched["R"], sched["cy"], perf["L"], perf["cy"], color=C_PERF, lw=2.0, ms=15)

    # ALERT STATE MACHINE
    eng = node(XL - 2, b_alert, XR - XL + 4, h_alert, "ALERT STATE MACHINE",
               "persistence gates, hysteresis with a dead zone, recovery runs: "
               "open, escalate, resolve.\nEvidence frozen at open and again at escalation.",
               ts=10.0, ss=8.4)
    arrow(ax, drift["cx"], drift["bot"], drift["cx"], eng["top"], color=C_DRIFT, lw=2.0, ms=15)
    arrow(ax, perf["cx"], perf["bot"], perf["cx"], eng["top"], color=C_PERF, lw=2.0, ms=15)

    # write-back rail on the right margin: results and alert rows return to
    # the store, and the engine decides streaks by rescanning them there
    oarrow(ax, [(eng["R"], eng["cy"]), (RX, eng["cy"]), (RX, hub["cy"]), (hub["R"], hub["cy"])],
           color=C_OUT, lw=1.8, ms=14)
    alabel(ax, RX - 0.9, (b_eval + eng["top"]) / 2, "results and alerts\nwritten back",
           color=C_OUT, size=7.8, ha="right")

    # OUTPUT: the dashboard reads the store, not the engine, so it sits on
    # the left where the read rail reaches it without crossing anything
    band(b_out, h_out, "OUTPUT : EARNING ATTENTION", GREEN_T, GREEN_E)
    y5 = b_out + 2.4
    pos5 = distribute(2, XL, XR, 10)
    dash = node(pos5[0][0], y5, pos5[0][1], nh, "read-only dashboard",
                "SET TRANSACTION READ ONLY,\nURL is the full chart spec")
    notif = node(pos5[1][0], y5, pos5[1][1], nh, "notifications",
                 "log and generic webhook, on a status\ntransition, retried up to five ticks")
    arrow(ax, notif["cx"], eng["bot"], notif["cx"], notif["top"], color=C_OUT, lw=2.0, ms=15)
    oarrow(ax, [(hub["L"], hub["cy"]), (LX, hub["cy"]), (LX, dash["cy"]), (dash["L"], dash["cy"])],
           color=C_OUT, lw=1.8, ms=14)
    alabel(ax, LX + 0.9, (b_eval + eng["top"]) / 2, "reads every table,\nwrites none",
           color=C_OUT, size=7.8, ha="left")
    save(fig, P("03_architecture.png"))


# =====================================================================
# 04 TWO CLOCKS  (concept: one window, two lifecycles)
# =====================================================================
def two_clocks(concept: ConceptTrace) -> None:
    fig, ax = canvas(12.4, 5.6, bg=CARD)
    Y = ax._Y
    title_block(ax, 4, Y - 4.4, "A window has two lifecycles",
                "drift closes for good at the watermark; performance stays open indefinitely, "
                "recomputed whenever labels arrive", tsize=15)
    x0, x1 = 14, 88
    ticks = np.linspace(x0, x1, 7)

    # drift lane
    yd = Y - 15
    ax.text(x0 - 2, yd + 0.9, "drift", fontsize=10, color=INK, fontweight="bold", ha="right",
            va="center", zorder=Z_LABEL)
    ax.text(x0 - 2, yd - 1.9, "sealed once", fontsize=7.6, color=MUTE, ha="right", va="center",
            zorder=Z_LABEL)
    ax.plot([x0, x1], [yd, yd], color=BORDER_D, lw=1.4, zorder=Z_NODE)
    for t in ticks:
        ax.plot([t, t], [yd - 0.7, yd + 0.7], color=BORDER_D, lw=1.0, zorder=Z_NODE)
    wm = ticks[2]
    rbox(ax, x0, yd - 1.4, wm - x0, 2.8, fc=BLUE_T, ec="none", r=0.4, z=Z_PANEL)
    ax.plot([wm, wm], [yd - 3.6, yd + 3.6], color=INK, lw=1.4, ls=(0, (3, 2)), zorder=Z_ARROW)
    ax.text(wm, yd + 4.6, "watermark", fontsize=8, color=INK, ha="center", va="center",
            zorder=Z_LABEL)
    ax.add_patch(Circle((wm, yd), 1.3, color=VOLT, zorder=Z_NODE_TEXT))
    ax.text(wm + 2.4, yd + 2.6, "computed once, never revised", fontsize=8.6, color=VOLT_DEEP,
            fontweight="bold", ha="left", va="center", zorder=Z_LABEL)
    for t in ticks[3:6]:
        ax.add_patch(Circle((t + (ticks[1] - ticks[0]) * 0.45, yd), 1.0, color=RED, alpha=0.6,
                            zorder=Z_NODE_TEXT))
    ax.text(ticks[4], yd - 4.0, "late predictions: excluded from the sealed result, but counted",
            fontsize=7.8, color=RED, ha="center", va="center", zorder=Z_LABEL)

    # performance lane
    yp = Y - 30
    ax.text(x0 - 2, yp + 0.9, "performance", fontsize=10, color=INK, fontweight="bold",
            ha="right", va="center", zorder=Z_LABEL)
    ax.text(x0 - 2, yp - 1.9, "reopens forever", fontsize=7.6, color=MUTE, ha="right",
            va="center", zorder=Z_LABEL)
    rbox(ax, x0, yp - 1.4, x1 - x0, 2.8, fc=GREEN_T, ec="none", r=0.4, z=Z_PANEL)
    ax.plot([x0, x1], [yp, yp], color=BORDER_D, lw=1.4, zorder=Z_NODE)
    for t in ticks:
        ax.plot([t, t], [yp - 0.7, yp + 0.7], color=BORDER_D, lw=1.0, zorder=Z_NODE)
    for i, t in enumerate(ticks[1::1][:5]):
        tx = t + (ticks[1] - ticks[0]) * 0.5
        ax.add_patch(Circle((tx, yp), 1.3, color=GREEN, zorder=Z_NODE_TEXT))
        arrow(ax, tx, yp + 5.2, tx, yp + 1.6, color=GREEN, lw=1.2, ms=9)
        ax.text(tx, yp + 6.3, f"recompute {i + 1}", fontsize=7.4, color=GREEN, ha="center",
                va="center", zorder=Z_LABEL)
    ax.text(x0, yp - 4.2,
            f"{concept.not_computable_at_first} windows read not_computable until labels arrive, "
            "shown as an explicit marker, never as a gap",
            fontsize=7.8, color=MUTE, style="italic", ha="left", va="center", zorder=Z_LABEL)

    # the principle
    rbox(ax, x0, 2.6, x1 - x0, 5.4, fc=INK, ec="none", r=2.7, z=Z_NODE)
    ax.text((x0 + x1) / 2, 5.3, "A label arriving late is the domain. A prediction arriving late "
            "is a pipeline bug.", fontsize=9.4, color="#FFFFFF", ha="center", va="center",
            zorder=Z_NODE_TEXT)
    save(fig, P("04_two_clocks.png"))


# =====================================================================
# 05 SEGMENT AGAINST GLOBAL  (the real per-window trace)
# =====================================================================
def segment_vs_global(trace: SegmentTrace) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.4, 5.9))
    fig.patch.set_facecolor(CARD)
    ax = fig.add_axes([0.075, 0.13, 0.9, 0.68])
    ax.set_facecolor(CARD)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    w = np.array(trace.windows)
    g = np.array(trace.global_psi)
    s = np.array(trace.segment_psi)
    a, b = trace.shift_windows
    top = s.max() * 1.22

    ax.axvspan(a - 0.5, b - 0.5, color=SAND_T, zorder=Z_BAND)
    ax.text((a + b - 1) / 2, top * 0.985, "injected shift, one standard deviation, "
            f"{trace.share * 100:.0f} percent of traffic", fontsize=7.6, color=MUTE, ha="center",
            va="top",
            style="italic", zorder=Z_LABEL)
    ax.axhline(PSI_FIRE, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=Z_ARROW)
    ax.text(w[0], PSI_FIRE + top * 0.015, "fire threshold 0.25", fontsize=7.8, color=INK,
            ha="left", va="bottom", zorder=Z_LABEL)
    ax.axhline(PSI_CLEAR, color=MUTE, lw=0.9, ls=(0, (1.5, 2.5)), zorder=Z_ARROW)
    # at the right end both series sit under 0.07, so the label touches nothing
    ax.text(w[-1], PSI_CLEAR - top * 0.015, "clear threshold 0.15", fontsize=7.2, color=MUTE,
            ha="right", va="top", zorder=Z_LABEL)

    ax.plot(w, g, color="#4F86C6", lw=2.0, marker="o", ms=3.2, zorder=Z_NODE_TEXT,
            label="global, age feature")
    ax.plot(w, s, color=VOLT, lw=2.4, marker="o", ms=3.4, zorder=Z_NODE_TEXT + 1,
            label=f"APAC segment, age feature, about {trace.segment_rows:.0f} rows a window")

    i_s = int(np.argmax(s))
    ax.annotate(f"{s[i_s]:.2f}", xy=(w[i_s], s[i_s]), xytext=(w[i_s] - 7, s[i_s] + top * 0.04),
                fontsize=10, color=VOLT_DEEP, fontweight="bold", ha="center",
                arrowprops={"arrowstyle": "-|>", "color": VOLT_DEEP, "lw": 1.2}, zorder=Z_LABEL)
    i_g = int(np.argmax(g))
    # no leader line: a leader from outside the band would cross the segment
    # series on its way in. The label sits just above the global peak, under
    # the clear threshold, where the segment line is a full unit higher.
    ax.text(w[i_g] - 1, g[i_g] + top * 0.03, f"{g[i_g]:.3f}, never fires", fontsize=8.6,
            color="#2F5F9E", ha="center", va="bottom", zorder=Z_LABEL)

    # the state machine on the segment line
    life = lifecycle(list(s))
    marks = [("opens", life.opened), ("escalates", life.escalated), ("resolves", life.resolved)]
    for label, idx in marks:
        if idx is None:
            continue
        ax.plot([w[idx]], [s[idx]], marker="o", ms=8, mfc="none", mec=INK, mew=1.2,
                zorder=Z_LABEL)
        ax.text(w[idx], s[idx] + top * (0.05 if label != "resolves" else 0.07), label,
                fontsize=7.6, color=INK, ha="center", va="bottom", zorder=Z_LABEL)

    ax.set_xlim(w[0] - 0.5, w[-1] + 0.5)
    ax.set_ylim(0, top)
    ax.set_xlabel("evaluation window (one hour each)", fontsize=9, color=INK)
    ax.set_ylabel("population stability index", fontsize=9, color=INK)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", color=BORDER_D, lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.2)
    fig.text(0.04, 0.95, "THE LIE OF THE AVERAGE", fontsize=15, color=INK, fontweight="bold",
             ha="left", va="center")
    fig.text(0.04, 0.905,
             f"a {trace.share * 100:.0f} percent segment breaches while the blend stays under its "
             "threshold; "
             f"the quiet windows show the segment's own noise floor at "
             f"{trace.predictions_per_window} predictions a window",
             fontsize=9.5, color=MUTE, style="italic", ha="left", va="center")
    save(fig, P("05_segment_vs_global.png"))

# =====================================================================
# 06 ALERT LIFECYCLE  (the real per-window trace; under it, the state
#                      machine's decision for every window)
# =====================================================================
ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
           7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
STATE_FILL = {"quiet": SURFACE, "counting": VOLT_SOFT, "open": VOLT_EDGE, "escalated": VOLT,
              "resolved": GREEN_SOFT}
STATE_NAME = {"quiet": "quiet", "counting": f"breach, counting to {FIRE_PERSISTENCE}",
              "open": "open", "escalated": "escalated"}


def alert_lifecycle(trace: CovariateTrace) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    fig = plt.figure(figsize=(12.4, 6.8))
    fig.patch.set_facecolor(CARD)
    ax = fig.add_axes([0.075, 0.33, 0.9, 0.5])
    sx = fig.add_axes([0.075, 0.175, 0.9, 0.085], sharex=ax)
    for a in (ax, sx):
        a.set_facecolor(CARD)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    w = np.array(trace.windows)
    v = np.array(trace.psi)
    life = lifecycle(list(v))
    b0, b1 = trace.blip_windows
    s0 = trace.shift_start
    top = v.max() * 4.5
    floor = v.min() * 0.4

    # the two thresholds and the dead zone between them, on a log axis so
    # the quiet windows, the thresholds and the shift all stay readable
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.1, 1, 10]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.01", "0.1", "1", "10"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.axhspan(PSI_CLEAR, PSI_FIRE, color=SAND_T, zorder=Z_BAND)
    ax.axhline(PSI_FIRE, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=Z_ARROW)
    ax.axhline(PSI_CLEAR, color=MUTE, lw=0.9, ls=(0, (1.5, 2.5)), zorder=Z_ARROW)
    ax.text(w[0], PSI_FIRE * 1.12, f"fire threshold {PSI_FIRE}", fontsize=7.8, color=INK,
            ha="left", va="bottom", zorder=Z_LABEL)
    ax.text(w[0], PSI_CLEAR * 0.9, f"clear threshold {PSI_CLEAR}", fontsize=7.2, color=MUTE,
            ha="left", va="top", zorder=Z_LABEL)
    # the note sits in the sustained band, where the series runs far above it
    ax.text((s0 + w[-1]) / 2 + 1, PSI_FIRE * 1.12, "the dead zone between them counts for "
            "neither side and breaks any streak", fontsize=7.4, color=MUTE, style="italic",
            ha="center", va="bottom", zorder=Z_LABEL)

    # the two injected events
    ax.axvspan(b0 - 0.5, b1 - 0.5, color=SAND_T, zorder=Z_BAND)
    ax.axvspan(s0 - 0.5, w[-1] + 0.5, color=SAND_T, zorder=Z_BAND)
    ax.text((s0 + w[-1]) / 2 + 1, floor * 1.35,
            f"injected shift from window {s0}, sustained to the end", fontsize=7.6, color=MUTE,
            style="italic", ha="center", va="bottom", zorder=Z_LABEL)

    ax.plot(w, v, color="#4F86C6", lw=2.0, marker="o", ms=3.2, zorder=Z_NODE_TEXT,
            label=f"global, age feature, {trace.predictions_per_window} rows a window")

    # the blip: one window over the line, and the rule that keeps it quiet
    i_b = int(np.argmax(v[b0:b1])) + b0
    ax.annotate(f"one breaching window, {b1 - b0} of {FIRE_PERSISTENCE}:\nsuppressed, "
                "never opens", xy=(w[i_b], v[i_b]), xytext=(w[i_b] - 9, v[i_b] * 0.55),
                fontsize=8.4, color=INK, ha="right", va="center",
                arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.1}, zorder=Z_LABEL)

    # the sustained shift: where the machine opens and escalates, with the
    # reading it froze as evidence at each step
    marks = [("opens", life.opened, FIRE_PERSISTENCE, "right", -1.2),
             ("escalates", life.escalated, ESCALATE_PERSISTENCE, "left", 1.2)]
    for label, idx, need, ha, dx in marks:
        if idx is None:
            continue
        ax.plot([w[idx]], [v[idx]], marker="o", ms=8, mfc="none", mec=INK, mew=1.2,
                zorder=Z_LABEL)
        ax.text(w[idx] + dx, v[idx] * 2.4, f"{label} on the {ORDINAL[need]} breaching window\n"
                f"evidence frozen at {v[idx]:.2f}", fontsize=7.8, color=INK, ha=ha,
                va="bottom", zorder=Z_LABEL)

    ax.set_xlim(w[0] - 0.5, w[-1] + 0.5)
    ax.set_ylim(floor, top)
    ax.set_ylabel("population stability index, log scale", fontsize=9, color=INK)
    ax.tick_params(labelsize=8, labelbottom=False)
    ax.grid(axis="y", color=BORDER_D, lw=0.5, alpha=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8.2)

    # the state strip: one cell per window, the streak count written in
    # while it is still deciding
    for i, (state, run) in enumerate(zip(life.state, life.breach_run, strict=True)):
        sx.add_patch(Rectangle((w[i] - 0.5, 0), 1, 1, fc=STATE_FILL[state], ec=CARD, lw=0.6,
                               zorder=Z_NODE))
        if 0 < run <= ESCALATE_PERSISTENCE:
            sx.text(w[i], 0.5, str(run), fontsize=6.2, ha="center", va="center",
                    color=PAPER if state == "escalated" else INK, zorder=Z_NODE_TEXT)
    sx.set_ylim(0, 1)
    sx.set_yticks([])
    sx.spines["left"].set_visible(False)
    sx.set_ylabel("alert\nstate", fontsize=8, color=INK, rotation=0, ha="right", va="center",
                  labelpad=6)
    sx.set_xlabel("evaluation window (one hour each)", fontsize=9, color=INK)
    sx.tick_params(labelsize=8)
    handles = [Patch(fc=STATE_FILL[k], ec=BORDER_D, lw=0.5, label=STATE_NAME[k])
               for k in ("quiet", "counting", "open", "escalated")]
    sx.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.95), ncol=4,
              frameon=False, fontsize=7.8, handlelength=1.2, columnspacing=1.6)

    fig.text(0.04, 0.95, "ALERTS YOU CAN TRUST", fontsize=15, color=INK, fontweight="bold",
             ha="left", va="center")
    fig.text(0.04, 0.905,
             f"one breaching window never opens an alert under a {FIRE_PERSISTENCE}-window "
             f"persistence rule; a sustained breach opens on its {ORDINAL[FIRE_PERSISTENCE]} "
             f"window and escalates on its {ORDINAL[ESCALATE_PERSISTENCE]}, "
             "with the evidence frozen at each step",
             fontsize=9.5, color=MUTE, style="italic", ha="left", va="center")
    save(fig, P("06_alert_lifecycle.png"))


# =====================================================================
# 07 PERFORMANCE BACKFILL  (the real per-window trace: what each window
#                           held at first evaluation, and what it read
#                           once every label was in)
# =====================================================================
def performance_backfill(trace: ConceptTrace) -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12.4, 6.8))
    fig.patch.set_facecolor(CARD)
    ax = fig.add_axes([0.075, 0.38, 0.9, 0.45])
    bx = fig.add_axes([0.075, 0.115, 0.9, 0.19], sharex=ax)
    for a in (ax, bx):
        a.set_facecolor(CARD)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    w = np.array(trace.windows)
    v = np.array(trace.pr_auc)
    n0 = np.array(trace.labels_at_first)
    ev = trace.event_start
    life = lifecycle(list(v), PR_AUC_FIRE, PR_AUC_CLEAR, direction="below")

    # the latest revision: every label in
    ax.axvspan(ev - 0.5, w[-1] + 0.5, color=SAND_T, zorder=Z_BAND)
    ax.text((ev + w[-1]) / 2, 1.04, f"injected label noise from window {ev}: flip rate "
            f"{trace.flip_rate[0]:.2f} to {trace.flip_rate[1]:.2f}", fontsize=7.6, color=MUTE,
            style="italic", ha="center", va="top", zorder=Z_LABEL)
    ax.axhline(PR_AUC_FIRE, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=Z_ARROW)
    ax.text(w[0], PR_AUC_FIRE - 0.015, f"fire threshold {PR_AUC_FIRE}, a breach is below it",
            fontsize=7.8, color=INK, ha="left", va="top", zorder=Z_LABEL)
    ax.axhline(PR_AUC_CLEAR, color=MUTE, lw=0.9, ls=(0, (1.5, 2.5)), zorder=Z_ARROW)
    ax.text(w[0], PR_AUC_CLEAR + 0.015, f"clear threshold {PR_AUC_CLEAR}", fontsize=7.2,
            color=MUTE, ha="left", va="bottom", zorder=Z_LABEL)
    ax.plot(w, v, color=VOLT, lw=2.4, marker="o", ms=3.4, zorder=Z_NODE_TEXT + 1,
            label="PR-AUC, latest revision, every label in")
    for lo, hi, mean, y_txt, va in ((0, ev, trace.before, 0.97, "bottom"),
                                    (ev, len(w), trace.after, 0.27, "top")):
        ax.hlines(mean, w[lo] - 0.5, w[hi - 1] + 0.5, color=VOLT_DEEP, lw=1.0,
                  ls=(0, (6, 3)), zorder=Z_ARROW)
        ax.text((w[lo] + w[hi - 1]) / 2, y_txt, f"mean {mean:.3f} over windows {w[lo]} to "
                f"{w[hi - 1]}", fontsize=8.6, color=VOLT_DEEP, fontweight="bold", ha="center",
                va=va, zorder=Z_LABEL)
    for label, idx in (("opens", life.opened), ("escalates", life.escalated)):
        if idx is None:
            continue
        ax.plot([w[idx]], [v[idx]], marker="o", ms=8, mfc="none", mec=INK, mew=1.2,
                zorder=Z_LABEL)
        ax.text(w[idx], v[idx] + 0.045, label, fontsize=7.6, color=INK, ha="center",
                va="bottom", zorder=Z_LABEL)
    ax.set_xlim(w[0] - 0.5, w[-1] + 0.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("PR-AUC", fontsize=9, color=INK)
    ax.tick_params(labelsize=8, labelbottom=False)
    ax.grid(axis="y", color=BORDER_D, lw=0.5, alpha=0.5)
    ax.legend(loc="lower left", frameon=False, fontsize=8.2)

    # the initial revision: what each window held when it was first
    # evaluated, against the minimum the pipeline will compute on
    bx.bar(w, n0, width=0.8, color=C_GRAY, zorder=Z_NODE)
    bx.axhline(MIN_LABELS, color=INK, lw=1.0, ls=(0, (1.5, 2.5)), zorder=Z_ARROW)
    bx.text(w[-1], MIN_LABELS + MIN_LABELS * 0.06, f"minimum {MIN_LABELS} labeled rows "
            "to compute", fontsize=7.4, color=INK, ha="right", va="bottom", zorder=Z_LABEL)
    bx.text(w[0], MIN_LABELS * 1.28, f"labels present at first evaluation, within "
            f"{trace.early_cutoff_hours:.0f} hour of prediction: about {n0.mean():.0f} of "
            f"{trace.predictions_per_window} a window, so all {trace.not_computable_at_first} "
            "windows read not_computable", fontsize=7.8, color=MUTE, style="italic",
            ha="left", va="top", zorder=Z_LABEL)
    bx.set_ylim(0, MIN_LABELS * 1.35)
    bx.set_yticks([0, MIN_LABELS])
    bx.set_ylabel("labels at first\nevaluation", fontsize=8, color=INK)
    bx.set_xlabel("evaluation window (one hour each)", fontsize=9, color=INK)
    bx.tick_params(labelsize=8)

    fig.text(0.04, 0.95, "TIME YOU CAN'T RUSH", fontsize=15, color=INK, fontweight="bold",
             ha="left", va="center")
    fig.text(0.04, 0.905,
             f"{trace.arrived_by[6] * 100:.0f} percent of a window's labels land within six "
             f"hours of prediction and {trace.arrived_by[1] * 100:.1f} percent within the "
             "first, so a window's first evaluation has almost nothing to score",
             fontsize=9.5, color=MUTE, style="italic", ha="left", va="center")
    fig.text(0.04, 0.877,
             f"once the rest arrive, PR-AUC reads {trace.before:.3f} before the injected label "
             f"noise and {trace.after:.3f} after",
             fontsize=9.5, color=MUTE, style="italic", ha="left", va="center")
    save(fig, P("07_performance_backfill.png"))


if __name__ == "__main__":
    trace = segment_trace()
    covariate = covariate_trace()
    concept = concept_trace()
    n_tests, n_files = test_counts()
    seg_life = lifecycle(trace.segment_psi)
    print(f"global max {max(trace.global_psi):.3f}, segment max {max(trace.segment_psi):.2f}, "
          f"lifecycle {(seg_life.opened, seg_life.escalated, seg_life.resolved)}, "
          f"PR-AUC {concept.before:.3f} to {concept.after:.3f}, "
          f"tests {n_tests} in {n_files} files")
    hero()
    banner(trace, n_tests, n_files)
    metrics_card(
        "Project metrics",
        "every figure here is re-derived from the database by a verifier on each run",
        [
            ("5", "seeded scenarios, verified at three seeds"),
            (str(n_tests), f"tests across {n_files} files"),
            (f"{max(trace.global_psi):.3f} vs {max(trace.segment_psi):.2f}",
             "global against segment drift, same feature, same windows"),
            (f"{concept.before:.3f} to {concept.after:.3f}",
             "PR-AUC before and after injected label noise"),
            (str(concept.not_computable_at_first),
             "windows correctly not computable before labels arrived"),
            ("Python, FastAPI, Postgres", "stack"),
        ],
        headline=2,
        footnote="Synthetic staged scenarios against a real Postgres. These figures prove the "
                 "pipeline behaves as claimed, not detection sensitivity on real traffic.",
    )
    architecture()
    two_clocks(concept)
    segment_vs_global(trace)
    alert_lifecycle(covariate)
    performance_backfill(concept)
