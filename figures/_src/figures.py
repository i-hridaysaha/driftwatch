"""Case-study figures for DriftWatch, drawn from the repository's own data.

    uv run --with matplotlib python figures/_src/figures.py

Every plotted number is computed here from the pure scenario generator and
the repository's own statistics (`driftwatch.demo.generator`,
`driftwatch.stats`), the same code the evaluation pipeline runs, so the
segment chart is the scenario's actual per-window trace and not a drawing
of it. The three performance numbers on the metrics card come from the
database (`driftwatch demo-verify concept_drift`); they are pinned below
with their provenance and are the only typed numbers in this file. The
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
    CARD,
    GREEN,
    GREEN_E,
    GREEN_T,
    HERO_ASPECT,
    INK,
    MUTE,
    PAPER,
    RED,
    SAND_E,
    SAND_T,
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

from driftwatch.demo.generator import generate_scenario_data  # noqa: E402
from driftwatch.demo.loader import load_scenario  # noqa: E402
from driftwatch.stats.binning import compute_continuous_edges  # noqa: E402
from driftwatch.stats.psi import psi  # noqa: E402

use_theme()
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)

KICKER = "ML MONITORING : DELAYED GROUND TRUTH"
TAGLINE = "Two clocks, one model."
BANNER_TAGLINE = "Separating data drift from performance decay, and alerting only on what persists"

# ---- numbers that come from the database, pinned with their provenance ----
# `driftwatch demo-verify concept_drift`, quoted verbatim in README.md:
#   PR-AUC healthy before label_noise, degraded after: early_mean=0.874, late_mean=0.387
#   degradation only visible after backfill: 60 windows initially not_computable
PR_AUC_BEFORE, PR_AUC_AFTER, NOT_COMPUTABLE_WINDOWS = 0.874, 0.387, 60
# figures/_src/sweeps.py, `clean`: forty clean seeds at the shipped segment geometry
CLEAN_SEEDS, CLEAN_SPURIOUS_OPENS = 40, 0
# patient.yaml, the profile every shipped scenario runs under
PSI_FIRE, PSI_CLEAR = 0.25, 0.15
FIRE_PERSISTENCE, ESCALATE_PERSISTENCE, RESOLVE_PERSISTENCE = 3, 8, 5


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


def lifecycle(values: list[float]) -> dict[str, int | None]:
    """Where the patient profile's state machine opens, escalates and
    resolves on this trace: the same streak rule the engine uses."""
    opened = escalated = resolved = None
    run_b = run_c = 0
    status = None
    for i, v in enumerate(values):
        kind = "B" if v >= PSI_FIRE else ("c" if v <= PSI_CLEAR else "d")
        run_b = run_b + 1 if kind == "B" else 0
        run_c = run_c + 1 if kind == "c" else 0
        if status is None and run_b >= FIRE_PERSISTENCE:
            status, opened = "open", i
        elif status == "open" and run_b >= ESCALATE_PERSISTENCE:
            status, escalated = "escalated", i
        elif status in ("open", "escalated") and run_c >= RESOLVE_PERSISTENCE:
            status, resolved = "resolved", i
    return {"opened": opened, "escalated": escalated, "resolved": resolved}


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
# =====================================================================
def architecture() -> None:
    fig, ax = canvas(12.6, 14.4, bg=CARD)
    Y = ax._Y
    title_block(ax, 4, Y - 5, "System architecture",
                "one append-only store, two evaluation paths that close at different times, "
                "one engine deciding what is worth saying", tsize=16)

    BL, BR = 6.0, 94.0          # band extent; the two rails run outside it
    XL, XR = BL + 5, BR - 5     # node extent inside a band
    LX, RX = 2.6, 97.4          # left rail (dashboard read), right rail (write-back)
    C_IN = "#3F5C86"; C_DRIFT = "#8A6A1F"; C_PERF = "#3E8047"; C_OUT = "#5C6B82"

    def node(x, y, w, h, title, sub, fc=PAPER, ec=BORDER_D, ts=9.6, ss=7.2, lw=1.3):
        rbox(ax, x, y, w, h, fc=fc, ec=ec, lw=lw, r=2.0, z=Z_NODE)
        ax.text(x + w / 2, y + h - 3.0, title, fontsize=ts, color=INK, fontweight="bold",
                ha="center", va="center", zorder=Z_NODE_TEXT)
        if sub:
            ax.text(x + w / 2, y + 2.9, sub, fontsize=ss, color=MUTE, ha="center",
                    va="center", zorder=Z_NODE_TEXT, linespacing=1.25)
        return {"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2, "cy": y + h / 2,
                "top": y + h, "bot": y, "L": x, "R": x + w}

    def band(y_bot, h, label, fc, ec, label_x=None):
        rbox(ax, BL, y_bot, BR - BL, h, fc=fc, ec=ec, lw=1.3, r=3, z=Z_BAND)
        # a band label never sits under an arrow, so a band with arrows
        # entering near its left edge carries its label in the middle
        ax.text(BL + 2.5 if label_x is None else label_x, y_bot + h - 2.8, label, fontsize=9.0,
                color=INK, fontweight="bold", ha="left" if label_x is None else "center",
                zorder=Z_BAND + 1)

    def distribute(n, left, right, gap):
        w = (right - left - gap * (n - 1)) / n
        return [(left + i * (w + gap), w) for i in range(n)]

    nh = 9.2
    gap_b = 5.4
    h_in, h_hub, h_eval, h_alert, h_out = 25.5, 11.0, 16.0, 11.0, 16.0
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
    ing = node(XL, y_ing, XR - XL, nh - 1.6, "FastAPI ingest",
               "schema failure refused with 422; duplicate skipped; late prediction "
               "stored and counted; nothing accepted is ever rewritten", ss=7.0)
    for k in srcs:
        arrow(ax, k["cx"], k["bot"], k["cx"], ing["top"], color=C_IN, lw=1.8, ms=13)

    # POSTGRES: the one emphasised block
    hub = node(XL - 2, b_hub, XR - XL + 4, h_hub, "POSTGRES : SINGLE SOURCE OF TRUTH",
               "predictions, labels, baselines, drift and performance results, alert rows. "
               "Append-only, windowed by event time.",
               fc=VOLT_SOFT, ec=VOLT_EDGE, ts=10.5, ss=7.4, lw=1.8)
    arrow(ax, ing["cx"], b_in, ing["cx"], hub["top"], color=C_IN, lw=2.0, ms=15)
    alabel(ax, ing["cx"] + 1.2, (b_in + hub["top"]) / 2, "every accepted row, kept for good",
           color=C_IN, size=7.2)

    # EVALUATION: scheduler, drift path, performance path
    band(b_eval, h_eval, "EVALUATION : TWO CLOCKS, THRESHOLDS AND PERSISTENCE FROM YAML PROFILES",
         SAND_T, SAND_E, label_x=50)
    y3 = b_eval + 2.4
    pos3 = distribute(3, XL, XR, 3.2)
    sched = node(pos3[0][0], y3, pos3[0][1], nh, "scheduler",
                 "one process, windows by event\ntime, closed past the watermark")
    drift = node(pos3[1][0], y3, pos3[1][1], nh, "drift path",
                 "computed once at the\nwatermark, then sealed")
    perf = node(pos3[2][0], y3, pos3[2][1], nh, "performance path",
                "recomputed on every label\nbatch, no watermark")
    arrow(ax, sched["cx"], hub["bot"], sched["cx"], sched["top"], color=C_DRIFT, lw=2.0, ms=15)
    alabel(ax, sched["cx"] + 1.2, (hub["bot"] + sched["top"]) / 2,
           "closed windows, predictions only", color=C_DRIFT, size=7.2)
    arrow(ax, sched["R"], sched["cy"], drift["L"], drift["cy"], color=C_DRIFT, lw=2.0, ms=15)
    arrow(ax, perf["cx"], hub["bot"], perf["cx"], perf["top"], color=C_PERF, lw=2.0, ms=15)
    alabel(ax, perf["cx"] - 1.2, (hub["bot"] + perf["top"]) / 2,
           "labels, whenever they land", color=C_PERF, size=7.2, ha="right")

    # ALERT STATE MACHINE
    eng = node(XL - 2, b_alert, XR - XL + 4, h_alert, "ALERT STATE MACHINE",
               "persistence gates, hysteresis with a dead zone, recovery runs: "
               "open, escalate, resolve. Evidence frozen at open and again at escalation.",
               ts=10.0, ss=7.2)
    arrow(ax, drift["cx"], drift["bot"], drift["cx"], eng["top"], color=C_DRIFT, lw=2.0, ms=15)
    arrow(ax, perf["cx"], perf["bot"], perf["cx"], eng["top"], color=C_PERF, lw=2.0, ms=15)

    # write-back rail on the right margin: results and alert rows return to
    # the store, and the engine decides streaks by rescanning them there
    oarrow(ax, [(eng["R"], eng["cy"]), (RX, eng["cy"]), (RX, hub["cy"]), (hub["R"], hub["cy"])],
           color=C_OUT, lw=1.8, ms=14)
    alabel(ax, RX - 0.9, (b_eval + eng["top"]) / 2, "results and alerts\nwritten back",
           color=C_OUT, size=6.8, ha="right")

    # OUTPUT: the dashboard reads the store, not the engine, so it sits on
    # the left where the read rail reaches it without crossing anything
    band(b_out, h_out, "OUTPUT : EARNING ATTENTION", GREEN_T, GREEN_E)
    y5 = b_out + 2.4
    pos5 = distribute(2, XL, XR, 10)
    dash = node(pos5[0][0], y5, pos5[0][1], nh, "read-only dashboard",
                "SET TRANSACTION READ ONLY,\nURL is the full chart spec")
    notif = node(pos5[1][0], y5, pos5[1][1], nh, "notifications",
                 "log and generic webhook, on a status\ntransition only, no retry")
    arrow(ax, notif["cx"], eng["bot"], notif["cx"], notif["top"], color=C_OUT, lw=2.0, ms=15)
    oarrow(ax, [(hub["L"], hub["cy"]), (LX, hub["cy"]), (LX, dash["cy"]), (dash["L"], dash["cy"])],
           color=C_OUT, lw=1.8, ms=14)
    alabel(ax, LX + 0.9, (b_eval + eng["top"]) / 2, "reads every table,\nwrites none",
           color=C_OUT, size=6.8, ha="left")
    save(fig, P("03_architecture.png"))


# =====================================================================
# 04 TWO CLOCKS  (concept: one window, two lifecycles)
# =====================================================================
def two_clocks() -> None:
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
            f"{NOT_COMPUTABLE_WINDOWS} windows read not_computable until labels arrive, "
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
    ax.text(w[0], PSI_CLEAR - top * 0.015, "clear threshold 0.15", fontsize=7.2, color=MUTE,
            ha="left", va="top", zorder=Z_LABEL)

    ax.plot(w, g, color="#4F86C6", lw=2.0, marker="o", ms=3.2, zorder=Z_NODE_TEXT,
            label="global, age feature")
    ax.plot(w, s, color=VOLT, lw=2.4, marker="o", ms=3.4, zorder=Z_NODE_TEXT + 1,
            label=f"APAC segment, age feature, about {trace.segment_rows:.0f} rows a window")

    i_s = int(np.argmax(s))
    ax.annotate(f"{s[i_s]:.2f}", xy=(w[i_s], s[i_s]), xytext=(w[i_s] - 7, s[i_s] + top * 0.04),
                fontsize=10, color=VOLT_DEEP, fontweight="bold", ha="center",
                arrowprops={"arrowstyle": "-|>", "color": VOLT_DEEP, "lw": 1.2}, zorder=Z_LABEL)
    i_g = int(np.argmax(g))
    ax.annotate(f"{g[i_g]:.3f}, never fires", xy=(w[i_g], g[i_g]),
                xytext=(w[i_g] - 9.5, g[i_g] + top * 0.22), fontsize=8.6, color="#2F5F9E",
                ha="center", arrowprops={"arrowstyle": "-|>", "color": "#2F5F9E", "lw": 1.0},
                zorder=Z_LABEL)

    # the state machine on the segment line
    life = lifecycle(list(s))
    marks = [("opens", life["opened"]), ("escalates", life["escalated"]),
             ("resolves", life["resolved"])]
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


if __name__ == "__main__":
    trace = segment_trace()
    n_tests, n_files = test_counts()
    print(f"global max {max(trace.global_psi):.3f}, segment max {max(trace.segment_psi):.2f}, "
          f"lifecycle {lifecycle(trace.segment_psi)}, tests {n_tests} in {n_files} files")
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
            (f"{PR_AUC_BEFORE} to {PR_AUC_AFTER}", "PR-AUC before and after injected label noise"),
            (str(NOT_COMPUTABLE_WINDOWS), "windows correctly not computable before labels arrived"),
            ("Python, FastAPI, Postgres", "stack"),
        ],
        headline=2,
        footnote="Synthetic staged scenarios against a real Postgres. These figures prove the "
                 "pipeline behaves as claimed, not detection sensitivity on real traffic.",
    )
    architecture()
    two_clocks()
    segment_vs_global(trace)
