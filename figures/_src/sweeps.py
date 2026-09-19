"""Two experiments the case study reports, run on the pure generator with
the repository's own statistics and streak rules -- no database needed.

    uv run python figures/_src/sweeps.py

1. False-open rate on clean data: the segment_isolated geometry with its
   event removed, across many seeds, under the patient profile. How often
   does a PSI or KS alert open on a quiet segment, and how often does the
   PSI sampling-noise floor refuse to gate at all?
2. Segment share against shift magnitude: at the shipped volume, for
   which (share, magnitude) cells does the global reading stay under the
   fire threshold while the segment fires?

Both use the same streak rule the alert engine uses (three consecutive
breaches to open under patient) on the same per-window statistics the
evaluation pipeline computes, so a cell here is what `driftwatch demo`
would have produced for that scenario.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

os.environ.setdefault("CONFIGS_DIR", "configs")


from driftwatch.demo.generator import generate_scenario_data  # noqa: E402
from driftwatch.demo.loader import load_scenario  # noqa: E402
from driftwatch.stats.binning import compute_continuous_edges  # noqa: E402
from driftwatch.stats.ks import ks  # noqa: E402
from driftwatch.stats.psi import psi, psi_null_floor  # noqa: E402

PSI_FIRE, PSI_CLEAR = 0.25, 0.15  # patient.yaml
KS_FIRE, KS_CLEAR = 0.3, 0.15
FIRE_PERSISTENCE = 3


def classify(value: float | None, fire: float, clear: float) -> str:
    if value is None:
        return "n"  # not computable
    if value >= fire:
        return "B"
    if value <= clear:
        return "c"
    return "d"


def opens(seq: str, persistence: int = FIRE_PERSISTENCE) -> bool:
    run = 0
    for c in seq:
        run = run + 1 if c == "B" else 0
        if run >= persistence:
            return True
    return False


def traces(scenario, feature: str, segment: str | None, data=None, ignore_floor=False):
    """Per-window (psi, ks) for one feature, globally or inside one region.
    `ignore_floor` reproduces the pre-v1.0.0 behaviour, where PSI was gated
    on at any volume."""
    data = data or generate_scenario_data(scenario)
    t0 = scenario.start_date
    hours = 3600

    def keep(record):
        return segment is None or record.segment_values.get("region") == segment

    base_all = [
        r.features[feature] for r in data.baseline_records if r.features.get(feature) is not None
    ]
    edges = compute_continuous_edges(base_all)
    base = [
        r.features[feature]
        for r in data.baseline_records
        if keep(r) and r.features.get(feature) is not None
    ]
    by_window: dict[int, list[float]] = defaultdict(list)
    for p in data.predictions:
        v = p.features.get(feature)
        if v is None or not keep(p):
            continue
        by_window[int((p.predicted_at - t0).total_seconds() // hours)].append(v)
    psi_vals, ks_vals, refused = [], [], 0
    n_buckets = len(edges) + 1
    for w in sorted(by_window):
        live = by_window[w]
        if not ignore_floor and psi_null_floor(len(base), len(live), n_buckets).ceiling > PSI_CLEAR:
            psi_vals.append(None)
            refused += 1
        else:
            psi_vals.append(psi(base, live, edges).value)
        ks_vals.append(ks(base, live).statistic)
    return psi_vals, ks_vals, refused


def false_open_sweep(seeds=range(2000, 2040), volume=None, ignore_floor=False) -> None:
    base = load_scenario("segment_isolated")
    if volume is not None:
        base = base.model_copy(
            update={"predictions_per_window": volume[0], "baseline_size": volume[1]}
        )
    print(
        f"# Clean data at the segment_isolated geometry ({base.predictions_per_window} "
        f"predictions per window, {base.baseline_size} baseline rows, APAC 15 percent), "
        f"{len(seeds)} seeds, patient profile"
        + (", PSI gated at any volume (pre-v1.0.0 behaviour)" if ignore_floor else "")
    )
    print("| Signal | Seeds with a spurious open | Windows where PSI was refused by its floor |")
    print("|---|---:|---:|")
    cells = [(f, s) for f in ("age", "income") for s in (None, "APAC")]
    tally = {cell: [0, 0, 0, 0] for cell in cells}  # psi opens, ks opens, refused, windows
    for seed in seeds:
        scenario = base.model_copy(update={"seed": seed, "events": []})
        data = generate_scenario_data(scenario)
        for cell in cells:
            psi_vals, ks_vals, refused = traces(scenario, cell[0], cell[1], data, ignore_floor)
            t = tally[cell]
            t[0] += opens("".join(classify(v, PSI_FIRE, PSI_CLEAR) for v in psi_vals))
            t[1] += opens("".join(classify(v, KS_FIRE, KS_CLEAR) for v in ks_vals))
            t[2] += refused
            t[3] += len(psi_vals)
    for (feature, segment), (psi_opens, ks_opens, refused_total, windows_total) in tally.items():
        where = "global" if segment is None else f"{segment} segment"
        print(
            f"| `{feature}`, {where}, PSI | {psi_opens} of {len(seeds)} | "
            f"{refused_total} of {windows_total} |"
        )
        print(f"| `{feature}`, {where}, KS D | {ks_opens} of {len(seeds)} | not applicable |")


def share_magnitude_sweep() -> None:
    base = load_scenario("segment_isolated")
    shares = (0.05, 0.15, 0.30)
    magnitudes = (0.25, 0.5, 1.0)
    print(
        f"\n# Segment share against shift magnitude, seed {base.seed}, "
        f"{base.predictions_per_window} predictions per window, patient profile"
    )
    print(
        "| APAC share | Shift (std) | Global age PSI, max | APAC age PSI, max "
        "| APAC PSI alert | APAC KS D alert | Global alert |"
    )
    print("|---:|---:|---:|---:|---|---|---|")
    for share in shares:
        rest = (1 - share) / 2
        features = []
        for f in base.features:
            if f.name == "region":
                features.append(f.model_copy(update={"weights": [rest, rest, share]}))
            else:
                features.append(f)
        for magnitude in magnitudes:
            event = base.events[0].model_copy(update={"magnitude": magnitude})
            scenario = base.model_copy(update={"features": features, "events": [event]})
            g_psi, g_ks, _ = traces(scenario, "age", None)
            s_psi, s_ks, refused = traces(scenario, "age", "APAC")
            g_max = max(v for v in g_psi if v is not None)
            s_present = [v for v in s_psi if v is not None]
            s_max = f"{max(s_present):.2f}" if s_present else "refused"
            s_psi_alert = (
                "refused (below floor)"
                if refused == len(s_psi)
                else (
                    "opens"
                    if opens("".join(classify(v, PSI_FIRE, PSI_CLEAR) for v in s_psi))
                    else "quiet"
                )
            )
            s_ks_alert = (
                "opens" if opens("".join(classify(v, KS_FIRE, KS_CLEAR) for v in s_ks)) else "quiet"
            )
            g_alert = (
                "opens"
                if opens("".join(classify(v, PSI_FIRE, PSI_CLEAR) for v in g_psi))
                or opens("".join(classify(v, KS_FIRE, KS_CLEAR) for v in g_ks))
                else "quiet"
            )
            print(
                f"| {share:.0%} | {magnitude} | {g_max:.3f} | {s_max} | {s_psi_alert} "
                f"| {s_ks_alert} | {g_alert} |"
            )


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "clean"):
        false_open_sweep()
    if which in ("all", "clean-before"):
        # the geometry the case study was first drafted on, 500 per window
        # against 1500, with PSI gated regardless of its floor
        false_open_sweep(volume=(500, 1500), ignore_floor=True)
    if which in ("all", "share"):
        share_magnitude_sweep()
