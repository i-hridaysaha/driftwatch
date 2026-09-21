# Figure source

The case study's figures, drawn from the repository rather than typed.

    uv run --with matplotlib python figures/_src/figures.py   # every PNG into figures/
    uv run python figures/_src/sweeps.py                     # the two tables the case study quotes

| File | Job |
|---|---|
| `theme.py` | the portfolio figure kit's palette, z-order contract and primitives, copied from the kit with its imports tidied for this repository's linter |
| `figures.py` | one function per figure; the segment, alert and performance charts are recomputed window by window from `driftwatch.demo.generator` with `driftwatch.stats` and `driftwatch.metrics`, the thresholds and persistence counts are read from `configs/profiles/patient.yaml`, the test count comes from `pytest --collect-only`, and the three performance numbers are recomputed and checked against their pinned `demo-verify` values, so a mismatch stops the render |
| `sweeps.py` | forty clean seeds at the segment geometry (spurious opens per signal), the same at the pre-v1.0.0 geometry with PSI gated regardless of its floor, and a segment-share against shift-magnitude sweep |

| Output | Case study (`Website/Case Study Writing/Case Studies/Drift Watch CASE_STUDY.md`) |
|---|---|
| `00_hero.png` | hero (CMS field) |
| `01_banner.png` | Figure 1 |
| `02_metrics_card.png` | Figure 2 |
| `03_architecture.png` | Figure 3, also embedded in the README |
| `04_two_clocks.png` | Figure 4 |
| `06_alert_lifecycle.png` | Figure 5, the actual per-window PSI of `covariate_shift` at its shipped seed, with the state machine's decision under it |
| `07_performance_backfill.png` | Figure 6, the actual per-window PR-AUC of `concept_drift` at its shipped seed, over the labels each window held at first evaluation |
| `05_segment_vs_global.png` | Figure 7, the actual per-window trace of `segment_isolated` at its shipped seed |

File numbers are render order and never change; figure numbers are the case study's reading order.

The project card is not drawn here. It is `Wix/Homepage/covers-src/gen.py` (`art_drift`) in the website repository, exported at 2400 by 1800 with the cached Playwright headless shell so it keeps the shell the other cards share. Its two lines are the same per-window PSI values `segment_trace()` computes, pasted into `gen.py` with the copy date, and it prints no number.
