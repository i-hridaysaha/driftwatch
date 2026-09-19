# Figure source

The case study's figures, drawn from the repository rather than typed.

    uv run --with matplotlib python figures/_src/figures.py   # every PNG into figures/
    uv run python figures/_src/sweeps.py                     # the two tables the case study quotes

| File | Job |
|---|---|
| `theme.py` | the portfolio figure kit's palette, z-order contract and primitives, copied from the kit with its imports tidied for this repository's linter |
| `figures.py` | one function per figure; the segment chart is recomputed window by window from `driftwatch.demo.generator` with `driftwatch.stats`, the test count comes from `pytest --collect-only`, and the three performance numbers are pinned with their `demo-verify` provenance |
| `sweeps.py` | forty clean seeds at the segment geometry (spurious opens per signal), the same at the pre-v1.0.0 geometry with PSI gated regardless of its floor, and a segment-share against shift-magnitude sweep |

| Output | Case study |
|---|---|
| `00_hero.png` | hero (CMS field) |
| `01_banner.png` | Figure 1 |
| `02_metrics_card.png` | Figure 2 |
| `03_architecture.png` | Figure 3, also embedded in the README |
| `04_two_clocks.png` | Figure 4 |
| `05_segment_vs_global.png` | Figure 5, the actual per-window trace of `segment_isolated` at its shipped seed |
