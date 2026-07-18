# driftwatch

A model-agnostic ML monitoring and drift detection service. Any model sends
predictions to it via a defined schema; it stores them, compares live feature
and prediction distributions against a registered training baseline, computes
performance once ground-truth labels arrive (often weeks later), and alerts
when something degrades.

This is under active development. See the phase notes below for what's built
so far.

## Design decisions

- **Model-agnostic core.** The engine doesn't know anything about the domain
  of the model it's watching. Per-model behavior (feature schema, drift test
  thresholds, alerting sensitivity) comes entirely from YAML config, never
  from branching code.
- **Delayed ground truth is first-class.** Predictions and labels arrive
  separately, often weeks apart. Storage is keyed on prediction ID; labels
  backfill asynchronously and trigger retroactive recomputation of
  performance for the window the prediction belongs to.
- **Segment-level drift, not just global.** Aggregate stats can hide a
  failure confined to one geography or category. Slice dimensions are
  configurable per model.
- **Alert discipline.** Multiple-comparison correction across feature tests,
  and a signal must persist across several consecutive evaluation windows
  before it escalates. Alert fatigue is treated as a design failure, not
  something to tune away later.
- **Effect size drives alerts, not statistical significance.** At production
  window sizes, a p-value testing "are these two samples identical" is
  essentially always significant regardless of whether the shift is large
  enough to act on — significance stops carrying information exactly when
  you have enough data for it to matter. See "Statistical methods" below.

## Stack

Python 3.11+, FastAPI, PostgreSQL, SQLAlchemy + Alembic, Pydantic v2,
APScheduler, Streamlit, scikit-learn, Docker Compose, pytest, ruff, mypy.
Dependencies are managed with [uv](https://docs.astral.sh/uv/).

## Status

**Phase 1: repo skeleton.** A FastAPI app with a single `/health` endpoint,
the config-loading layer (Pydantic v2 schema + loader for per-profile YAML),
two shipped detection profiles (`aggressive` / `conservative` — see below),
Docker Compose wiring for Postgres + API + dashboard, and CI running
ruff/mypy/pytest.

**Phase 2: data model and migrations.** SQLAlchemy models and an Alembic
migration for the full storage layer — `models`, `baselines` +
`baseline_records` (the reference sample a live window is compared against),
`predictions` and `labels` as separate append-only tables keyed on a
caller-supplied `prediction_id`, `evaluation_windows` (records which
baseline version and config hash it was evaluated against, so a drift
timeline stays interpretable across a retrain), `drift_results`, and
`performance_results` (append-only, since a label backfill can retroactively
revise a window's metrics).

**Phase 3 (this commit): ingestion API.** Three endpoints: baseline
registration, prediction ingestion, and label backfill — all under
`/models/{model_id}/...`, all batched. Predictions are immutable: a
resent `prediction_id` with an identical payload is a no-op, a resent
`prediction_id` with a *different* payload is a 409. Labels are
corrections, not immutable events: a different payload under an existing
`prediction_id` is accepted as a new label row, not a conflict. A label
batch collects every evaluation window it touches and recomputes each
exactly once via a standalone `recompute_performance_for_window`
function — the same function the scheduler (phase 5) will call. Adds
`ModelConfig` (per-model YAML: feature schema, segments, which profile to
use) and a name-keyed metrics registry (`pr_auc`, `precision_at_threshold`,
`recall_at_threshold`, `precision_at_k`, `roc_auc`, `rmse`, `mae`) — which
metrics run for a given model is declared entirely in that model's profile
YAML, not branched on in code. Every `performance_results` row carries a
`status` (`computed` / `not_computable`, with a reason) — a metric that's
mathematically undefined for a window (e.g. `roc_auc` when every label
received so far is the same class) is recorded as a visible gap with a
cause, not silently dropped; a missing row would render as a chart gap
that reads as "no problem," when a degenerate window is itself a signal
worth surfacing.

**Phase 4 (this commit): statistical core.** Pure, deterministic functions
in `src/driftwatch/stats/` — PSI, KS, chi-square, Jensen-Shannon divergence,
and Benjamini-Hochberg correction — each with known-answer tests (identical
inputs give zero, a hand-derived case gives a hand-verified value, and
degenerate inputs like an empty window or a single-value feature return a
defined, explained result — never an exception, and never a bare `nan`).
PSI, KS, chi-square, and JSD all report `status` (`computed` /
`not_computable`, with a reason) on every result, the same convention
`PerformanceResult` already uses, so the system has one consistent story
about values that can't be computed. Baseline registration now computes and
freezes real PSI/JSD bin edges and chi-square categories (previously an
empty placeholder pending this phase); each continuous feature's binning
also records `effective_bins`, the actual post-deduplication bin count,
since skewed or low-cardinality data can collapse most of the requested 10
quantile bins into far fewer. See "Statistical methods" below for the
epsilon, effect-size, and multiple-comparison-scope decisions this phase
made explicit. No scheduler or alerting yet — those land in later phases.

## Statistical methods

- **PSI empty-bucket epsilon: `1e-4`.** PSI's per-bucket term is
  `(live_prop - baseline_prop) * ln(live_prop / baseline_prop)`, which is
  undefined when either proportion is exactly zero — and a zero-count bucket
  is normal, not exceptional (it's exactly what every under/overflow bucket
  looks like before any drift happens). `1e-4` is the floor most published
  PSI implementations use, so our values stay roughly comparable to PSI
  numbers reported by other tools. It's applied to *every* bucket's
  proportion (via `max(proportion, 1e-4)`), not just exact zeros, so there's
  no discontinuity right at the epsilon boundary — see the docstring on
  `driftwatch.stats.psi.EPSILON` for the full reasoning.
- **PSI/JSD bin edges are frozen at baseline registration, forever.** Edges
  are quantiles of the baseline sample, computed once and stored on the
  `Baseline` row; every future evaluation window bins its live data against
  those same edges. A live value below the baseline's minimum or above its
  maximum lands in an explicit overflow/underflow bucket rather than being
  clipped into the nearest real bin — that bucket's baseline proportion is
  always exactly zero by construction, so any live data landing there is
  unambiguously flagged as something the baseline never produced, not
  silently absorbed or errored on. Chi-square applies the same principle to
  categoricals: a category unseen in the baseline is bucketed as "unseen,"
  not dropped or rejected.
- **Effect size drives alerting, never the p-value.** `ContinuousDriftConfig
  .ks_statistic_threshold` gates on the KS D-statistic (bounded [0, 1]);
  `CategoricalDriftConfig.cramers_v_threshold` gates on Cramer's V, the
  standard chi-square effect-size measure (also bounded [0, 1], and — unlike
  the raw chi-square statistic — independent of sample size and category
  count). Both p-values are still computed and reported on every
  `DriftResult` row, for context, but neither is ever compared against a
  threshold to decide significance. The reason: a p-value answers "is there
  enough data to be confident these two samples aren't bit-for-bit
  identical," and at the sample sizes a production evaluation window
  accumulates, the answer to that question is essentially always yes,
  independent of whether the shift is one worth acting on. Gating on p-value
  at scale means every window eventually alerts, regardless of severity —
  effect size is the only thing left that still carries information once
  you have enough data for the significance test to saturate. PSI and JSD
  don't have this problem in the first place: both are already pure effect-
  size measures, with no p-value at all.
- **Benjamini-Hochberg scope: one model, one window, no broader.**
  `driftwatch.stats.correction.benjamini_hochberg` must only ever be called
  across the feature-level p-values (KS, chi-square) produced by a single
  model's single evaluation window. Pooling across models, or across time
  windows for the same model, would silently change what "the family of
  tests" means and make the correction's false-discovery-rate guarantee
  meaningless — a model with 50 features evaluated over 30 windows is 30
  independent 50-comparison experiments, not one 1500-comparison experiment.
  Enforced by convention and documented on the function itself, since
  Python can't express "this list must come from exactly one evaluation" in
  the type system.
- **JSD convention: divergence, not distance; log base 2.**
  `driftwatch.stats.jsd.jsd` returns the Jensen-Shannon *divergence* in
  `[0, 1]` (squaring `scipy.spatial.distance.jensenshannon`'s output, called
  with `base=2`), not scipy's own return value, which is the *distance*
  (the divergence's square root). This is the more common convention on
  drift-monitoring dashboards and is what `PredictionScoreDriftConfig
  .jsd_threshold` is written against — worth knowing if cross-referencing
  against scipy directly.

## Detection profiles

Two reusable YAML profiles ship in `configs/profiles/`, each a complete set
of evaluation window, drift test thresholds, multiple-comparison correction,
and alerting parameters. A monitored model's config (introduced once
baseline registration exists) references one of these by name.

- **`aggressive.yaml`** — for fast-moving, adversarial inputs (e.g. fraud,
  real-time bidding). Short 1-hour evaluation windows, small effect-size
  thresholds — PSI 0.1 (the standard "moderate shift" convention boundary),
  KS D-statistic 0.15 (a modest but real shift, large enough not to be
  sampling noise even at this profile's small 50-row minimum window), and
  Cramer's V 0.1 (Cohen's 1988 "small" effect for one degree of freedom) —
  and escalates after a single breaching window. Trades false positives for
  the earliest possible signal.
- **`conservative.yaml`** — for slow-moving population drift (e.g. credit
  risk, demand forecasting). Daily windows, larger effect-size thresholds —
  PSI 0.25 (the "significant shift" convention boundary), KS D-statistic 0.3
  (real, unambiguous separation — a large 500+ row window makes even modest
  shifts easy to detect, so the bar has to be higher than the aggressive
  profile's), and Cramer's V 0.3 (Cohen's "medium" effect) — and requires
  the signal to persist across 5 consecutive windows before escalating.
  Prioritizes avoiding alert fatigue over catching the earliest signal.

Note these thresholds are not the same numbers the old `ks_alpha`/
`chi_square_alpha` significance levels used before this phase renamed them
to effect-size thresholds — a 0.05 significance level and a 0.05 KS
statistic threshold mean entirely different things, so each value above was
re-derived from what it actually gates on, not carried over.

The right choice depends on how fast the underlying population can actually
shift and how expensive a false alarm is versus a missed one — not on a
fixed rule of thumb.

## Running locally

The test suite includes migration/schema tests that need a real Postgres
reachable at `DATABASE_URL` (defaults to
`postgresql+psycopg://driftwatch:driftwatch@localhost:5432/driftwatch`).
Bring one up with `docker compose up postgres` (or point `DATABASE_URL` at
any Postgres 16 instance with a `driftwatch` database).

```bash
uv sync
uv run pytest
uv run alembic upgrade head
uv run uvicorn driftwatch.api.main:app --reload
```

## Running with Docker Compose

```bash
docker compose up --build
```

Brings up Postgres, the API (`localhost:8000`), and a dashboard stub
(`localhost:8501`).
