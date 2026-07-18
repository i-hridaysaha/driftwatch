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
worth surfacing. No drift computation, scheduler, or
alerting yet — those land in later phases.

## Detection profiles

Two reusable YAML profiles ship in `configs/profiles/`, each a complete set
of evaluation window, drift test thresholds, multiple-comparison correction,
and alerting parameters. A monitored model's config (introduced once
baseline registration exists) references one of these by name.

- **`aggressive.yaml`** — for fast-moving, adversarial inputs (e.g. fraud,
  real-time bidding). Short 1-hour evaluation windows, loose significance
  thresholds (PSI 0.1, KS/chi-square α=0.10), and escalates after a single
  breaching window. Trades false positives for the earliest possible signal.
- **`conservative.yaml`** — for slow-moving population drift (e.g. credit
  risk, demand forecasting). Daily windows, strict thresholds (PSI 0.25,
  KS/chi-square α=0.01), and requires the signal to persist across 5
  consecutive windows before escalating. Prioritizes avoiding alert fatigue
  over catching the earliest signal.

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
