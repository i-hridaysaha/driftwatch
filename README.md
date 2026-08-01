# driftwatch

A model-agnostic ML monitoring service: any model POSTs its predictions to a
shared schema, and driftwatch stores them, compares live feature and score
distributions against a frozen training baseline, computes performance once
ground-truth labels arrive (often weeks later), and raises disciplined alerts
when something degrades.

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.11-blue)
[![CI](https://github.com/i-hridaysaha/driftwatch/actions/workflows/ci.yml/badge.svg)](https://github.com/i-hridaysaha/driftwatch/actions/workflows/ci.yml)

📄 **Full write-up:** https://www.hridaysaha.com/projects-1/drift-watch%3A-ml-model-monitoring-service

The dashboard is a local Streamlit app, not a hosted demo — the reproducible
proof lives in the seeded scenarios below.

## Results

Correctness is demonstrated by four seeded scenarios. Each injects a specific
fault (or none), and a CI-run verifier asserts numerically that driftwatch
detected exactly that fault — and nothing else.

| Scenario | Injected fault | Expected outcome | Verified |
|---|---|---|---|
| `clean` | none | no alerts | ✅ no false positives |
| `covariate_shift` | global input-distribution shift | drift alert fires | ✅ |
| `segment_isolated` | shift confined to one segment (APAC) | segment alert fires, global stays quiet | ✅ |
| `concept_drift` | label noise; inputs stable, only visible after labels backfill | PR-AUC degrades and escalates | ✅ |

> **Data note:** every number here comes from *synthetic, staged* scenarios
> against a real Postgres — not a benchmark. Parameters were tuned to produce
> a legible signal on a dashboard, not to claim detection sensitivity on real
> production data. The value on show is the **pipeline and its evaluation
> discipline**, not the absolute figures.

Real `demo-verify` output (this run, printed by CI on every push):

```
$ driftwatch demo-verify segment_isolated
[PASS] no unexpected alerts: none
[PASS] global age PSI stays under the fire threshold throughout: max=0.094, fire_threshold=0.25
[PASS] APAC segment breaches, fires, and (once the shift ends) resolves: ['escalated', 'resolved']

$ driftwatch demo-verify concept_drift
[PASS] no feature or prediction-score drift (inputs stay stable): none
[PASS] global PR-AUC degrades and escalates after label_noise: status=escalated
[PASS] PR-AUC healthy before, degraded after (final backfilled values): early_mean=0.874, late_mean=0.386, clear=0.65, fire=0.5
[PASS] degradation only visible after backfill: 57 windows initially not_computable for lack of labels
```

Regenerating any scenario reproduces `early_mean`/`late_mean` to the last
digit — that is what "deterministic" means here in practice.

## Why this is non-trivial

- **Delayed ground truth is first-class.** Predictions and labels arrive
  separately, often weeks apart. Labels backfill asynchronously keyed on
  `prediction_id` and retroactively recompute the performance of the window
  they belong to — the `concept_drift` scenario is only detectable because of
  this.
- **Effect size drives alerts, never the p-value.** At production window
  sizes a p-value testing "are these two samples identical" is essentially
  always significant regardless of whether the shift matters. Alerting gates
  on KS D-statistic, Cramér's V, PSI, and JSD (all bounded, sample-size
  independent); p-values are reported for context only.
- **Segment-level drift, not just global.** Aggregate stats hide a failure
  confined to one geography or category. `segment_isolated` proves a
  15%-weighted segment can breach while the global metric stays quiet.
- **Alert fatigue is treated as a design failure.** Hysteresis (separate fire
  and clear thresholds), consecutive-window persistence gates, and
  Benjamini-Hochberg correction scoped to one model × one window keep a
  boundary-hugging statistic from flapping.
- **`not_computable` is never a silent gap.** A metric that is mathematically
  undefined for a window (e.g. `roc_auc` when every label so far is one class)
  is recorded with a reason and rendered as an explicit marker — a missing
  point on a monitoring chart reads as "all clear," which is the most
  dangerous confusion such a dashboard can produce.

## Approach

```
model → POST predictions/labels (FastAPI) → Postgres (append-only, keyed on prediction_id)
                                                      │
     scheduler windows by event time ────────────────┤
                                                      ▼
   drift core: PSI · KS · chi-square · JSD + BH correction   (pure, deterministic)
   performance: pr_auc / precision / recall / rmse / mae      (once labels arrive)
                                                      │
                        stateful alerting (open → escalated → resolved, hysteresis)
                                                      │
                              read-only Streamlit dashboard (URL = full chart spec)
```

Per-model behaviour — feature schema, drift thresholds, segments, alert
sensitivity — comes entirely from YAML config, never from branching code. The
engine knows nothing about the domain of the model it watches.

## Data

Synthetic, generated by a pure seeded function (`src/driftwatch/demo/`) — no
I/O, no wall-clock, drawing from one `numpy` generator in a fixed call order,
so regenerating months later produces byte-identical rows. All four scenarios
use a `Beta(0.4, 0.4)` prediction-score distribution — a demo-legibility
choice (it gives the `label = bernoulli(score)` mechanic enough rank
correlation for `concept_drift` to visibly degrade), not a claim that real
scores look bimodal. Baseline bin edges are frozen at registration; live
values outside the baseline range land in explicit overflow buckets rather
than being clipped.

## Detection & evaluation

- **Drift** — continuous features via PSI + KS; categoricals via chi-square +
  Cramér's V; prediction scores via JSD. Chosen because they are effect-size
  measures that stay informative at large sample sizes, where significance
  tests saturate.
- **Performance** — `pr_auc` for the binary-classification demo models
  (precision/recall-style metrics matter more than accuracy under imbalance);
  `rmse`/`mae` available for regression. Which metrics run for a model is
  declared in its profile YAML, not branched in code.
- **Two profiles ship** — `aggressive.yaml` (1-hour windows, fires on one
  breaching window; for fast, adversarial inputs) and `patient.yaml` (needs 3
  consecutive breaches, wide hysteresis gap; for slow population drift).
- **Reproducibility** — the generator seed lives in each scenario's YAML; the
  same `(range, window duration)` always yields the same UTC-aligned window
  boundaries, computed by a scheduler tick or a CLI backfill years apart.

## Run it yourself

```bash
uv sync                                        # pinned deps from uv.lock
docker compose up -d postgres                  # or point DATABASE_URL at any Postgres 16
uv run alembic upgrade head                    # apply migrations

uv run driftwatch demo concept_drift           # reset DB, load, evaluate — one command
uv run driftwatch demo-verify concept_drift    # assert it produced what it claims

uv run streamlit run src/driftwatch/dashboard/app.py   # explore the result
```

Or bring the whole stack up (Postgres + API on `:8000` + scheduler +
dashboard on `:8501`):

```bash
docker compose up --build
```

Full test suite (needs a Postgres reachable at `DATABASE_URL`):

```bash
uv run pytest      # 204 passed
```

## Repo map

```
src/driftwatch/
  api/          FastAPI ingestion — baselines, predictions, labels
  stats/        PSI, KS, chi-square, JSD, Benjamini-Hochberg (pure functions)
  evaluation/   per-window drift + performance computation
  scheduler/    APScheduler, event-time windowing, evaluate_window()
  alerting/     stateful alerts, hysteresis, pluggable notifications
  metrics/      name-keyed metric registry (pr_auc, rmse, ...)
  dashboard/    read-only Streamlit app
  demo/         deterministic scenario generator + verifier
  db/, config/  SQLAlchemy models · YAML profile/model loader
configs/        profiles/ · models/ · scenarios/
alembic/        migrations
tests/          43 files, 204 tests
```

## Limitations & next steps

- **Synthetic data only.** The scenarios are staged demonstrations, not an
  empirical benchmark of detection sensitivity.
- **Dashboard read-only is enforced per-transaction** (`SET TRANSACTION READ
  ONLY`), not by a dedicated read-only DB role or replica — provisioning that
  is real deployment work a later phase would do.
- **Docker Compose is CI-verified, not run locally** — the build machine has
  no Docker, so a CI smoke test builds the images and polls `/health`; that
  liveness check has no DB dependency, so it doesn't exercise migrations.
- **Notifications ship as logging + generic webhook only** — Slack/email are
  thin adapters a real deployment builds on a webhook receiver, deliberately
  not special-cased here.

Deeper methodology (statistical-choice rationale, alerting lifecycle, dashboard
state model) lives in the [case study](https://www.hridaysaha.com/projects-1/drift-watch%3A-ml-model-monitoring-service).

## License

MIT © Hriday Saha
