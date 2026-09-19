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
| `clean` | none | no alerts | ✅ no alerts, zero significant readings |
| `covariate_shift` | global input-distribution shift | drift alert fires; a one-window blip does not | ✅ |
| `segment_isolated` | shift confined to one segment (APAC, 15% of traffic) | both segment alerts open, escalate and resolve; global stays quiet | ✅ global max 0.042, segment max 1.23 |
| `concept_drift` | label noise; inputs stable, only visible after labels backfill | PR-AUC degrades and escalates | ✅ 0.874 before, 0.387 after |
| `burst_shift` (aggressive profile) | a four-window burst, then a one-window blip | the burst opens on its first window, escalates, resolves; the blip opens and resolves too | ✅ |

Every scenario is generated and verified in CI at its shipped seed and at
two others (`--seed 7`, `--seed 11`), one matrix job each.

> **Data note:** every number here comes from *synthetic, staged* scenarios
> against a real Postgres — not a benchmark. Parameters were tuned to produce
> a legible signal on a dashboard, not to claim detection sensitivity on real
> production data. The value on show is the **pipeline and its evaluation
> discipline**, not the absolute figures.

Real `demo-verify` output (this run, printed by CI on every push):

```
$ driftwatch demo-verify segment_isolated
[PASS] no unexpected alerts: none
[PASS] global age PSI stays under the fire threshold throughout: max=0.042, fire_threshold=0.25
[PASS] APAC age psi alert opens, escalates, and (once the shift ends) resolves: status=resolved, opened at 1.188, escalated at 1.122
[PASS] APAC age ks alert opens, escalates, and (once the shift ends) resolves: status=resolved, opened at 0.418, escalated at 0.429

$ driftwatch demo-verify concept_drift
[PASS] no feature or prediction-score drift (inputs stay stable): none
[PASS] global PR-AUC degrades and escalates after label_noise: status=escalated
[PASS] PR-AUC healthy before, degraded after (final backfilled values): early_mean=0.874, late_mean=0.387, clear=0.65, fire=0.5
[PASS] degradation only visible after backfill: 60 windows initially not_computable for lack of labels

$ driftwatch demo-verify burst_shift
[PASS] no unexpected alerts: none
[PASS] global age psi: the burst opens on its FIRST breaching window, escalates, and resolves: opened at window 20: True, status=resolved, opened at 0.327, escalated at 0.375
[PASS] global age psi: the one-window blip opens (aggressive does not suppress it) and resolves without escalating: status=resolved, never escalated
[PASS] global age ks: the burst opens on its FIRST breaching window, escalates, and resolves: opened at window 20: True, status=resolved, opened at 0.239, escalated at 0.244
[PASS] global age ks: the one-window blip opens (aggressive does not suppress it) and resolves without escalating: status=resolved, never escalated
```

Regenerating any scenario reproduces every number above to the last digit;
that is what "deterministic" means here in practice. Two cheaper experiments
that need no database live in `figures/_src/sweeps.py`: forty clean seeds at
the segment scenario's geometry open zero spurious alerts on either PSI or
KS, and a sweep of segment share against shift magnitude shows where the
global reading stays quiet while the segment fires.

Before v1.0.0 the segment scenario ran at 500 predictions a window, which
gave the 15% segment about 75 rows against a 225-row baseline slice. At
that volume PSI's own sampling noise (null mean about 0.20) sat in the dead
zone between the patient profile's 0.15 clear and 0.25 fire thresholds: the
segment breached on noise one quiet window in four, and after the shift
ended its PSI alert could never assemble five clear windows, so it stayed
escalated while the KS alert resolved. The verifier of the time passed on
the union of the two. Since v1.0.0 the verifier checks each alert's whole
journey and the pipeline refuses to gate on a statistic where its noise
ceiling exceeds the clear threshold; since v1.1.0 that floor is simulated
on the registered baseline itself for PSI and JSD, closed-form for KS D and
Cramér's V, and every scenario is sized so that none of them trips it (the
first run of the aggressive scenario did, on a fifteen-category feature's
Cramér's V, which is how that floor got added).

## Why this is non-trivial

- **Delayed ground truth is first-class.** Predictions and labels arrive
  separately, often weeks apart. Labels backfill asynchronously keyed on
  `prediction_id` and retroactively recompute the performance of the window
  they belong to — the `concept_drift` scenario is only detectable because of
  this.
- **Effect size drives alerts, never the p-value.** At production window
  sizes a p-value testing "are these two samples identical" is essentially
  always significant regardless of whether the shift matters. Alerting gates
  on the KS D-statistic, Cramér's V and JSD (each in [0, 1]) and on PSI
  (unbounded); a real shift of a given size reads the same on any of them
  whatever the window size. p-values are stored for context only, and
  Benjamini-Hochberg correction of them is informational: it never decides
  what fires.
- **Effect sizes are stable as volume grows, not as it shrinks.** Under no
  shift at all every one of them reads sampling noise, and the noise grows
  as the sample shrinks: PSI's is roughly (bins − 1)(1/n_baseline + 1/n_live),
  which at a few dozen rows sits above every fire threshold shipped here.
  The pipeline computes each statistic's null ceiling per window and refuses
  to gate where it exceeds the profile's clear threshold, recording the row
  as `not_computable` with the floor in the reason rather than as a number
  that would flap. PSI's and JSD's ceilings are simulated on the registered
  baseline at registration (`driftwatch.stats.psi.simulate_psi_null`, stored
  per feature and per segment slice); KS D's and Cramér's V's are closed
  form (`driftwatch.stats.floors`).
- **The request flags, the scheduler works.** `POST /labels` inserts the
  labels and sets `performance_stale` on the windows they touch, in one
  transaction. The scheduler's next tick claims each flagged window with a
  single `UPDATE`, recomputes it, re-evaluates its alerts and notifies. A
  month of labels costs the request one update per window, not 720
  recomputes and webhook calls, and performance lags a batch by at most one
  tick.
- **A notification that failed is retried, a bounded number of times.** A
  round in which a channel raised leaves the alert's delivered status behind
  its real status; each tick retries it until it succeeds or five attempts
  are spent, after which the error stays on the row. An alert that opens
  during a webhook outage still reaches someone once the webhook is back.
- **Segment-level drift, not just global.** Aggregate stats hide a failure
  confined to one geography or category. `segment_isolated` proves a
  15%-weighted segment can breach while the global metric stays quiet. A
  categorical feature is never tested inside a segment defined by that same
  feature: every row in `region=EU` has `region == EU`, so that slot is
  `not_configured`, not a permanent `not_computable` alert.
- **Alert fatigue is treated as a design failure.** Hysteresis (separate fire
  and clear thresholds, with the gap between them a dead zone that breaks a
  streak in either direction) and consecutive-window persistence gates for
  opening, escalating and resolving keep a boundary-hugging statistic from
  flapping. Persistence is the only control on the roughly two dozen
  effect-size gates a window evaluates.
- **`not_computable` is never a silent gap.** A metric that is mathematically
  undefined for a window (e.g. `roc_auc` when every label so far is one class)
  is recorded with a reason and rendered as an explicit marker — a missing
  point on a monitoring chart reads as "all clear," which is the most
  dangerous confusion such a dashboard can produce.

## Approach

![Predictions, labels and baseline registrations enter through one FastAPI endpoint into an append-only Postgres store. A scheduler drives the drift path, label ingestion drives the performance path, both feed an alert state machine, results and alert rows are written back to the store, notifications leave on status transitions, and a read-only dashboard reads the store directly.](figures/03_architecture.png)

```
model → POST predictions / labels / baseline (FastAPI, schema-validated, X-API-Key) → Postgres (append-only)
                                                      │
     scheduler tick: windows past the watermark ─────┤──── labels flag the windows they touch
                                                      ▼                          ▼
   drift path: PSI · KS · chi-square · JSD, sealed once    performance path: recomputed by the next tick
   (each refused below its sampling-noise floor)           (pr_auc / precision / recall)
                                                      │                          │
                        stateful alerting (open → escalated → resolved, hysteresis, dead zone)
                                                      │
                  drift results, performance results and alert rows written back to Postgres
                        │                                              │
   notifications (log, webhook; on transition, retried per tick)   read-only dashboard (reads Postgres)
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
- **Two profiles ship, and both are exercised** — `aggressive.yaml` (1-hour
  windows, fires on one breaching window; for fast, adversarial inputs) by
  the `burst_shift` scenario, and `patient.yaml` (needs 3 consecutive
  breaches, wide hysteresis gap; for slow population drift) by the other
  four.
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
uv run pytest      # 219 passed
```

Set `API_KEY` before starting the API to require `X-API-Key` on every write
endpoint (`/health` stays open); left unset, the API is open and logs a
warning at startup. Registering a baseline for a model that already has one
is refused with 409 unless the request says `replace_active: true`, because
that write changes what every future drift number is measured against.

Redraw the case study's figures from the generator and the repository's
own statistics (no database needed; `matplotlib` is pulled in for the run):

```bash
uv run --with matplotlib python figures/_src/figures.py
```

## Repo map

```
src/driftwatch/
  api/          FastAPI ingestion — baselines, predictions, labels
  stats/        PSI, KS, chi-square, JSD, Benjamini-Hochberg, and each one's null floor (pure functions)
  evaluation/   per-window drift + performance computation
  scheduler/    APScheduler, event-time windowing, evaluate_window()
  alerting/     stateful alerts, hysteresis, pluggable notifications
  metrics/      name-keyed metric registry (pr_auc, rmse, ...)
  dashboard/    read-only Streamlit app
  demo/         deterministic scenario generator + verifier
  db/, config/  SQLAlchemy models · YAML profile/model loader
configs/        profiles/ · models/ · scenarios/
alembic/        migrations
tests/          31 test files, 219 tests
figures/        the case study's figures and the script that draws them from the generator
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
  not special-cased here. Retry is bounded to five rounds a tick apart; an
  outage longer than that leaves the error on the alert row and nothing else.
- **Performance lags a label batch by up to one scheduler tick** (five
  minutes by default), the price of keeping recomputation out of the
  request path. A retried notification round re-runs every channel, so a
  log line can repeat while the webhook catches up.
- **Authentication is one shared key.** `API_KEY` protects every write, and
  it is a single secret for every caller; per-model credentials and rotation
  are deployment work.
- **No model version column.** `config_hash` on every window makes a profile
  change detectable after the fact; a retrained model behind the same
  `model_id` is not.
- **The sampling-noise floors refuse rather than correct.** A signal a
  profile asks for at a volume that cannot carry it is a silence alert, not
  a corrected statistic; the operator widens the window or drops the test.

Deeper methodology (statistical-choice rationale, alerting lifecycle, dashboard
state model) lives in the [case study](https://www.hridaysaha.com/projects-1/drift-watch%3A-ml-model-monitoring-service).

## License

MIT © Hriday Saha
