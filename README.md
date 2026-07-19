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
shipped detection profiles (see "Detection profiles" below), Docker Compose
wiring for Postgres + API + dashboard, and CI running ruff/mypy/pytest.

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
made explicit.

**Phase 5: scheduler and evaluation pipeline.** A single `evaluate_window()`
function that both the scheduler and a new CLI backfill command call —
computing drift for every feature (global and per-segment) and then
performance for whatever labels have arrived, idempotently. See "Evaluation
pipeline" below for the event-time-vs-ingestion-time, the
drift-final-vs-performance-revisable lifecycle split, and the
idempotency-by-constraint decisions this phase made explicit.

**Phase 6: alerting.** An alert is a stateful row (`open` -> `escalated` ->
`resolved`), not an event per window — sustained drift across 20 windows is
one row with an updated `last_seen`, not 20. See "Alerting" below for the
hysteresis, evidence-snapshot, not-computable-as-its-own-alert-type,
retroactive-recompute, aggregation-cap, and notification-idempotency
decisions this phase made explicit.

**Phase 7: Streamlit dashboard.** Read-only, enforced at the transaction
level (`SET TRANSACTION READ ONLY`), with every chart's full configuration —
model, date range, feature, test method, segment, window, metric — carried
in the URL query string, so a chart can be regenerated identically from its
link alone. See "Dashboard" below for the three-state-rendering,
deterministic-color, and SQL-aggregation decisions this phase made explicit.

**Phase 8 (this commit): demo data generator.** Four declarative YAML
scenarios (`clean`, `covariate_shift`, `segment_isolated`, `concept_drift`),
a fully deterministic, seeded, pure generator, and a post-generation
verification script that asserts each scenario numerically produced what it
claims — run in CI for all four. `driftwatch demo <scenario>` is one command
from an empty database to a fully evaluated dataset. See "Demo data
generator" below for the determinism, staggered-label, and
scenario-tuning-via-config-not-results decisions this phase made explicit.

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

## Evaluation pipeline

- **Windows are keyed on event time (`predicted_at`), never ingestion time.**
  A prediction ingested late still lands in the window it actually belongs
  to. All window arithmetic is UTC, and boundaries are aligned to the Unix
  epoch (not to whenever a model happened to be registered or a scheduler
  tick happened to fire) — see `driftwatch.scheduler.windowing
  .compute_window_boundaries`. The same `(range, window duration)` always
  produces exactly the same boundaries, computed by a scheduler tick today
  or a CLI backfill next year.
- **A window has two separate lifecycles, not one.** This distinction is
  load-bearing, not cosmetic — scenario 4 of the demo generator (phase 8:
  concept drift that's only visible once labels backfill, with input
  distributions staying stable throughout) depends on it:
  - **Drift is computed once and is final.** From the predictions that exist
    in the window at `window_end + evaluation.watermark`, never recomputed
    for that window again.
  - **Performance stays open indefinitely.** It's recomputed every time a
    new label arrives for a prediction in that window — weeks or months
    later, with no watermark or finality of its own — tracked via
    `label_watermark` (below). `recompute_performance_for_window` is what
    does this, called directly by the label ingestion endpoint, completely
    independent of whether the window's drift has already run.

  Nothing in this codebase uses the unqualified word "sealed" — every
  occurrence says "drift watermark elapsed" or similar, specifically because
  the whole window is never sealed, only drift is.
- **Late-arriving predictions: drift watermark, not retroactive
  re-evaluation.** A window's drift becomes eligible for its one-time
  computation at `window_end + evaluation.watermark`
  (`driftwatch.scheduler.windowing.is_drift_watermark_elapsed`). A
  straggler prediction landing after that point is still stored — predictions
  are never rejected — but is permanently excluded from that window's drift
  stats, and increments `EvaluationWindow.late_prediction_count` with a
  logged warning (see below), so a rising late-arrival rate is visible as
  the data pipeline problem it is, not silently absorbed. This is a
  deliberate choice, not an oversight: labels arriving weeks late is this
  system's core premise (see "Delayed ground truth is first-class" above),
  but a *prediction* arriving late relative to its own event time is a
  pipeline hiccup, not a fundamental feature of the domain — and a
  monitoring dashboard is better served by a drift chart that never
  silently moves again once published than by one that's occasionally more
  complete but can revise under the viewer.
- **Late predictions are counted, not dropped silently.** Every prediction
  ingested after its window's drift has already been evaluated increments
  that window's `late_prediction_count` and logs a warning naming the
  window and the prediction — see `driftwatch.api.routes.predictions`. This
  is a data point about the ingestion pipeline's health, not just the
  model's: a climbing rate means predictions are arriving later relative to
  their own event time than the configured watermark assumes.
- **Idempotency is enforced by the database, not application logic.** A
  window is "claimed" by inserting its `EvaluationWindow` row inside a
  SAVEPOINT; `uq_evaluation_windows_model_range` raises `IntegrityError` if
  another process already claimed the same window concurrently, and that's
  caught and treated as a no-op — safe even with two evaluation runs racing
  on the same window, not just against a well-behaved caller. `DriftResult`
  has its own functional unique index (`evaluation_window_id, feature_name,
  test_method, COALESCE(segment_dimension, ''), COALESCE(segment_value,
  '')`, since Postgres treats `NULL != NULL` in plain unique constraints) so
  a re-run can never duplicate rows even if the claim step were somehow
  bypassed. `PerformanceResult` deliberately has no such constraint — it's
  designed to hold multiple rows per window as labels backfill over time —
  so its idempotency instead comes from the scheduler only ever calling
  `recompute_performance_for_window` once per window's *initial* evaluation;
  every later call is a legitimate, intentional revision, not a duplicate.
- **Every window records what it was evaluated against.** `baseline_id`
  (which baseline version), `config_hash` (a fingerprint of the resolved
  `ModelConfig` + `Profile` at evaluation time — see
  `driftwatch.config.loader.compute_config_hash`), and `label_watermark`
  (the latest label `received_at` incorporated into the most recent
  performance computation). Together these mean a drift timeline stays
  interpretable across a model retrain or a config change, and a dashboard
  can tell whether more labels have arrived since a window's performance was
  last computed without re-running anything.
- **Minimum sample size, from config, gates drift and performance alike.**
  Below `evaluation.min_window_size` (globally) or `segments
  .min_segment_size` (per segment), every configured test/metric gets a
  `not_computable` row with a reason instead of a number computed from too
  few samples to mean anything — the same status/reason convention used
  throughout, applied here to stop segment-level charts on tiny samples from
  reading as confident signal when they're really just noise.
- **The CLI evaluates historical ranges through the identical code path.**
  `uv run driftwatch evaluate --model-id <id> --start <iso> --end <iso>
  [--force]` calls the exact same `evaluate_window()` the scheduler calls —
  the only difference is that a CLI backfill bypasses the watermark
  entirely (an explicit historical range means "evaluate these windows now,"
  not "whatever happens to be ready"). This is what lets the demo generator
  (phase 8) seed 30+ windows of deterministic history in one shot, something
  a scheduler that only ever asks "what window is 'now' in" cannot do.
  `--force` deletes and re-evaluates windows that already exist in the given
  range, for byte-identical demo-data regeneration.
- **APScheduler misfire handling is explicit, not left at the library
  default.** `misfire_grace_time=None` (a late tick always eventually runs,
  never gets silently dropped past some deadline), `coalesce=True` (a
  backlog of missed ticks collapses into one run, since each tick already
  scans the full unevaluated backlog — see `find_drift_ready_unevaluated_windows`
  — so running it N times back to back after an outage would just repeat
  the same discovery query for nothing), `max_instances=1` (avoid two ticks
  doing redundant concurrent work; correctness doesn't depend on this, since
  `evaluate_window` is already safe under concurrent execution, it's purely
  to not waste work). See `driftwatch.scheduler.run`.

## Alerting

- **An alert is a stateful row with a lifecycle, not an event per window.**
  `Alert.status` moves through `open` -> `escalated` -> `resolved`. Its
  identity is `(model_id, kind, signal_name, feature_name, segment_dimension,
  segment_value)` — `kind` is `drift`, `performance`, or `not_computable`,
  and `signal_name` is a test method (`psi`, `ks`, ...) or a metric name
  (`pr_auc`, `rmse`, ...). Sustained drift across 20 consecutive windows
  updates ONE row's `last_seen_window_id`/`last_seen_at`, never inserts a
  second row — a partial unique index enforces at most one non-resolved,
  non-aggregate row per identity (`uq_alerts_active_identity`, `COALESCE`-
  based since Postgres treats `NULL != NULL`). A *resolved* alert doesn't
  block a fresh row when the same signal fires again later — that's a new
  episode, kept as its own row so the history isn't lost.
- **No `PENDING` state, and no counter persisted before an alert exists.**
  `driftwatch.alerting.engine` recomputes the relevant streak from scratch
  every time, scanning bounded recent `DriftResult`/`PerformanceResult`
  history for that exact signal (`driftwatch.alerting.streaks
  .fetch_drift_history` / `fetch_performance_history`) and walking backward
  from the most recent entry until the classification changes
  (`current_streak`). An `Alert` row is only created once a signal actually
  reaches `open`; there's nothing to maintain or clean up for a streak that
  never gets that far. How far back that rescan looks is itself an explicit,
  validated config value — `AlertingConfig.history_scan_windows` — not
  derived from the persistence counts it has to satisfy: a scan bound that's
  merely inferred as "the max of the persistence counts" is tautologically
  always exactly enough, which can never catch a real misconfiguration.
  Making it a separate field lets a config-load validator
  (`_history_scan_covers_persistence_with_margin`) require it to exceed the
  largest configured persistence count by a margin, and fail loudly at
  startup if it doesn't — a profile where the rescan bound can't see far
  enough back to satisfy its own escalate/resolve gate would otherwise leave
  alerts silently, permanently unable to fire.
- **Hysteresis via a three-way classifier, not the existing
  `is_significant`.** `DriftResult.is_significant` (fire-threshold-only,
  used for that row's own informational display) is left untouched.
  Alerting reclassifies every window's statistic independently as
  `breach` / `dead_zone` / `clear` using BOTH the fire and clear thresholds
  (`*_clear_threshold` on each drift-test config, `clear_threshold` on a
  `MetricSpec`). A `dead_zone` reading — between clear and fire — breaks
  whichever streak was building, the same as a value on the opposite side
  would; that's what stops a statistic hovering at the boundary from
  flapping an alert open and resolved every window. Resolving requires
  `resolve_persistence_windows` CONSECUTIVE clear windows, symmetric with
  `fire_persistence_windows` for opening — any interruption (a breach or a
  dead-zone reading) resets that count to zero, it doesn't just pause it.
- **`not_computable` is never "no drift."** It's its own `AlertKind`, with
  its own persistence gate (`not_computable_persistence_windows`) completely
  independent of the drift/performance fire count. A feature that's been
  uncomputable for that many consecutive windows raises a `not_computable`
  alert alongside whatever `drift`-kind alert may or may not also exist for
  the same signal — silence about a broken feature is exactly the failure
  mode this service exists to prevent. Unlike drift/performance, a
  `not_computable` alert resolves the instant the signal becomes computable
  again, no clear-persistence needed: "did it compute" is a clean binary
  signal with nothing to flap around, unlike a continuous statistic near a
  threshold.
- **Every alert freezes its evidence at the moment it opens — and again at
  the moment it escalates.** `evidence_statistic`, `evidence_threshold`,
  `evidence_baseline_id`, `evidence_config_hash`, and `evidence_window_id`
  are set once, when a row transitions to `open`, and never touched again —
  not even when the model's config changes later. `driftwatch.alerting
  .test_engine.test_evidence_survives_config_change` proves this directly:
  re-evaluating a later window under a changed profile leaves a prior
  alert's evidence untouched, so it still explains why IT fired, under the
  rules in effect then. Escalation gets its own second snapshot —
  `escalation_evidence_statistic`/`_threshold`/`_baseline_id`/
  `_config_hash`/`_window_id` — frozen once at the `open` -> `escalated`
  transition, same discipline, same never-touched-again guarantee. An alert
  that opened at a PSI of 0.11 and escalated at 0.42 can say both: the open
  snapshot alone would only ever be able to explain why it first fired, not
  why it got worse. See `test_escalation_evidence_is_a_separate_snapshot_
  from_open_evidence`.
- **Performance alerts and retroactive label backfill.** A window can
  recompute its performance many times as labels trickle in (see "Evaluation
  pipeline" above). `fetch_performance_history` deduplicates to the LATEST
  `PerformanceResult` per window before the streak is ever computed, so ten
  recomputes of one window contribute exactly one entry to the streak, not
  ten — a window whose metrics degrade after labels arrive still alerts, but
  never re-alerts purely because it recomputed again. The same `(model,
  kind=performance, metric_name, segment)` identity from the point above is
  what makes this safe: a degraded window updates the one open alert's
  `last_seen`, it doesn't spawn another.
- **Alert volume is capped per run, with the overflow counted, not hidden.**
  If more new alerts would open in one evaluation run than
  `AlertingConfig.max_alerts_per_run` (e.g. 50 segments breach at once), the
  individual rows are rolled into one aggregate `Alert`
  (`is_aggregate=True`, `aggregated_signal_count`, `aggregated_signals` —
  the list of what got rolled up) instead of flooding the table and every
  notification channel with N separate rows. The cap only throttles the
  RATE of new opens in a single run; already-open individual alerts from
  prior runs keep updating, escalating, and resolving normally regardless of
  it. The aggregate itself resolves once a later run's new-open count drops
  back to or under the cap. Known scope limit, stated rather than engineered
  around: `not_computable` aggregation doesn't distinguish whether the
  overflow came from drift tests or performance metrics, since both share
  that one `AlertKind` — a simultaneous overflow from both in the same run
  is rare enough not to warrant a further identity split.
- **Notification delivery is pluggable, and idempotent by design.**
  `driftwatch.alerting.notifications.NotificationChannel` is a two-method
  interface (`notify(alert, event)`); `LoggingNotificationChannel` (always
  on) and `WebhookNotificationChannel` (generic JSON POST, added if
  `Settings.alert_webhook_url` is configured) are the two implementations —
  deliberately not a Slack or email integration, which are thin adapters a
  real deployment builds on top of a webhook receiver, not something this
  service should special-case. Re-notification policy: `notify_if_needed`
  compares the alert's current `status` against `last_notified_status` and
  only dispatches on an actual TRANSITION (opened, escalated, resolved) —
  never on steady-state continuation of an already-notified status. A still-
  open alert does not re-notify on every scheduler tick just because its
  `last_seen` advanced; calling `notify_if_needed` repeatedly with an
  unchanged status is a no-op after the first call.
- **A failing notification channel never fails the evaluation run it's
  attached to.** A hanging or erroring webhook is an expected operational
  condition, not a bug — `notify_if_needed` calls each channel in isolation,
  catches any exception, logs it, and records it on the alert
  (`last_notification_error`, `last_notification_error_at`) instead of
  letting it propagate. `WebhookNotificationChannel` deliberately does not
  catch its own errors anymore — the centralized catch in `notify_if_needed`
  is what makes this true for every channel uniformly, not just the shipped
  webhook one. There's no automatic retry: a failed delivery isn't
  re-attempted on the next touch of an already-notified status, matching
  the no-renotify-on-steady-state policy above; `last_notification_error`
  is what an operator (or a log/metrics pipeline) watches instead.
  `test_raising_channel_leaves_alert_transaction_committed` proves the part
  that actually matters operationally: a raising channel still leaves the
  alert's state transition committed, not rolled back.

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
  and fires on a single breaching window
  (`fire_persistence_windows: 1`), escalating after 3 more consecutive
  breaches. Its hysteresis gap is narrow (PSI clears at 0.07, only 0.03
  below the 0.1 fire line) since fast-moving inputs are expected to cross
  back and forth quickly. Trades false positives for the earliest possible
  signal.
- **`patient.yaml`** — for slow-moving population drift on a
  binary-classification model, where a single bad window is noise, not a
  signal. Hourly windows (like `aggressive`, unlike an earlier
  daily-window `conservative` profile this replaced — see below), wide
  effect-size thresholds — PSI 0.25 (the "significant shift" convention
  boundary), KS D-statistic 0.3, and Cramer's V 0.3 (Cohen's "medium"
  effect) — and requires 3 consecutive breaching windows to fire
  (`fire_persistence_windows: 3`) and 8 to escalate. Its hysteresis gap is
  wide (PSI clears at 0.15, a full 0.1 below the 0.25 fire line) so a
  single noisy window drifting back toward baseline doesn't reset a real,
  slow-building trend. Precision/recall-style performance metrics
  (`pr_auc`, matching `aggressive`'s), not regression error metrics.
  Prioritizes avoiding alert fatigue over catching the earliest signal.

Persistence counts and hysteresis gap widths are not incidental — they're
as deliberate a difference between the two profiles as the drift
thresholds themselves; see "Alerting" above for how both are used.

An earlier third profile, `conservative.yaml`, shipped alongside these two
through phase 7: daily windows and regression error metrics (`rmse`/`mae`)
for slow-moving, continuous-label domains like demand forecasting. Phase 8
removed it — every demo scenario needed a binary-classification model
(`rmse`/`mae` don't apply to one), and once `patient.yaml` was written to
fill that gap, nothing in the repo exercised `conservative.yaml` against
real data anymore, only shallow config-loading tests. An unverified profile
sitting in a repo about monitoring *correctness* is worse than not shipping
it at all, so it was deleted rather than kept as an untested example —
`patient.yaml` is what its "wait for a real signal" design point looks like
for the kind of model this project actually demonstrates.

Note these thresholds are not the same numbers the old `ks_alpha`/
`chi_square_alpha` significance levels used before this phase renamed them
to effect-size thresholds — a 0.05 significance level and a 0.05 KS
statistic threshold mean entirely different things, so each value above was
re-derived from what it actually gates on, not carried over.

The right choice depends on how fast the underlying population can actually
shift and how expensive a false alarm is versus a missed one — not on a
fixed rule of thumb.

## Dashboard

`src/driftwatch/dashboard/app.py` is a read-only Streamlit app
(`streamlit run src/driftwatch/dashboard/app.py`, or the `dashboard` service
in `docker-compose.yml`) built entirely on the tables the evaluation
pipeline already writes — `drift_results`, `performance_results`, `alerts`,
`evaluation_windows` — never on raw `predictions`/`labels`.

- **Read-only is enforced by Postgres, not just by "we didn't write an
  INSERT."** `driftwatch.dashboard.db.read_only_session` issues `SET
  TRANSACTION READ ONLY` as the first statement of every session and never
  commits, so an accidental write anywhere in the dashboard raises
  immediately at the database rather than silently succeeding.
- **A chart's URL is its full specification.** Model, date range, feature,
  test method, segment, window, metric, and view all live in
  `st.query_params` (see `driftwatch.dashboard.state`), not
  `st.session_state` — the same link reopened months later reproduces the
  same chart. The default date range itself comes from `MIN(window_start)`/
  `MAX(window_end)` in `evaluation_windows` for the selected model
  (`queries.default_time_range`), never `datetime.now()`.
- **Three states, always distinguishable, never a silent gap.** Every
  drift/performance chart LEFT JOINs the full grid of (window x signal)
  against whatever result rows actually exist: `computed` (a real value,
  colored breach/normal), `not_computable` (an explicit attempt was
  recorded with a reason, shown on hover, rendered as a triangle in its own
  lane below the chart's zero line), `not_configured` (the window WAS
  evaluated, but not this exact signal — a test not in the profile's
  configured methods for this feature's dtype, or a segment value with
  zero live predictions that window — a grey diamond in a second lane),
  and `missing` (rendered as a black cross in a third lane). None of the
  three "empty" states ever just breaks the line and silently disappears.
- **`missing` is its own state, deliberately the most alarming one on the
  chart, and is never conflated with `not_configured`.** A window that was
  never evaluated at all (scheduler downtime, a gap in a historical
  backfill) is a completely different fact from a window that WAS
  evaluated and simply had nothing configured to say about this signal —
  the dashboard reconciles the actual EvaluationWindow rows against the
  full EXPECTED window grid for the visible range
  (`driftwatch.dashboard.queries._expected_window_grid`, built from the
  same `driftwatch.scheduler.windowing.compute_window_boundaries` the real
  scheduler and CLI backfill use, so "what windows should exist" is never
  reimplemented a second time) rather than just listing whatever
  EvaluationWindow rows happen to exist. Without this, a genuinely missing
  window contributes zero rows to the chart's data and the line connecting
  its neighbors draws straight across the gap with no break and no marker
  — indistinguishable from uninterrupted quiet monitoring, which is the
  most dangerous confusion a monitoring dashboard can produce. This
  applies to the drift timeline, segment view, and performance timeline
  alike.
- **Thresholds are reference lines with values, not just a color.** Every
  drift and performance chart draws its fire and clear thresholds as
  labelled dashed rules, positioned by the actual visible time domain
  (`driftwatch.dashboard.charts._threshold_rules` anchors the label to the
  rightmost timestamp in view via the chart's own temporal scale) rather
  than a hardcoded pixel offset, which — found out the hard way — is not
  reliably inside a chart whose declared width includes axis-label space
  Vega-Lite subtracts from the actual drawable plot.
- **Every temporal encoding is explicitly UTC.** Vega-Lite's default
  temporal scale renders axis ticks (and a tooltip's formatted timestamp)
  in the *viewer's browser timezone*, not the data's own — since every
  timestamp in this project is UTC (`driftwatch.db.models`), left
  unpinned, the identical chart would render different x-axis tick labels
  depending on where it's opened, which breaks this project's own
  "same data produces a pixel-identical chart on every run" requirement.
  `driftwatch.dashboard.charts._utc_x` / `_utc_tooltip` force `scale=utc`
  and a `utc:`-prefixed format string everywhere a `:T` field is encoded.
- **Color is a hash of the name, not the row's position.** Feature and
  segment colors come from `driftwatch.dashboard.colors.stable_color`, an
  `md5`-based hash — not Python's built-in `hash()`, which is
  process-randomized for strings unless `PYTHONHASHSEED` is pinned, and
  would silently break "the same feature is always the same color" across
  restarts. A feature's color is identical whether it's the only one
  plotted or one of ten.
- **The segment view shares one y-scale across every facet.** Global and
  each segment value are laid out as Vega-Lite facets of one chart, with
  the y-domain computed once from the *entire* multi-segment frame before
  faceting (`charts._with_plot_columns`) rather than per facet — otherwise
  a facet with a much larger statistic would silently rescale its own
  panel and no longer be visually comparable to the others in the same row.
- **Suppressed alerts are a genuine replay, not a separate calculation.**
  The "suppressed (never reached persistence)" table reuses
  `driftwatch.alerting.streaks.classify_drift`/`classify_performance` — the
  exact functions the real alerting engine calls — against date-bounded
  history, so this view can never quietly drift from what the engine would
  actually have decided. The grouping into episodes
  (`driftwatch.alerting.reporting.find_suppressed_episodes`) is a small,
  independently unit-tested pure function.
- **Alert history shows and filters by the data's own time only — never
  wall-clock evaluation time.** `Alert.first_opened_at`/`escalated_at`/
  `resolved_at`/`last_seen_at` record when the evaluation run actually
  happened in real time — correct for what they're for (notification
  currency), but a historical backfill evaluates a whole date range of
  windows in one fast burst "now", which would put today's date in every
  one of those columns right next to a data event from months earlier,
  with nothing in the table telling a reader the two apart. The table
  never surfaces those four columns at all: `fetch_alert_history` instead
  joins `evidence_window_id` / `escalation_evidence_window_id` /
  `last_seen_window_id` to `evaluation_windows` and returns
  `opened_window_start`/`opened_window_end`/`escalated_window_end`/
  `last_seen_window_end` — self-describing column names, all on the same
  time axis as every chart around them, and also what the range filter
  itself uses.
- **Known limitation, stated rather than engineered around:** this
  connects with the same application database role as the rest of the
  service, not a dedicated read-only role or replica — the transaction-level
  guard above is what actually makes "no writes" true regardless, and
  provisioning a separate role is real infra work a later phase (or a real
  deployment) would do, not a gap in this phase's read-only guarantee.

## Demo data generator

`src/driftwatch/demo/` produces fully deterministic evaluation history for
the dashboard and README screenshots — every published number in this
project comes from a real `driftwatch demo <scenario>` run against a real
Postgres, never a hand-edited result.

**These scenarios are staged demonstrations, not a benchmark.** Every
scenario parameter — event magnitude, segment weights, sample sizes, and
in particular the prediction-score distribution — was chosen and tuned to
produce a clear, legible signal on a dashboard screenshot, and none of it
is an empirical claim about this service's detection sensitivity on real
production data. The score distribution all four scenarios use,
`Beta(0.4, 0.4)` (U-shaped: scores cluster near 0 and 1), is a concrete
example: it was chosen because a moderate, unimodal score distribution like
`Beta(2, 5)` — closer to what a real fraud or risk model's score
distribution might look like — makes the generator's `label = bernoulli
(score)` mechanic produce labels with almost no rank correlation to the
score at all (PR-AUC caps out around 0.5 regardless of injected noise; see
`concept_drift` below), leaving nothing for `concept_drift`'s label-noise
event to visibly degrade. `Beta(0.4, 0.4)` is not a claim that real fraud
scores look bimodal — it is a demo-generator implementation detail, chosen
to make one specific chart legible, and should not be read as anything
else.

```bash
uv run python -m driftwatch.cli demo clean            # reset DB, load, evaluate — one command
uv run python -m driftwatch.cli demo-verify clean      # assert it produced what it claims
```

`driftwatch demo <scenario>` resets the *entire* database (every app table,
not just rows for that scenario's `model_id` — see `driftwatch.demo.build
.reset_database`), registers the baseline, ingests predictions, ingests
labels on their staggered delay schedule, and runs `evaluate_range` over the
full window span, in one call. Because the reset is global, only one
scenario's data can live in the database at a time — `demo-verify` checks
this explicitly (`EvaluationWindow` count for the scenario's `model_id` must
match `scenario.total_windows`) rather than letting an empty or
wrong-scenario database pass every other check vacuously.

- **Deterministic by construction, not by convention.** `generate_scenario_data`
  is a pure function — no I/O, no wall-clock — that draws from one
  `numpy.random.Generator` seeded once from `scenario.seed`, in a fixed call
  order (baseline, then window 0..N-1 in order; within each window:
  segments, then features in declared order, then prediction score, then
  label offsets/noise/delay). Every timestamp is computed from
  `scenario.start_date`, a fixed date declared in the YAML, never
  `datetime.now()` — regenerating months later produces byte-identical rows
  and identical dashboard chart URLs. Enforced by
  `tests/demo/test_generator_determinism.py`, which runs every shipped
  scenario twice and asserts the resulting data is equal, not just
  documented as an intention.
- **Windows must agree with the profile, or fail loudly at build time.**
  `evaluate_range` groups predictions into windows using the *profile's*
  `evaluation.window`, but the generator groups them into windows using the
  *scenario's own* declared `window` field. If those two ever disagree — an
  easy mistake, since nothing else connects them — the generator's "window
  index N" (what every event's `start_window` and every verification
  assertion is expressed in) silently stops corresponding to what
  `evaluate_range` actually evaluates. `build_scenario` checks this
  explicitly (`_check_window_matches_profile`) and raises before writing
  anything, rather than producing a quietly wrong dataset.
- **Labels arrive staggered, in two phases, for a real before/after-backfill
  split.** Each label's delay is drawn from a configurable lognormal
  (`scenario.label_delay`). Labels with `delay_hours <= early_cutoff_hours`
  are inserted *before* the single `evaluate_range` call, so a window's
  initial performance reflects only them; labels with longer delay are
  inserted *after*, triggering `recompute_performance_for_window` for each
  touched window. `concept_drift` deliberately sets `early_cutoff_hours`
  short relative to the median delay, so most windows are initially
  `not_computable` for lack of labels and the performance degradation is
  only visible once backfill completes — mirroring the drift-final vs.
  performance-revisable split described above, not a separate mechanism.
- **A structural, expected `not_computable` pattern, documented rather than
  hidden.** All four scenarios segment by `region`, which is also a
  regular monitored feature — a real, common combination (the ingestion
  API requires a segment dimension to also be a declared feature; see
  `driftwatch.api.validation.extract_segment_values`). But evaluating a
  categorical feature's drift *within* a segment defined by that same
  feature is structurally degenerate: every row in, say, the `region=EU`
  segment has `region == 'EU'` by construction, so chi-square can never see
  more than one category there and always reports `not_computable`. This
  is an inherent property of `_evaluate_all_features`'s "evaluate every
  declared feature at every segment level, including the segment-defining
  feature itself" design, not a scenario bug — `driftwatch/demo/verify.py`
  excludes exactly this pattern (`_is_structural_self_segment_alert`) from
  every "no unexpected alerts" check, rather than tuning it away or
  special-casing each scenario.
- **Scenarios are tuned via config, never via results.** Getting all four
  scenarios to produce exactly their claimed signal — and nothing else —
  took real iteration against a live database: segment population weights,
  `predictions_per_window`, and `baseline_size` all affect how much
  sampling noise a fixed effect-size threshold sees at the segment level,
  and a scenario that looked clean at one sample size sometimes wasn't at
  another (see `segment_isolated`'s injected shift magnitude, tuned down
  from an initial 5σ — which blew through the *global* PSI threshold even
  confined to a 15%-weighted segment — to 1σ, verified numerically to stay
  under threshold globally while still breaching locally). Every number in
  every scenario's alert history comes from a real evaluation run;
  `driftwatch demo-verify <scenario>` is what makes that a checked
  guarantee instead of a claim.

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

# in another terminal: runs the scheduler as its own long-lived process
uv run python -m driftwatch.scheduler.run

# evaluate an arbitrary historical range (same code path as the scheduler)
uv run driftwatch evaluate --model-id example-model \
  --start 2026-01-01T00:00:00Z --end 2026-02-01T00:00:00Z
```

## Running with Docker Compose

```bash
docker compose up --build
```

Brings up Postgres, the API (`localhost:8000`), a `scheduler` service (same
image as the API, running `driftwatch.scheduler.run` as its own process so
scaling the API never runs multiple schedulers by accident), and a dashboard
stub (`localhost:8501`). Docker isn't available on the machine this was
built on, so this hasn't been run locally — but CI's
`docker-compose-smoke-test` job runs `docker compose up --build` on every
push and polls `/health` before tearing down, on a runner that does have
Docker. That job only proves the images build and the API process starts
and responds — `/health` has no DB dependency, so it doesn't exercise
migrations or any DB-backed endpoint; that's a stated scope limit of the
smoke test, not an oversight.
