"""Sampling-noise floors for the effect sizes that gate alerts.

Every effect size here is stable as the sample grows -- that is why alerts
gate on them and not on p-values -- and every one of them reads noise, not
zero, under no shift at all, and that noise grows as the sample shrinks.
A threshold is only a threshold where a quiet window can reliably read as
clear, so the evaluation pipeline computes each statistic's null ceiling
(the level a quiet window stays under about 97.5 percent of the time) from
the two sample sizes before gating, and refuses to gate where that ceiling
exceeds the profile's clear threshold (driftwatch.evaluation.drift).

PSI's floor lives in driftwatch.stats.psi (a chi-square approximation, and
preferably a simulation on the registered baseline). The three below are
closed-form:

- KS D: Kolmogorov's asymptotic distribution. The 97.5th percentile of D
  under the null is c(0.025) * sqrt((n + m) / (n m)) with c from the
  Kolmogorov distribution (about 1.36).
- Cramer's V for a 2 x k table: V = sqrt(chi2 / N) with N = n + m, and
  chi2 ~ chi-square(k - 1) under independence, so the ceiling is
  sqrt(chi2_{k-1, 0.975} / N).
- Jensen-Shannon divergence (bits) between two empirical bucket
  distributions from one source: to second order 2 N_eff JSD(nats) ~
  chi-square(B - 1) with 1/N_eff = 1/n + 1/m, giving a ceiling of
  chi2_{B-1, 0.975} (1/n + 1/m) / (8 ln 2). Used only when a baseline
  carries no simulated table for the prediction score.
"""

import math

from scipy import stats as scipy_stats

NULL_CEILING_PERCENTILE = 0.975


def ks_null_ceiling(n_baseline: int, n_live: int) -> float:
    if n_baseline <= 0 or n_live <= 0:
        return math.inf
    c = float(scipy_stats.kstwobign.ppf(NULL_CEILING_PERCENTILE))
    return c * math.sqrt((n_baseline + n_live) / (n_baseline * n_live))


def cramers_v_null_ceiling(n_baseline: int, n_live: int, n_categories: int) -> float:
    if n_baseline <= 0 or n_live <= 0 or n_categories < 2:
        return math.inf
    chi2_ceiling = float(scipy_stats.chi2.ppf(NULL_CEILING_PERCENTILE, n_categories - 1))
    return math.sqrt(chi2_ceiling / (n_baseline + n_live))


def jsd_null_ceiling(n_baseline: int, n_live: int, n_buckets: int) -> float:
    if n_baseline <= 0 or n_live <= 0 or n_buckets < 2:
        return math.inf
    chi2_ceiling = float(scipy_stats.chi2.ppf(NULL_CEILING_PERCENTILE, n_buckets - 1))
    k = 1.0 / n_baseline + 1.0 / n_live
    return chi2_ceiling * k / (8.0 * math.log(2.0))
