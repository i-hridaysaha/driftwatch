import math
from collections.abc import Sequence
from dataclasses import dataclass

from driftwatch.stats.binning import bucket_proportions
from driftwatch.stats.result import StatStatus

EPSILON = 1e-4
"""Floor applied to every bucket proportion before computing PSI's ratio/log
terms.

PSI's per-bucket term is (live_prop - baseline_prop) * ln(live_prop /
baseline_prop). A bucket with zero observations in either sample makes that
term undefined (division by zero, or ln(0)) -- but a zero-count bucket is
common and meaningful in practice: it's exactly what happens in the
under/overflow buckets before any drift occurs, and in any sparsely
populated interior bin.

1e-4 is the floor used by most published PSI implementations and by the
articles the formula is usually cited from; using the same value keeps our
PSI numbers roughly comparable to PSI values reported by other tools. The
floor is applied to BOTH baseline and live proportions, uniformly, not just
to zeros -- clipping only exact zeros would create a discontinuity right at
the epsilon boundary, where a bucket with a true proportion of 0.00001 would
be treated completely differently from one with exactly 0. Flooring
everything at 1e-4 means the floor only ever changes the result for buckets
that are already negligibly small, and never for a bucket carrying real
signal (real bucket proportions in a 10-12 bucket scheme are on the order of
0.05-0.1, several orders of magnitude above the floor).
"""


@dataclass(frozen=True)
class PSIResult:
    status: StatStatus
    value: float | None
    not_computable_reason: str | None = None


def psi(
    baseline_values: Sequence[float], live_values: Sequence[float], edges: Sequence[float]
) -> PSIResult:
    """Population Stability Index between a live sample and the frozen baseline.

    `edges` must be the baseline's own frozen bin edges (see
    driftwatch.stats.binning.compute_continuous_edges), computed once at
    baseline registration and never recomputed per window -- comparing PSI
    values across windows only means anything if every window is binned
    against the same reference.

    Returns status=COMPUTED with value=0.0 for identical distributions, and
    status=NOT_COMPUTABLE with a reason -- never an exception, and never a
    bare nan -- when there isn't enough data to compute a meaningful value.
    This mirrors driftwatch.db.models.PerformanceResult's status/reason
    pattern from the ingestion API: a value that can't be computed is always
    a visible, explained row, not a silent gap.
    """
    if not baseline_values:
        return PSIResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason="baseline sample is empty",
        )
    if not live_values:
        return PSIResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason="live sample is empty",
        )
    if not edges:
        return PSIResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason="no frozen bin edges available (e.g. all-null baseline feature)",
        )

    baseline_props = bucket_proportions(baseline_values, edges)
    live_props = bucket_proportions(live_values, edges)

    total = 0.0
    for baseline_prop, live_prop in zip(baseline_props, live_props, strict=True):
        baseline_floored = max(baseline_prop, EPSILON)
        live_floored = max(live_prop, EPSILON)
        total += (live_floored - baseline_floored) * math.log(live_floored / baseline_floored)
    return PSIResult(status=StatStatus.COMPUTED, value=total)
