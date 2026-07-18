from collections.abc import Sequence
from dataclasses import dataclass

from scipy.spatial.distance import jensenshannon

from driftwatch.stats.binning import bucket_proportions
from driftwatch.stats.result import StatStatus


@dataclass(frozen=True)
class JSDResult:
    status: StatStatus
    value: float | None
    not_computable_reason: str | None = None


def jsd(
    baseline_values: Sequence[float], live_values: Sequence[float], edges: Sequence[float]
) -> JSDResult:
    """Jensen-Shannon divergence between a live prediction-score sample and
    the baseline, using the same frozen bucket edges as psi() (see
    driftwatch.stats.binning.compute_continuous_edges) so the two tests are
    directly comparable.

    Convention: this returns the JS *divergence* using log base 2, bounded
    [0, 1], not the JS *distance* (its square root, also common in the
    literature and what scipy.spatial.distance.jensenshannon returns by
    default). scipy's function is called with base=2 and the result is
    squared to undo scipy's implicit sqrt. Divergence in [0, 1] is the more
    common convention for drift-monitoring dashboards and is what
    PredictionScoreDriftConfig.jsd_threshold is written against; anyone
    cross-referencing against scipy directly needs to know this function
    does NOT return scipy's raw output.

    Unlike psi(), no epsilon floor is needed here: JS divergence is defined
    in terms of the mixture distribution M = (P + Q) / 2, which is positive
    wherever either P or Q is positive, so it never produces the log(0) or
    division-by-zero PSI's ratio-based formula is prone to.

    Returns status=NOT_COMPUTABLE with a reason -- never a bare nan -- if
    either sample is empty or there are no frozen edges to bin against, the
    same status/reason convention as psi() and
    driftwatch.db.models.PerformanceResult.
    """
    if not baseline_values:
        return JSDResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason="baseline sample is empty",
        )
    if not live_values:
        return JSDResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason="live sample is empty",
        )
    if not edges:
        return JSDResult(
            status=StatStatus.NOT_COMPUTABLE,
            value=None,
            not_computable_reason="no frozen bin edges available (e.g. all-null baseline feature)",
        )

    baseline_props = bucket_proportions(baseline_values, edges)
    live_props = bucket_proportions(live_values, edges)

    distance = jensenshannon(baseline_props, live_props, base=2)
    return JSDResult(status=StatStatus.COMPUTED, value=float(distance) ** 2)
