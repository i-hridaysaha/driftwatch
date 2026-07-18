from collections.abc import Sequence
from dataclasses import dataclass

from scipy import stats as scipy_stats

from driftwatch.stats.result import StatStatus


@dataclass(frozen=True)
class KSResult:
    status: StatStatus
    statistic: float | None
    p_value: float | None
    not_computable_reason: str | None = None


def ks(baseline_values: Sequence[float], live_values: Sequence[float]) -> KSResult:
    """Two-sample Kolmogorov-Smirnov test between a live sample and the baseline.

    `statistic` (the KS D-statistic, in [0, 1]) is the value alert thresholds
    are compared against -- see driftwatch.config.schema.ContinuousDriftConfig
    .ks_statistic_threshold. `p_value` is reported alongside for context but
    is never itself the trigger: at the sample sizes a production window
    accumulates, the null hypothesis (identical distributions) is essentially
    always false at the level of floating-point precision, and the p-value
    collapses toward zero regardless of whether the shift is large enough to
    matter operationally. A fixed effect-size threshold answers "is this
    shift big enough to act on"; a p-value only ever answers "is there
    enough data to detect that the two samples aren't bit-for-bit identical,"
    which at scale is always yes.

    Returns status=NOT_COMPUTABLE with a reason -- never a bare nan -- if
    either sample is empty, the same status/reason convention as psi() and
    driftwatch.db.models.PerformanceResult.
    """
    if not baseline_values:
        return KSResult(
            status=StatStatus.NOT_COMPUTABLE,
            statistic=None,
            p_value=None,
            not_computable_reason="baseline sample is empty",
        )
    if not live_values:
        return KSResult(
            status=StatStatus.NOT_COMPUTABLE,
            statistic=None,
            p_value=None,
            not_computable_reason="live sample is empty",
        )
    result = scipy_stats.ks_2samp(baseline_values, live_values)
    return KSResult(
        status=StatStatus.COMPUTED, statistic=float(result.statistic), p_value=float(result.pvalue)
    )
