from collections.abc import Sequence
from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error

from driftwatch.metrics.registry import register


@register("rmse")
def rmse(y_true: Sequence[float], y_pred: Sequence[float], params: dict[str, Any]) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


@register("mae")
def mae(y_true: Sequence[float], y_pred: Sequence[float], params: dict[str, Any]) -> float:
    return float(mean_absolute_error(y_true, y_pred))
