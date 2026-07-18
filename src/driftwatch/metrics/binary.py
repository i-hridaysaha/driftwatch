import math
from collections.abc import Sequence
from typing import Any

from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from driftwatch.metrics.registry import register


@register("pr_auc")
def pr_auc(y_true: Sequence[float], y_pred: Sequence[float], params: dict[str, Any]) -> float:
    value = float(average_precision_score(y_true, y_pred))
    if math.isnan(value):
        raise ValueError("pr_auc is undefined for this input (degenerate class distribution)")
    return value


@register("roc_auc")
def roc_auc(y_true: Sequence[float], y_pred: Sequence[float], params: dict[str, Any]) -> float:
    # sklearn warns and returns NaN rather than raising when y_true has a single
    # class; convert that into an explicit failure so callers can't mistake a
    # NaN for a real computed value.
    value = float(roc_auc_score(y_true, y_pred))
    if math.isnan(value):
        raise ValueError("roc_auc is undefined when only one class is present in y_true")
    return value


@register("precision_at_threshold")
def precision_at_threshold(
    y_true: Sequence[float], y_pred: Sequence[float], params: dict[str, Any]
) -> float:
    threshold = float(params["threshold"])
    predicted_labels = [1 if score >= threshold else 0 for score in y_pred]
    return float(precision_score(y_true, predicted_labels, zero_division=0))


@register("recall_at_threshold")
def recall_at_threshold(
    y_true: Sequence[float], y_pred: Sequence[float], params: dict[str, Any]
) -> float:
    threshold = float(params["threshold"])
    predicted_labels = [1 if score >= threshold else 0 for score in y_pred]
    return float(recall_score(y_true, predicted_labels, zero_division=0))


@register("precision_at_k")
def precision_at_k(
    y_true: Sequence[float], y_pred: Sequence[float], params: dict[str, Any]
) -> float:
    n = int(params["n"])
    ranked = sorted(zip(y_pred, y_true, strict=True), reverse=True)
    top_n = ranked[:n]
    if not top_n:
        return 0.0
    return sum(1 for _, label in top_n if label == 1) / len(top_n)
