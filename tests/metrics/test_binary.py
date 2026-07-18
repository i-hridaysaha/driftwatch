import math

from driftwatch.metrics.registry import get_metric

Y_TRUE = [0, 0, 0, 1, 1]
Y_PRED = [0.1, 0.4, 0.35, 0.8, 0.65]


def test_pr_auc_known_answer() -> None:
    value = get_metric("pr_auc")(Y_TRUE, Y_PRED, {})

    assert value == 1.0


def test_roc_auc_known_answer() -> None:
    value = get_metric("roc_auc")(Y_TRUE, Y_PRED, {})

    assert value == 1.0


def test_precision_at_threshold() -> None:
    value = get_metric("precision_at_threshold")(Y_TRUE, Y_PRED, {"threshold": 0.5})

    # predicted positive at >= 0.5: indices 3 (0.8, true=1) and 4 (0.65, true=1) -> 2/2
    assert value == 1.0


def test_recall_at_threshold() -> None:
    value = get_metric("recall_at_threshold")(Y_TRUE, Y_PRED, {"threshold": 0.9})

    # nothing scores >= 0.9, so zero true positives out of 2 actual positives
    assert value == 0.0


def test_precision_at_k_perfect_ranking() -> None:
    value = get_metric("precision_at_k")(Y_TRUE, Y_PRED, {"n": 2})

    # top 2 by score: 0.8 (true=1), 0.65 (true=1) -> 2/2
    assert value == 1.0


def test_precision_at_k_includes_a_false_positive() -> None:
    value = get_metric("precision_at_k")([0, 0, 0, 1, 1], [0.1, 0.9, 0.35, 0.8, 0.65], {"n": 2})

    # top 2 by score: 0.9 (true=0), 0.8 (true=1) -> 1/2
    assert math.isclose(value, 0.5)
