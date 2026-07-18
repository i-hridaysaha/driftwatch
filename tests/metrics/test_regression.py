import math

from driftwatch.metrics.registry import get_metric


def test_rmse_known_answer() -> None:
    value = get_metric("rmse")([1.0, 2.0, 3.0], [1.0, 2.0, 5.0], {})

    # errors: 0, 0, 2 -> mean squared error = 4/3 -> rmse = sqrt(4/3)
    assert math.isclose(value, math.sqrt(4 / 3))


def test_mae_known_answer() -> None:
    value = get_metric("mae")([1.0, 2.0, 3.0], [1.0, 2.0, 5.0], {})

    # errors: 0, 0, 2 -> mean absolute error = 2/3
    assert math.isclose(value, 2 / 3)
