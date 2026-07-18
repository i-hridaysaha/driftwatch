from collections.abc import Callable, Sequence
from typing import Any

MetricFn = Callable[[Sequence[float], Sequence[float], dict[str, Any]], float]

_REGISTRY: dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def decorator(fn: MetricFn) -> MetricFn:
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_metric(name: str) -> MetricFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown metric: {name!r}")
    return _REGISTRY[name]


def available_metrics() -> list[str]:
    return sorted(_REGISTRY)
