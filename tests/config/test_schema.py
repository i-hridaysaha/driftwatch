import pytest
from pydantic import ValidationError

from driftwatch.config.schema import AlertingConfig


def _alerting(**overrides: int) -> dict[str, int]:
    base = {
        "fire_persistence_windows": 1,
        "escalate_persistence_windows": 3,
        "resolve_persistence_windows": 2,
        "not_computable_persistence_windows": 2,
        "history_scan_windows": 6,
        "max_alerts_per_run": 10,
    }
    base.update(overrides)
    return base


def test_valid_alerting_config_loads() -> None:
    config = AlertingConfig.model_validate(_alerting())
    assert config.history_scan_windows == 6


def test_escalate_below_fire_rejected() -> None:
    with pytest.raises(ValidationError, match="escalate_persistence_windows"):
        AlertingConfig.model_validate(
            _alerting(fire_persistence_windows=5, escalate_persistence_windows=3)
        )


def test_history_scan_windows_exactly_at_largest_persistence_rejected() -> None:
    """No margin at all: the escalate gate (3) could theoretically be reached
    by a fetch of exactly 3 rows, but the validator requires margin beyond
    the bare minimum -- see AlertingConfig._MIN_HISTORY_SCAN_MARGIN."""
    with pytest.raises(ValidationError, match="history_scan_windows"):
        AlertingConfig.model_validate(
            _alerting(escalate_persistence_windows=3, history_scan_windows=3)
        )


def test_history_scan_windows_below_margin_rejected() -> None:
    with pytest.raises(ValidationError, match="history_scan_windows"):
        AlertingConfig.model_validate(
            _alerting(escalate_persistence_windows=8, history_scan_windows=9)
        )


def test_history_scan_windows_at_exact_margin_accepted() -> None:
    config = AlertingConfig.model_validate(
        _alerting(escalate_persistence_windows=8, history_scan_windows=10)
    )
    assert config.history_scan_windows == 10
