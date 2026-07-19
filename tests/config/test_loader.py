import pytest

from driftwatch.config.loader import ProfileNotFoundError, load_profile


@pytest.mark.parametrize("name", ["aggressive", "conservative"])
def test_shipped_profiles_load(name: str) -> None:
    profile = load_profile(name)

    assert profile.name == name


def test_conservative_is_more_patient_than_aggressive() -> None:
    aggressive = load_profile("aggressive")
    conservative = load_profile("conservative")

    assert (
        conservative.alerting.fire_persistence_windows
        > aggressive.alerting.fire_persistence_windows
    )
    assert (
        conservative.alerting.escalate_persistence_windows
        > aggressive.alerting.escalate_persistence_windows
    )
    assert (
        conservative.alerting.resolve_persistence_windows
        > aggressive.alerting.resolve_persistence_windows
    )
    assert conservative.evaluation.min_window_size > aggressive.evaluation.min_window_size

    # hysteresis gap is also meaningfully wider for the patient profile, not just
    # the thresholds themselves -- see configs/profiles/*.yaml for the reasoning
    aggressive_gap = (
        aggressive.drift_tests.continuous.psi_threshold
        - aggressive.drift_tests.continuous.psi_clear_threshold
    )
    conservative_gap = (
        conservative.drift_tests.continuous.psi_threshold
        - conservative.drift_tests.continuous.psi_clear_threshold
    )
    assert conservative_gap > aggressive_gap


def test_missing_profile_raises() -> None:
    with pytest.raises(ProfileNotFoundError):
        load_profile("does-not-exist")
