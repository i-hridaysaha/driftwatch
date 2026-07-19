import pytest

from driftwatch.config.loader import ProfileNotFoundError, load_profile


@pytest.mark.parametrize("name", ["aggressive", "patient"])
def test_shipped_profiles_load(name: str) -> None:
    profile = load_profile(name)

    assert profile.name == name


def test_patient_is_more_patient_than_aggressive() -> None:
    aggressive = load_profile("aggressive")
    patient = load_profile("patient")

    assert (
        patient.alerting.fire_persistence_windows > aggressive.alerting.fire_persistence_windows
    )
    assert (
        patient.alerting.escalate_persistence_windows
        > aggressive.alerting.escalate_persistence_windows
    )
    assert (
        patient.alerting.resolve_persistence_windows
        > aggressive.alerting.resolve_persistence_windows
    )
    # min_window_size is deliberately NOT compared here -- unlike the old
    # conservative profile this replaced, patient keeps aggressive's window
    # size (both 50) and only differs on persistence/threshold philosophy,
    # not window cadence. See configs/profiles/patient.yaml.

    # hysteresis gap is also meaningfully wider for the patient profile, not just
    # the thresholds themselves -- see configs/profiles/*.yaml for the reasoning
    aggressive_gap = (
        aggressive.drift_tests.continuous.psi_threshold
        - aggressive.drift_tests.continuous.psi_clear_threshold
    )
    patient_gap = (
        patient.drift_tests.continuous.psi_threshold
        - patient.drift_tests.continuous.psi_clear_threshold
    )
    assert patient_gap > aggressive_gap


def test_missing_profile_raises() -> None:
    with pytest.raises(ProfileNotFoundError):
        load_profile("does-not-exist")
