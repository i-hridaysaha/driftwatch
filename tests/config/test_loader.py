import pytest

from driftwatch.config.loader import ProfileNotFoundError, load_profile


@pytest.mark.parametrize("name", ["aggressive", "conservative"])
def test_shipped_profiles_load(name: str) -> None:
    profile = load_profile(name)

    assert profile.name == name


def test_conservative_is_more_patient_than_aggressive() -> None:
    aggressive = load_profile("aggressive")
    conservative = load_profile("conservative")

    assert conservative.alerting.persistence_windows > aggressive.alerting.persistence_windows
    assert conservative.evaluation.min_window_size > aggressive.evaluation.min_window_size


def test_missing_profile_raises() -> None:
    with pytest.raises(ProfileNotFoundError):
        load_profile("does-not-exist")
