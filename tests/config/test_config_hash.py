from driftwatch.config.loader import compute_config_hash, load_model_config, load_profile


def test_same_config_hashes_the_same() -> None:
    model_config = load_model_config("example-model")
    profile = load_profile(model_config.profile)

    a = compute_config_hash(model_config, profile)
    b = compute_config_hash(model_config, profile)

    assert a == b


def test_different_profile_hashes_differently() -> None:
    model_config = load_model_config("example-model")
    aggressive = load_profile("aggressive")
    conservative = load_profile("conservative")

    a = compute_config_hash(model_config, aggressive)
    b = compute_config_hash(model_config, conservative)

    assert a != b
