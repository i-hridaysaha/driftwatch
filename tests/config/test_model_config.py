import pytest

from driftwatch.config.loader import ModelConfigNotFoundError, load_model_config


def test_example_model_config_loads() -> None:
    config = load_model_config("example-model")

    assert config.model_id == "example-model"
    assert config.profile == "aggressive"
    assert config.prediction_type == "binary_classification"
    assert [f.name for f in config.schema_.features] == ["age", "region", "income"]
    assert config.segments.dimensions == ["region"]


def test_missing_model_config_raises() -> None:
    with pytest.raises(ModelConfigNotFoundError):
        load_model_config("does-not-exist")
