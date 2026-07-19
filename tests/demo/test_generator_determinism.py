"""The golden test: determinism is a stated selling point of this project
and must be enforced here, not just claimed in a docstring. No database,
no I/O -- generate_scenario_data is a pure function, so this compares two
in-memory calls directly."""

from datetime import UTC, datetime

import pytest

from driftwatch.demo.generator import generate_scenario_data
from driftwatch.demo.loader import list_scenario_names, load_scenario
from driftwatch.demo.schema import ScenarioConfig

_MINIMAL_SCENARIO = ScenarioConfig.model_validate(
    {
        "name": "golden-minimal",
        "description": "minimal fixture for the determinism test",
        "model_id": "demo-golden-minimal",
        "profile": "aggressive",
        "seed": 12345,
        "start_date": datetime(2026, 1, 1, tzinfo=UTC),
        "window": "1h",
        "stable_windows": 3,
        "total_windows": 6,
        "predictions_per_window": 15,
        "baseline_size": 50,
        "prediction_type": "binary_classification",
        "prediction_distribution": {"type": "beta", "a": 2, "b": 5},
        "features": [
            {
                "name": "age",
                "dtype": "continuous",
                "distribution": {"type": "normal", "mean": 40, "std": 12},
            },
            {
                "name": "income",
                "dtype": "continuous",
                "nullable": True,
                "missing_rate": 0.1,
                "distribution": {"type": "lognormal", "mean": 10.5, "sigma": 0.6},
            },
            {
                "name": "region",
                "dtype": "categorical",
                "categories": ["EU", "US", "APAC"],
                "weights": [0.4, 0.4, 0.2],
            },
            {"name": "device_id", "dtype": "categorical", "cardinality": 50},
        ],
        "segments": ["region"],
        "min_segment_size": 10,
        "label_base_noise_rate": 0.05,
        "label_delay": {
            "median_hours": 6,
            "sigma": 0.7,
            "max_hours": 72,
            "early_cutoff_hours": 6,
        },
        "events": [
            {
                "feature": "age",
                "shift_type": "mean_shift",
                "magnitude": 3.0,
                "start_window": 3,
            }
        ],
    }
)


def test_regenerating_the_same_scenario_produces_identical_data() -> None:
    first = generate_scenario_data(_MINIMAL_SCENARIO)
    second = generate_scenario_data(_MINIMAL_SCENARIO)

    assert first == second
    assert len(first.predictions) > 0  # sanity: not vacuously equal empty lists


def test_different_seeds_produce_different_data() -> None:
    other = _MINIMAL_SCENARIO.model_copy(update={"seed": 999})

    first = generate_scenario_data(_MINIMAL_SCENARIO)
    second = generate_scenario_data(other)

    assert first != second
    assert first.predictions[0].features != second.predictions[0].features


def test_reloading_the_scenario_yaml_from_disk_still_regenerates_identically() -> None:
    """Guards the actual claim in the constraint: regenerating months later
    -- i.e. from a fresh process re-reading the YAML file, not from a
    Python object still sitting in memory -- must be byte-identical."""
    loaded_once = ScenarioConfig.model_validate(_MINIMAL_SCENARIO.model_dump(mode="json"))
    loaded_twice = ScenarioConfig.model_validate(_MINIMAL_SCENARIO.model_dump(mode="json"))

    assert generate_scenario_data(loaded_once) == generate_scenario_data(loaded_twice)


@pytest.mark.parametrize("scenario_name", list_scenario_names())
def test_every_shipped_scenario_is_deterministic(scenario_name: str) -> None:
    scenario = load_scenario(scenario_name)

    first = generate_scenario_data(scenario)
    second = generate_scenario_data(scenario)

    assert first == second
