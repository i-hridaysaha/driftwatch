"""Derives the real ModelConfig the evaluation pipeline reads
(driftwatch.config.loader.load_model_config) from a scenario's own
declarative schema/segments/profile -- the scenario YAML is the single
source of truth for feature schema and segment dimensions (per this
phase's own constraint), so the model config file is a generated
artifact of it, not a second hand-maintained copy that could drift out
of sync with the scenario.
"""

from pathlib import Path
from typing import Any

import yaml

from driftwatch.config.schema import ModelConfig
from driftwatch.demo.schema import ScenarioConfig
from driftwatch.settings import get_settings


def derive_model_config_dict(scenario: ScenarioConfig) -> dict[str, Any]:
    return {
        "model_id": scenario.model_id,
        "profile": scenario.profile,
        "prediction_type": scenario.prediction_type,
        "schema": {
            "features": [
                {"name": f.name, "dtype": f.dtype, "nullable": f.nullable}
                for f in scenario.features
            ],
            "prediction": {"dtype": "continuous"},
            "label": {"dtype": "categorical"},
        },
        "segments": {
            "dimensions": scenario.segments,
            "min_segment_size": scenario.min_segment_size,
        },
    }


def write_model_config_yaml(scenario: ScenarioConfig) -> Path:
    """Writes configs/models/<model_id>.yaml, validating it against the
    real ModelConfig schema before writing -- a scenario that would
    produce a model config the evaluation pipeline can't actually load
    must fail loudly here, at generation time, not silently at whatever
    later point evaluate_range happens to first touch this model_id."""
    data = derive_model_config_dict(scenario)
    ModelConfig.model_validate(data)

    path = get_settings().configs_dir / "models" / f"{scenario.model_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path
