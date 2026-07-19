from pathlib import Path

import yaml

from driftwatch.demo.schema import ScenarioConfig
from driftwatch.settings import get_settings


class ScenarioNotFoundError(FileNotFoundError):
    pass


def scenarios_dir() -> Path:
    return get_settings().configs_dir / "scenarios"


def list_scenario_names() -> list[str]:
    return sorted(p.stem for p in scenarios_dir().glob("*.yaml"))


def load_scenario(name: str) -> ScenarioConfig:
    path = scenarios_dir() / f"{name}.yaml"
    if not path.exists():
        raise ScenarioNotFoundError(f"no scenario config at {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    return ScenarioConfig.model_validate(raw)
