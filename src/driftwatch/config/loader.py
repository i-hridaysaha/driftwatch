from functools import lru_cache

import yaml

from driftwatch.config.schema import ModelConfig, Profile
from driftwatch.metrics.registry import available_metrics
from driftwatch.settings import get_settings


class ProfileNotFoundError(FileNotFoundError):
    pass


class ModelConfigNotFoundError(FileNotFoundError):
    pass


@lru_cache
def load_profile(name: str) -> Profile:
    path = get_settings().configs_dir / "profiles" / f"{name}.yaml"
    if not path.exists():
        raise ProfileNotFoundError(f"no profile config at {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    profile = Profile.model_validate(raw)
    known = available_metrics()
    unknown = [spec.name for spec in profile.performance_metrics if spec.name not in known]
    if unknown:
        raise ValueError(f"profile {name!r} references unknown metrics: {unknown}")
    return profile


@lru_cache
def load_model_config(model_id: str) -> ModelConfig:
    path = get_settings().configs_dir / "models" / f"{model_id}.yaml"
    if not path.exists():
        raise ModelConfigNotFoundError(f"no model config at {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    config = ModelConfig.model_validate(raw)
    load_profile(config.profile)
    return config
