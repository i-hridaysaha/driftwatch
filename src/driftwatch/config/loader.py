from functools import lru_cache

import yaml

from driftwatch.config.schema import ModelConfig, Profile
from driftwatch.hashing import compute_payload_hash
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


def compute_config_hash(model_config: ModelConfig, profile: Profile) -> str:
    """Fingerprints everything about a model's config that could change how a
    window is evaluated (feature schema, segments, which profile, and that
    profile's full content) into one hash, stored on every EvaluationWindow.
    Lets a later reader tell whether a window's config has since changed --
    e.g. before deciding whether to trust an old window's drift verdict when
    comparing it against today's."""
    return compute_payload_hash(
        {
            "model_config": model_config.model_dump(by_alias=True),
            "profile": profile.model_dump(),
        }
    )
