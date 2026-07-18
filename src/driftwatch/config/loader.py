from functools import lru_cache

import yaml

from driftwatch.config.schema import Profile
from driftwatch.settings import get_settings


class ProfileNotFoundError(FileNotFoundError):
    pass


@lru_cache
def load_profile(name: str) -> Profile:
    path = get_settings().configs_dir / "profiles" / f"{name}.yaml"
    if not path.exists():
        raise ProfileNotFoundError(f"no profile config at {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    return Profile.model_validate(raw)
