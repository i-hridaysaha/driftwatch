from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://driftwatch:driftwatch@localhost:5432/driftwatch"
    configs_dir: Path = Path("configs")
    alert_webhook_url: str | None = None
    """If set, alert notifications are also POSTed here as JSON (see
    driftwatch.alerting.notifications.WebhookNotificationChannel), alongside
    the always-on logging channel."""
    api_key: str | None = None
    """`API_KEY`. If set, every write endpoint (baselines, predictions,
    labels) requires the request header `X-API-Key` to equal it; `/health`
    stays open. If unset (or empty, which Compose passes through when the
    variable is not exported) the write endpoints are open and the API logs
    a warning at startup -- the shipped demo has no secret to configure,
    and a monitor that silently refused every prediction because a key was
    missing would be its own kind of silent failure. See
    driftwatch.api.deps.require_api_key."""

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_key_is_unset(cls, value: object) -> object:
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
