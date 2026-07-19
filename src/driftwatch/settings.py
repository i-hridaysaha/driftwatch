from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://driftwatch:driftwatch@localhost:5432/driftwatch"
    configs_dir: Path = Path("configs")
    alert_webhook_url: str | None = None
    """If set, alert notifications are also POSTed here as JSON (see
    driftwatch.alerting.notifications.WebhookNotificationChannel), alongside
    the always-on logging channel."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
