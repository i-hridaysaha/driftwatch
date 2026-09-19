import logging
import secrets

from fastapi import Header, HTTPException

from driftwatch.db.session import get_db
from driftwatch.settings import get_settings

__all__ = ["get_db", "require_api_key"]

logger = logging.getLogger(__name__)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Write-endpoint authentication: a shared key in `X-API-Key`, compared
    in constant time, required whenever Settings.api_key is configured.
    Anything that can reach the API can otherwise register a baseline and
    silently change what every future drift number is measured against,
    which is the one write worth protecting even in a demo."""
    expected = get_settings().api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
