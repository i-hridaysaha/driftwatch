import re
from datetime import timedelta

_PATTERN = re.compile(r"^(\d+)([smhd])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: str) -> timedelta:
    """Parses a duration string like '30s', '10m', '1h', '1d' into a timedelta.

    Used for both evaluation.window and evaluation.watermark -- kept as a
    standalone module with no dependencies so both config validation and the
    scheduler's window arithmetic can import it without a circular import.
    """
    match = _PATTERN.match(value.strip())
    if not match:
        raise ValueError(f"invalid duration {value!r}, expected e.g. '30s', '10m', '1h', '1d'")
    amount, unit = match.groups()
    return timedelta(seconds=int(amount) * _UNIT_SECONDS[unit])
