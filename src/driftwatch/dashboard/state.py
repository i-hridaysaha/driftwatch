from collections.abc import MutableMapping
from datetime import date


def parse_date_param(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def format_date_param(value: date) -> str:
    return value.isoformat()


def resolve_time_range(
    params: MutableMapping[str, str], default_range: tuple[date, date]
) -> tuple[date, date]:
    """Resolves the visible time range: explicit ?start=&end= query params
    win if present and parseable; otherwise falls back to `default_range`,
    which callers derive from the data actually present in the database
    (driftwatch.dashboard.queries.default_time_range) -- never
    datetime.now(). Always writes the resolved value back into `params`, so
    the URL reflects exactly what's rendered even on first load before a
    user has touched anything: a chart's URL alone is then enough to
    regenerate it identically later, which is the whole point of this
    function existing instead of plain st.session_state.
    """
    start = parse_date_param(params.get("start")) or default_range[0]
    end = parse_date_param(params.get("end")) or default_range[1]
    if start > end:
        start, end = default_range
    params["start"] = format_date_param(start)
    params["end"] = format_date_param(end)
    return start, end


def get_str(params: MutableMapping[str, str], key: str, default: str) -> str:
    value = params.get(key)
    return value if value else default


def set_str(params: MutableMapping[str, str], key: str, value: str) -> None:
    params[key] = value


def get_optional_str(params: MutableMapping[str, str], key: str) -> str | None:
    value = params.get(key)
    return value if value else None


def set_optional_str(params: MutableMapping[str, str], key: str, value: str | None) -> None:
    if value is None:
        params.pop(key, None)
    else:
        params[key] = value


def get_optional_int(params: MutableMapping[str, str], key: str) -> int | None:
    value = params.get(key)
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None
