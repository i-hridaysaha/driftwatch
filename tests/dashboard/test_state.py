from datetime import date

from driftwatch.dashboard.state import (
    get_optional_int,
    get_optional_str,
    get_str,
    parse_date_param,
    resolve_time_range,
    set_optional_str,
    set_str,
)


def test_parse_date_param_valid() -> None:
    assert parse_date_param("2026-01-15") == date(2026, 1, 15)


def test_parse_date_param_missing_or_malformed_returns_none() -> None:
    assert parse_date_param(None) is None
    assert parse_date_param("") is None
    assert parse_date_param("not-a-date") is None


def test_resolve_time_range_falls_back_to_default_when_no_params() -> None:
    params: dict[str, str] = {}
    default = (date(2026, 1, 1), date(2026, 2, 1))

    start, end = resolve_time_range(params, default)

    assert (start, end) == default
    # writes the resolved range back so the URL reflects it immediately,
    # even on first load before the user has touched anything
    assert params["start"] == "2026-01-01"
    assert params["end"] == "2026-02-01"


def test_resolve_time_range_prefers_explicit_query_params() -> None:
    params = {"start": "2026-01-10", "end": "2026-01-20"}
    default = (date(2026, 1, 1), date(2026, 2, 1))

    start, end = resolve_time_range(params, default)

    assert (start, end) == (date(2026, 1, 10), date(2026, 1, 20))


def test_resolve_time_range_ignores_malformed_params() -> None:
    params = {"start": "garbage", "end": "2026-01-20"}
    default = (date(2026, 1, 1), date(2026, 2, 1))

    start, end = resolve_time_range(params, default)

    assert start == date(2026, 1, 1)  # fell back to default
    assert end == date(2026, 1, 20)  # this one parsed fine, kept


def test_resolve_time_range_falls_back_when_start_after_end() -> None:
    params = {"start": "2026-02-01", "end": "2026-01-01"}
    default = (date(2026, 1, 1), date(2026, 3, 1))

    start, end = resolve_time_range(params, default)

    assert (start, end) == default


def test_get_str_uses_default_when_absent() -> None:
    assert get_str({}, "view", "drift_timeline") == "drift_timeline"
    assert get_str({"view": "alerts"}, "view", "drift_timeline") == "alerts"


def test_set_str_round_trips() -> None:
    params: dict[str, str] = {}
    set_str(params, "view", "alerts")
    assert params["view"] == "alerts"


def test_optional_str_round_trips_and_clears() -> None:
    params: dict[str, str] = {}
    set_optional_str(params, "segment", "EU")
    assert get_optional_str(params, "segment") == "EU"

    set_optional_str(params, "segment", None)
    assert get_optional_str(params, "segment") is None
    assert "segment" not in params


def test_get_optional_int() -> None:
    assert get_optional_int({}, "window_id") is None
    assert get_optional_int({"window_id": "42"}, "window_id") == 42
    assert get_optional_int({"window_id": "not-an-int"}, "window_id") is None
