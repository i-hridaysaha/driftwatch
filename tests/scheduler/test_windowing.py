from datetime import UTC, datetime, timedelta

import pytest

from driftwatch.durations import parse_duration
from driftwatch.scheduler.windowing import compute_window_boundaries, is_drift_watermark_elapsed


def test_parse_duration_known_values() -> None:
    assert parse_duration("30s") == timedelta(seconds=30)
    assert parse_duration("10m") == timedelta(minutes=10)
    assert parse_duration("1h") == timedelta(hours=1)
    assert parse_duration("1d") == timedelta(days=1)


def test_parse_duration_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("1w")
    with pytest.raises(ValueError, match="invalid duration"):
        parse_duration("abc")


def test_window_boundaries_are_epoch_aligned_not_range_start_aligned() -> None:
    # a 1-hour window starting mid-hour must still snap to the hour boundary
    range_start = datetime(2026, 1, 1, 0, 30, tzinfo=UTC)
    range_end = datetime(2026, 1, 1, 2, 30, tzinfo=UTC)

    windows = compute_window_boundaries(range_start, range_end, timedelta(hours=1))

    assert windows == [
        (datetime(2026, 1, 1, 0, tzinfo=UTC), datetime(2026, 1, 1, 1, tzinfo=UTC)),
        (datetime(2026, 1, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 2, tzinfo=UTC)),
        (datetime(2026, 1, 1, 2, tzinfo=UTC), datetime(2026, 1, 1, 3, tzinfo=UTC)),
    ]


def test_window_boundaries_are_deterministic_across_calls() -> None:
    range_start = datetime(2026, 3, 5, 7, tzinfo=UTC)
    range_end = datetime(2026, 3, 6, 13, tzinfo=UTC)

    first = compute_window_boundaries(range_start, range_end, timedelta(hours=1))
    second = compute_window_boundaries(range_start, range_end, timedelta(hours=1))

    assert first == second
    assert len(first) == 30  # 30 hourly windows between 07:00 day 1 and 13:00 day 2


def test_window_boundaries_rejects_naive_datetimes() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_window_boundaries(
            datetime(2026, 1, 1), datetime(2026, 1, 2, tzinfo=UTC), timedelta(hours=1)  # noqa: DTZ001
        )


def test_window_boundaries_rejects_non_positive_duration() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        compute_window_boundaries(
            datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC), timedelta(0)
        )


def test_window_boundaries_empty_range_gives_no_windows() -> None:
    same_instant = datetime(2026, 1, 1, tzinfo=UTC)

    assert compute_window_boundaries(same_instant, same_instant, timedelta(hours=1)) == []


def test_is_drift_watermark_elapsed_before_and_after_watermark() -> None:
    window_end = datetime(2026, 1, 1, 1, tzinfo=UTC)
    watermark = timedelta(minutes=10)

    assert not is_drift_watermark_elapsed(window_end, watermark, as_of=window_end)
    assert not is_drift_watermark_elapsed(
        window_end, watermark, as_of=window_end + timedelta(minutes=9)
    )
    assert is_drift_watermark_elapsed(
        window_end, watermark, as_of=window_end + timedelta(minutes=10)
    )
    assert is_drift_watermark_elapsed(window_end, watermark, as_of=window_end + timedelta(hours=1))
