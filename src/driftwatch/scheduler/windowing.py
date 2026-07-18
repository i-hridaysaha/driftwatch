from datetime import UTC, datetime, timedelta

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def compute_window_boundaries(
    range_start: datetime, range_end: datetime, window_duration: timedelta
) -> list[tuple[datetime, datetime]]:
    """Deterministic, epoch-aligned window boundaries covering
    [range_start, range_end).

    Windows are keyed on event time, never ingestion time -- this function
    only ever sees `predicted_at`-derived range bounds, and knows nothing
    about when data was actually ingested. Boundaries are aligned to the
    Unix epoch (not to whenever the model happened to be registered, or to
    whenever this function happens to be called), and are pure arithmetic on
    `window_duration` -- so the same (range, duration) always produces
    exactly the same window boundaries, computed by a scheduler tick today or
    a CLI backfill run next year.

    All datetimes must be UTC-aware; naive datetimes are a caller bug, not
    silently assumed to be UTC.
    """
    if window_duration <= timedelta(0):
        raise ValueError("window_duration must be positive")
    if range_start.tzinfo is None or range_end.tzinfo is None:
        raise ValueError("range_start and range_end must be timezone-aware (UTC)")

    first_index = (range_start - EPOCH) // window_duration
    windows: list[tuple[datetime, datetime]] = []
    index = first_index
    while True:
        window_start = EPOCH + index * window_duration
        if window_start >= range_end:
            break
        window_end = window_start + window_duration
        if window_end > range_start:
            windows.append((window_start, window_end))
        index += 1
    return windows


def is_drift_watermark_elapsed(window_end: datetime, watermark: timedelta, as_of: datetime) -> bool:
    """True once `as_of` reaches window_end + watermark -- the point at which
    a window's DRIFT statistics become eligible for their one-time,
    never-recomputed evaluation.

    This governs drift only, not the whole window's lifecycle. See
    driftwatch.config.schema.EvaluationConfig.watermark for why drift is
    computed once on a fixed delay rather than retroactively re-evaluated
    when late predictions arrive: predictions, unlike labels, are not
    expected to be routinely delayed, so a bounded, predictable evaluation
    latency is worth more than catching the rare straggler. Performance
    metrics for the same window are a completely separate lifecycle --
    they keep being recomputed indefinitely as labels backfill (see
    driftwatch.evaluation.performance.recompute_performance_for_window and
    EvaluationWindow.label_watermark), with no watermark of their own."""
    return as_of >= window_end + watermark
