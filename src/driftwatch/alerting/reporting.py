from collections.abc import Sequence
from dataclasses import dataclass

from driftwatch.alerting.streaks import StreakKind


@dataclass(frozen=True)
class SuppressedEpisode:
    """A maximal run of consecutive `kind` classifications (chronological)
    that never reached `persistence_required` -- the complement of the
    alerts table: every episode here is one driftwatch.alerting.engine
    looked at and correctly declined to escalate into a real Alert row,
    because the signal reverted (or the window range ended) before its
    persistence gate was satisfied. Read-only: this never touches the
    alerts table and never mutates anything, it only replays history
    through the same pure classify_drift/classify_performance functions
    the real engine uses.
    """

    kind: StreakKind
    start_window_id: int
    end_window_id: int
    length: int
    peak_value: float | None
    """max statistic (drift) / worst metric_value (performance) seen during
    the episode, on whatever scale that signal's classify_* call used --
    None if every window in the episode was itself not_computable (no
    numeric value ever existed to report)."""


def find_suppressed_episodes(
    classifications: Sequence[tuple[int, StreakKind, float | None]],
    persistence_required: int,
    target_kind: StreakKind,
) -> list[SuppressedEpisode]:
    """`classifications`: one signal identity's (window_id, classification,
    value) history, oldest-first. Groups consecutive runs of `target_kind`
    and returns every run whose length stayed below `persistence_required`
    -- i.e. every run that a real Alert would NOT have opened for. A run
    still in progress at the end of the supplied history (hasn't reverted
    yet, just hasn't reached persistence as of the most recent window) is
    reported the same way: suppressed as of now, may still fire later.

    Deliberately does not need a Session or any DB access -- callers fetch
    the (bounded, already-aggregated) DriftResult/PerformanceResult history
    for one signal and classify each row with the same classify_drift /
    classify_performance functions driftwatch.alerting.engine uses, so this
    never reimplements or drifts from the real firing logic.
    """
    episodes: list[SuppressedEpisode] = []
    run: list[tuple[int, float | None]] = []

    def flush() -> None:
        if run and len(run) < persistence_required:
            values = [v for _, v in run if v is not None]
            episodes.append(
                SuppressedEpisode(
                    kind=target_kind,
                    start_window_id=run[0][0],
                    end_window_id=run[-1][0],
                    length=len(run),
                    peak_value=max(values) if values else None,
                )
            )
        run.clear()

    for window_id, classification, value in classifications:
        if classification == target_kind:
            run.append((window_id, value))
        else:
            flush()
    flush()
    return episodes
