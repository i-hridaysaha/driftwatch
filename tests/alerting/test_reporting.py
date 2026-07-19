from driftwatch.alerting.reporting import find_suppressed_episodes


def test_run_below_persistence_is_suppressed() -> None:
    # windows 1,2,3 breach (persistence needs 3) then window 4 clears
    history = [
        (1, "breach", 0.5),
        (2, "breach", 0.6),
        (3, "clear", 0.1),
    ]
    episodes = find_suppressed_episodes(history, persistence_required=3, target_kind="breach")

    assert len(episodes) == 1
    assert episodes[0].start_window_id == 1
    assert episodes[0].end_window_id == 2
    assert episodes[0].length == 2
    assert episodes[0].peak_value == 0.6


def test_run_at_persistence_is_not_reported() -> None:
    """A run that reaches persistence corresponds to a real Alert row --
    reporting.py's job is only the complement of that, not a restatement of
    it."""
    history = [(1, "breach", 0.5), (2, "breach", 0.6), (3, "breach", 0.7)]
    episodes = find_suppressed_episodes(history, persistence_required=3, target_kind="breach")

    assert episodes == []


def test_run_still_in_progress_at_end_of_history_is_suppressed() -> None:
    """Hasn't reverted, just hasn't reached persistence as of the most
    recent window yet -- still correctly "suppressed as of now"."""
    history = [(1, "clear", 0.1), (2, "breach", 0.4)]
    episodes = find_suppressed_episodes(history, persistence_required=3, target_kind="breach")

    assert len(episodes) == 1
    assert episodes[0].start_window_id == 2
    assert episodes[0].end_window_id == 2
    assert episodes[0].length == 1


def test_multiple_interrupted_runs_are_each_reported_separately() -> None:
    history = [
        (1, "breach", 0.3),
        (2, "clear", 0.1),
        (3, "breach", 0.4),
        (4, "breach", 0.9),
        (5, "clear", 0.1),
    ]
    episodes = find_suppressed_episodes(history, persistence_required=3, target_kind="breach")

    assert len(episodes) == 2
    assert (episodes[0].start_window_id, episodes[0].end_window_id) == (1, 1)
    assert (episodes[1].start_window_id, episodes[1].end_window_id) == (3, 4)
    assert episodes[1].peak_value == 0.9


def test_dead_zone_and_not_computable_both_interrupt_a_breach_run() -> None:
    history = [
        (1, "breach", 0.3),
        (2, "dead_zone", 0.2),
        (3, "breach", 0.35),
        (4, "not_computable", None),
        (5, "breach", 0.4),
    ]
    episodes = find_suppressed_episodes(history, persistence_required=2, target_kind="breach")

    assert [e.length for e in episodes] == [1, 1, 1]


def test_all_not_computable_run_has_no_peak_value() -> None:
    history = [(1, "not_computable", None), (2, "not_computable", None)]
    episodes = find_suppressed_episodes(
        history, persistence_required=3, target_kind="not_computable"
    )

    assert len(episodes) == 1
    assert episodes[0].peak_value is None


def test_not_computable_target_kind_ignores_breach_runs() -> None:
    history = [(1, "breach", 0.9), (2, "breach", 0.9), (3, "not_computable", None)]
    episodes = find_suppressed_episodes(
        history, persistence_required=2, target_kind="not_computable"
    )

    assert len(episodes) == 1
    assert episodes[0].start_window_id == 3
    assert episodes[0].length == 1


def test_empty_history_returns_no_episodes() -> None:
    assert find_suppressed_episodes([], persistence_required=2, target_kind="breach") == []
