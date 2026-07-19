from datetime import UTC, datetime

from sqlalchemy.orm import Session

from driftwatch.alerting.notifications import (
    AlertEvent,
    LoggingNotificationChannel,
    notify_if_needed,
)
from driftwatch.db.models import Alert, AlertKind, AlertStatus, Model


class _RecordingChannel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, AlertEvent]] = []

    def notify(self, alert: Alert, event: AlertEvent) -> None:
        self.calls.append((alert.id, event))


class _RaisingChannel:
    """Simulates a hanging or erroring webhook -- an expected operational
    condition, not a bug. notify_if_needed must never let this propagate."""

    def notify(self, alert: Alert, event: AlertEvent) -> None:
        raise RuntimeError("webhook unreachable")


def _alert(status: AlertStatus, last_notified_status: AlertStatus | None = None) -> Alert:
    alert = Alert(
        id=1,
        model_id="example-model",
        kind=AlertKind.DRIFT,
        signal_name="psi",
        feature_name="age",
        status=status,
        first_opened_at=datetime.now(UTC),
        last_seen_at=datetime.now(UTC),
        last_notified_status=last_notified_status,
    )
    return alert


def test_notifies_on_first_open() -> None:
    alert = _alert(AlertStatus.OPEN, last_notified_status=None)
    channel = _RecordingChannel()

    dispatched = notify_if_needed(alert, [channel])

    assert dispatched is True
    assert channel.calls == [(1, "opened")]
    assert alert.last_notified_status == AlertStatus.OPEN


def test_does_not_renotify_on_steady_state() -> None:
    # already notified for OPEN; nothing changed since
    alert = _alert(AlertStatus.OPEN, last_notified_status=AlertStatus.OPEN)
    channel = _RecordingChannel()

    dispatched = notify_if_needed(alert, [channel])

    assert dispatched is False
    assert channel.calls == []


def test_renotifies_on_escalation() -> None:
    alert = _alert(AlertStatus.ESCALATED, last_notified_status=AlertStatus.OPEN)
    channel = _RecordingChannel()

    dispatched = notify_if_needed(alert, [channel])

    assert dispatched is True
    assert channel.calls == [(1, "escalated")]
    assert alert.last_notified_status == AlertStatus.ESCALATED


def test_renotifies_on_resolution() -> None:
    alert = _alert(AlertStatus.RESOLVED, last_notified_status=AlertStatus.ESCALATED)
    channel = _RecordingChannel()

    dispatched = notify_if_needed(alert, [channel])

    assert dispatched is True
    assert channel.calls == [(1, "resolved")]


def test_repeated_calls_after_transition_only_notify_once() -> None:
    alert = _alert(AlertStatus.OPEN, last_notified_status=None)
    channel = _RecordingChannel()

    notify_if_needed(alert, [channel])
    notify_if_needed(alert, [channel])
    notify_if_needed(alert, [channel])

    assert len(channel.calls) == 1


def test_dispatches_to_all_configured_channels() -> None:
    alert = _alert(AlertStatus.OPEN, last_notified_status=None)
    channel_a = _RecordingChannel()
    channel_b = _RecordingChannel()

    notify_if_needed(alert, [channel_a, channel_b])

    assert channel_a.calls == [(1, "opened")]
    assert channel_b.calls == [(1, "opened")]


def test_logging_channel_does_not_raise() -> None:
    alert = _alert(AlertStatus.OPEN)

    LoggingNotificationChannel().notify(alert, "opened")


def test_raising_channel_does_not_propagate_and_records_failure() -> None:
    alert = _alert(AlertStatus.OPEN, last_notified_status=None)

    dispatched = notify_if_needed(alert, [_RaisingChannel()])  # must not raise

    assert dispatched is True
    # no retry loop: the transition is still marked notified even though
    # delivery failed -- the failure is surfaced via last_notification_error,
    # not by holding the alert in an un-notified state forever
    assert alert.last_notified_status == AlertStatus.OPEN
    assert alert.last_notification_error is not None
    assert "webhook unreachable" in alert.last_notification_error
    assert alert.last_notification_error_at is not None


def test_one_raising_channel_does_not_block_the_others() -> None:
    alert = _alert(AlertStatus.OPEN, last_notified_status=None)
    good = _RecordingChannel()

    notify_if_needed(alert, [_RaisingChannel(), good])

    assert good.calls == [(1, "opened")]


def test_raising_channel_leaves_alert_transaction_committed(db_session: Session) -> None:
    """The scenario the constraint is actually about: a notification failure
    happening mid-evaluation-run must not fail that run or roll back the
    alert row it's attached to. Proven here by actually committing the
    session afterward and re-reading the row back."""
    db_session.add(Model(model_id="example-model"))
    db_session.flush()
    alert = Alert(
        model_id="example-model",
        kind=AlertKind.DRIFT,
        signal_name="psi",
        feature_name="age",
        status=AlertStatus.OPEN,
    )
    db_session.add(alert)
    db_session.flush()
    alert_id = alert.id

    notify_if_needed(alert, [_RaisingChannel()])  # must not raise
    db_session.commit()  # must not roll back

    persisted = db_session.get(Alert, alert_id)
    assert persisted is not None
    assert persisted.status == AlertStatus.OPEN
    assert persisted.last_notified_status == AlertStatus.OPEN
    assert persisted.last_notification_error is not None
