import logging
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, Literal

import httpx

from driftwatch.db.models import Alert, AlertStatus
from driftwatch.settings import get_settings

logger = logging.getLogger(__name__)

AlertEvent = Literal["opened", "escalated", "resolved"]

_STATUS_TO_EVENT: dict[AlertStatus, AlertEvent] = {
    AlertStatus.OPEN: "opened",
    AlertStatus.ESCALATED: "escalated",
    AlertStatus.RESOLVED: "resolved",
}


class NotificationChannel(ABC):
    @abstractmethod
    def notify(self, alert: Alert, event: AlertEvent) -> None: ...


class LoggingNotificationChannel(NotificationChannel):
    """Always-available fallback: writes the alert to the application log.
    No external dependency, no delivery failure mode."""

    def notify(self, alert: Alert, event: AlertEvent) -> None:
        logger.warning(
            "ALERT %s: model=%s kind=%s signal=%s feature=%s segment=%s=%s "
            "status=%s alert_id=%d aggregate=%s",
            event.upper(),
            alert.model_id,
            alert.kind.value,
            alert.signal_name,
            alert.feature_name,
            alert.segment_dimension,
            alert.segment_value,
            alert.status.value,
            alert.id,
            alert.is_aggregate,
        )


class WebhookNotificationChannel(NotificationChannel):
    """Generic HTTP POST of a JSON payload. Deliberately not a Slack/email/
    PagerDuty integration -- those are thin adapters a real deployment can
    build on top of a webhook receiver, not something this service should
    special-case.

    Does not catch its own delivery errors -- notify_if_needed does that
    centrally for every channel, so a hanging or erroring webhook (an
    expected operational condition, not a bug) is recorded on the alert
    rather than silently swallowed here."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self._url = url
        self._timeout = timeout

    def notify(self, alert: Alert, event: AlertEvent) -> None:
        payload = _alert_payload(alert, event)
        response = httpx.post(self._url, json=payload, timeout=self._timeout)
        response.raise_for_status()


def _alert_payload(alert: Alert, event: AlertEvent) -> dict[str, Any]:
    return {
        "event": event,
        "alert_id": alert.id,
        "model_id": alert.model_id,
        "kind": alert.kind.value,
        "signal_name": alert.signal_name,
        "feature_name": alert.feature_name,
        "segment_dimension": alert.segment_dimension,
        "segment_value": alert.segment_value,
        "status": alert.status.value,
        "is_aggregate": alert.is_aggregate,
        "aggregated_signal_count": alert.aggregated_signal_count,
        "evidence_statistic": alert.evidence_statistic,
        "evidence_threshold": alert.evidence_threshold,
        "evidence_reason": alert.evidence_reason,
        "first_opened_at": alert.first_opened_at.isoformat(),
        "last_seen_at": alert.last_seen_at.isoformat(),
    }


MAX_NOTIFICATION_ATTEMPTS = 5
"""Delivery rounds a single status transition gets before it is given up
on. Each round is one scheduler tick apart (driftwatch.scheduler.jobs
.retry_pending_notifications), so at the default five-minute tick a webhook
outage of up to about twenty-five minutes still delivers the transition."""


def notify_if_needed(alert: Alert, channels: list[NotificationChannel]) -> bool:
    """Re-notification policy: notify only on a STATUS TRANSITION (opened,
    escalated, resolved), never on steady-state continuation of an
    already-notified status -- a still-open alert does not re-notify on
    every scheduler tick just because its last_seen advanced. Idempotent
    once a round succeeds: calling this again with an unchanged status is a
    no-op. Returns True if a delivery round was attempted for a real
    transition.

    A channel raising (a hanging or erroring webhook is an expected
    operational condition) must never fail the evaluation run this is
    called from, nor roll back the alert row's transition it's attached to
    -- so every channel is called in isolation, and a failure is caught,
    logged, and recorded on the alert (last_notification_error).

    Retry is bounded and lives in the row, not in a queue: a round in which
    any channel failed leaves last_notified_status behind the real status
    and counts one attempt, so the next scheduler tick tries the whole
    round again (every channel, so a log line can repeat; the webhook is
    the one that matters), up to MAX_NOTIFICATION_ATTEMPTS. After that the
    status is marked delivered with the error left on the row, so a dead
    webhook cannot make the alert table grow an unbounded backlog. An
    earlier version recorded the failure once and never retried; an alert
    that opened during a webhook outage then reached nobody, ever."""
    if alert.last_notified_status == alert.status:
        return False
    event = _STATUS_TO_EVENT[alert.status]
    failed = False
    for channel in channels:
        try:
            channel.notify(alert, event)
        except Exception as exc:
            failed = True
            logger.exception(
                "notification channel %s failed to deliver alert %d event %s",
                type(channel).__name__,
                alert.id,
                event,
            )
            alert.last_notification_error = f"{type(channel).__name__}: {exc}"
            alert.last_notification_error_at = datetime.now(UTC)
    if failed:
        alert.notification_attempts += 1
        if alert.notification_attempts < MAX_NOTIFICATION_ATTEMPTS:
            return True  # status stays undelivered; the scheduler retries next tick
        logger.error(
            "giving up on alert %d event %s after %d attempts",
            alert.id,
            event,
            alert.notification_attempts,
        )
    else:
        alert.notification_attempts = 0
    alert.last_notified_status = alert.status
    alert.last_notified_at = datetime.now(UTC)
    return True


def default_channels() -> list[NotificationChannel]:
    """Logging is always on; the webhook is added only if configured (see
    Settings.alert_webhook_url) -- the same channel list used by both the
    scheduler and the label ingestion endpoint, so notification behavior
    doesn't depend on which code path triggered the alert evaluation."""
    channels: list[NotificationChannel] = [LoggingNotificationChannel()]
    webhook_url = get_settings().alert_webhook_url
    if webhook_url:
        channels.append(WebhookNotificationChannel(webhook_url))
    return channels
