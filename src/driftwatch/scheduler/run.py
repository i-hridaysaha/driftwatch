import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from driftwatch.alerting.notifications import default_channels
from driftwatch.db.session import SessionLocal
from driftwatch.scheduler.jobs import (
    evaluate_pending_windows,
    recompute_stale_windows,
    retry_pending_notifications,
)

logger = logging.getLogger(__name__)

TICK_INTERVAL_MINUTES = 5


def run_evaluation_tick() -> None:
    """One tick, three jobs, in this order: seal any window whose drift
    watermark has elapsed; recompute performance for every window label
    ingestion has flagged since the last tick; retry any alert transition
    whose delivery failed. Performance therefore lags a label batch by at
    most one tick, which is the price of keeping recomputation out of the
    request that carried the labels."""
    session = SessionLocal()
    channels = default_channels()
    try:
        evaluated = evaluate_pending_windows(session)
        recomputed = recompute_stale_windows(session, channels)
        retried = retry_pending_notifications(session, channels)
        logger.info(
            "evaluated %d window(s), recomputed %d stale window(s), retried %d notification(s)",
            evaluated,
            recomputed,
            retried,
        )
    finally:
        session.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    scheduler = BlockingScheduler()
    scheduler.add_job(
        run_evaluation_tick,
        trigger="interval",
        minutes=TICK_INTERVAL_MINUTES,
        id="evaluate_pending_windows",
        # A missed tick (scheduler was down, or a prior tick was still running)
        # must be recoverable, not silently dropped. misfire_grace_time=None
        # means there is no deadline after which a late run is abandoned -- it
        # always eventually executes once the scheduler can get to it, however
        # late. coalesce=True collapses any backlog of missed ticks into a
        # single run instead of firing once per missed tick: evaluate_pending_
        # windows already scans the full unevaluated backlog on every call
        # (see find_drift_ready_unevaluated_windows), so running it N times
        # back to back after an outage would just repeat the same discovery
        # query for no benefit. max_instances=1 prevents two ticks from
        # overlapping and doing redundant work concurrently -- correctness
        # doesn't depend on this (evaluate_window is safe under concurrent
        # execution via its own claim), it's purely to avoid wasted work.
        misfire_grace_time=None,
        coalesce=True,
        max_instances=1,
    )
    logger.info("driftwatch scheduler starting (tick every %d minutes)", TICK_INTERVAL_MINUTES)
    scheduler.start()


if __name__ == "__main__":
    main()
