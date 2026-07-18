import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from driftwatch.db.session import SessionLocal
from driftwatch.scheduler.jobs import evaluate_pending_windows

logger = logging.getLogger(__name__)

TICK_INTERVAL_MINUTES = 5


def run_evaluation_tick() -> None:
    session = SessionLocal()
    try:
        count = evaluate_pending_windows(session)
        logger.info("evaluated %d window(s)", count)
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
