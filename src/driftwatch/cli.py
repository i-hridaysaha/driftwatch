import argparse
import sys
from collections.abc import Callable
from datetime import datetime

from sqlalchemy.orm import Session

from driftwatch.config.loader import load_model_config, load_profile
from driftwatch.db.session import SessionLocal
from driftwatch.durations import parse_duration
from driftwatch.evaluation.drift import evaluate_window
from driftwatch.scheduler.windowing import compute_window_boundaries


def _parse_iso_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not a valid ISO 8601 datetime") from exc
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(f"{value!r} must be timezone-aware (e.g. end with 'Z')")
    return parsed


def evaluate_range(
    session: Session,
    model_id: str,
    start: datetime,
    end: datetime,
    *,
    force: bool = False,
    on_result: Callable[[datetime, datetime, bool], None] | None = None,
) -> tuple[int, int]:
    """Evaluates every window in [start, end) for `model_id`, via the exact
    same evaluate_window() the scheduler calls on its normal tick -- drift
    (final, computed once) and initial performance for whatever labels
    already exist. The only difference: this bypasses the drift watermark
    entirely, since a historical backfill is explicitly asking for these
    exact windows now, not "whatever is drift-ready" -- the scheduler's
    watermark gate lives in
    driftwatch.scheduler.jobs.find_drift_ready_unevaluated_windows, which
    this function never calls. Commits after each window. Returns
    (evaluated_count, skipped_count)."""
    model_config = load_model_config(model_id)
    profile = load_profile(model_config.profile)
    window_duration = parse_duration(profile.evaluation.window)

    evaluated = 0
    skipped = 0
    for window_start, window_end in compute_window_boundaries(start, end, window_duration):
        result = evaluate_window(session, model_id, window_start, window_end, force=force)
        was_evaluated = result is not None
        evaluated += was_evaluated
        skipped += not was_evaluated
        session.commit()
        if on_result is not None:
            on_result(window_start, window_end, was_evaluated)

    return evaluated, skipped


def _cmd_evaluate(args: argparse.Namespace) -> int:
    def report(window_start: datetime, window_end: datetime, was_evaluated: bool) -> None:
        label = f"[{window_start.isoformat()}, {window_end.isoformat()})"
        print(f"evaluated {label}" if was_evaluated else f"skipped (already evaluated) {label}")

    session = SessionLocal()
    try:
        evaluated, skipped = evaluate_range(
            session, args.model_id, args.start, args.end, force=args.force, on_result=report
        )
    finally:
        session.close()

    print(f"done: {evaluated} evaluated, {skipped} skipped")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    # Lazy import: driftwatch.demo.build imports evaluate_range from this
    # module, so a top-level import here would be circular.
    from driftwatch.demo.build import build_scenario
    from driftwatch.demo.loader import ScenarioNotFoundError, list_scenario_names

    try:
        summary = build_scenario(args.scenario, reset=not args.no_reset, seed=args.seed)
    except ScenarioNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"available scenarios: {', '.join(list_scenario_names())}", file=sys.stderr)
        return 1

    print(f"scenario: {summary.scenario_name}  (model_id={summary.model_id})")
    print(f"baseline records: {summary.baseline_records}")
    print(f"predictions: {summary.predictions}")
    print(f"labels: {summary.labels_early} before evaluation, {summary.labels_late} retroactive")
    print(f"windows: {summary.windows_evaluated} evaluated, {summary.windows_skipped} skipped")
    return 0


def _cmd_demo_verify(args: argparse.Namespace) -> int:
    from driftwatch.demo.verify import verify_scenario

    ok = verify_scenario(args.scenario, verbose=True)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="driftwatch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate an arbitrary historical time range for a model"
    )
    evaluate_parser.add_argument("--model-id", required=True)
    evaluate_parser.add_argument("--start", required=True, type=_parse_iso_utc)
    evaluate_parser.add_argument("--end", required=True, type=_parse_iso_utc)
    evaluate_parser.add_argument(
        "--force",
        action="store_true",
        help="delete and re-evaluate windows that already exist in this range",
    )
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    demo_parser = subparsers.add_parser(
        "demo", help="reset the database and load a named demo scenario end to end"
    )
    demo_parser.add_argument("scenario", help="scenario name, e.g. clean, covariate_shift")
    demo_parser.add_argument(
        "--no-reset",
        action="store_true",
        help="load additively instead of truncating all data first",
    )
    demo_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the scenario's seed (CI verifies every scenario at several)",
    )
    demo_parser.set_defaults(func=_cmd_demo)

    demo_verify_parser = subparsers.add_parser(
        "demo-verify", help="assert a loaded demo scenario produced what it claims"
    )
    demo_verify_parser.add_argument("scenario", help="scenario name")
    demo_verify_parser.set_defaults(func=_cmd_demo_verify)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
