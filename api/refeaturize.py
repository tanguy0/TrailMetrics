"""Rebuild stored feature rows from the stream blobs, for every athlete.

    python -m api.refeaturize                 # every athlete
    python -m api.refeaturize --athlete 123   # one of them
    python -m api.refeaturize --dry-run       # count what is stale, write nothing

Run this once after deploying a ``FEATURE_VERSION`` bump that changes the maths
rather than which streams are read. Nothing *requires* it — each athlete's next
sync does the same work for them (see
:mod:`src.usecases.refeaturize_athlete_activities`, wired into
:mod:`src.usecases.sync_athlete_activities`) — but doing it here fixes the whole
population in one pass, before anyone opens the app to a half-rebuilt history.

It talks to Postgres and object storage only: no Strava credentials, no rate
limit, and it works for athletes whose Strava connection is long gone. Anything
the blobs cannot answer is left at its old version and reported, for the sync's
Strava path to pick up.
"""

import argparse
import logging
import sys
from typing import List, Optional

from api.deps import (
    get_activity_repository,
    get_athlete_repository,
    get_stream_store,
)
from src.usecases.refeaturize_athlete_activities import (
    RefeaturizeAthleteActivities,
    RefeaturizeAthleteActivitiesInput,
)

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild stored feature rows from the stream blobs.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--athlete", type=int, default=None,
        help="Only this athlete id (default: all of them).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report how many rows are out of date without rewriting any.",
    )
    parser.add_argument(
        "--max-activities", type=int, default=None,
        help="Stop after this many activities per athlete.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    activities = get_activity_repository()
    athletes = get_athlete_repository()
    ids = (
        [args.athlete] if args.athlete is not None
        else [athlete.id for athlete in athletes.list_all()]
    )
    if not ids:
        print("No athletes.")
        return 0

    if args.dry_run:
        total = 0
        for athlete_id in ids:
            stale = len(activities.stale_activities(athlete_id))
            total += stale
            if stale:
                print(f"athlete {athlete_id}: {stale} stale")
        print(f"{total} stale row(s) across {len(ids)} athlete(s); nothing written.")
        return 0

    usecase = RefeaturizeAthleteActivities(
        activities=activities, streams=get_stream_store(), athletes=athletes,
    )
    rebuilt = needs_strava = failed = 0
    for athlete_id in ids:
        try:
            outcome = usecase.execute(RefeaturizeAthleteActivitiesInput(
                athlete_id=athlete_id, max_activities=args.max_activities,
            ))
        except Exception as error:
            # One athlete's storage problem must not stop the population pass.
            print(f"athlete {athlete_id}: FAILED — {error}", file=sys.stderr)
            failed += 1
            continue
        if outcome.stale:
            print(
                f"athlete {athlete_id}: {outcome.rebuilt} rebuilt, "
                f"{outcome.summary_only} summary-only, "
                f"{outcome.needs_strava} left for Strava, {outcome.failed} failed"
            )
        rebuilt += outcome.done
        needs_strava += outcome.needs_strava

    print(f"\n{rebuilt} row(s) rebuilt across {len(ids)} athlete(s).")
    if needs_strava:
        print(
            f"{needs_strava} row(s) had no usable blob and stayed at their old "
            "version — those still need a Strava re-import, which each athlete's "
            "next sync will do for them."
        )
    if failed:
        print(f"{failed} athlete(s) failed; see the errors above.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
