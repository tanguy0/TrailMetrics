"""Fit the population durability coefficients from every athlete's past year.

    python -m api.calibrate_durability --output durability_population.json
    python -m api.calibrate_durability --max-athletes 50 --validation-fraction 0.25

Reads Postgres and object storage only (like :mod:`api.refeaturize`). For each
athlete: the reference speed from their past-year best efforts, then the steady
segments of their past-year long runs. The pooled segments go through
:func:`src.domain.durability.calibration.calibrate` — athlete-level split, robust
population fit, leave-one-activity-out validation — and the versioned result is
written as JSON. Nothing is written to the database: adopting a calibration is a
reviewed code change (replace ``PLACEHOLDER_POPULATION`` with the loaded values).
"""

import argparse
import logging
import sys
from datetime import date
from typing import List, Optional

from api.deps import data_source_for, get_athlete_repository
from src.domain.durability.calibration import CalibrationSettings, calibrate, to_json
from src.domain.durability.config import DEFAULT_CONFIG
from src.domain.durability.history import athlete_reference, durability_activity_ids
from src.domain.durability.segments import DurabilitySegment, extract_segments

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fit population durability coefficients.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", default="durability_population.json")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-athletes", type=int, default=None)
    parser.add_argument("--version", default="", help="Version string to stamp.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = DEFAULT_CONFIG
    today = date.today()
    athletes = get_athlete_repository().list_all()[: args.max_athletes]
    segments: List[DurabilitySegment] = []
    for athlete in athletes:
        try:
            data = data_source_for(athlete)
            summaries = data.summaries()
            reference = athlete_reference(data, summaries, today, config)
            if not reference.known:
                continue
            count = 0
            for activity_id in durability_activity_ids(summaries, today, config):
                stream = data.stream(activity_id)
                if stream is None:
                    continue
                found, _ = extract_segments(stream, reference, config.exposure,
                                            config.personalization, athlete_key=str(athlete.id))
                segments.extend(found)
                count += len(found)
            print(f"athlete {athlete.id}: {count} segments")
        except Exception as error:
            print(f"athlete {athlete.id}: FAILED — {error}", file=sys.stderr)

    try:
        result = calibrate(segments, config,
                           CalibrationSettings(validation_fraction=args.validation_fraction),
                           version=args.version)
    except ValueError as error:
        print(f"calibration impossible: {error}", file=sys.stderr)
        return 1
    with open(args.output, "w") as handle:
        handle.write(to_json(result))
    print(f"wrote {args.output}: {result.coefficients.to_dict()}")
    print(f"validation: {result.metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
