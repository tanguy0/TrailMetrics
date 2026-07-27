"""Postgres :class:`ActivityRepository` — one feature row per activity.

The read path is what makes the app feel fast: a whole page of trends over ten
years of running is a single indexed query returning a few hundred small rows,
with no per-second data touched. The generated column families (time per gradient
band, best effort per distance) round-trip through one JSONB column, so adding a
PR distance in Python needs no migration.
"""

import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.domain.dataset.features import (
    GENERATED_COLUMNS,
    band_column,
    best_column,
)
from src.domain.ports.storage import ActivityRepository
from src.domain.progress.models import GRADIENT_BANDS, PR_DISTANCES
from src.infrastructure.postgres.pool import Database

# Bump when build_activity_features changes in a way that invalidates stored rows.
FEATURE_VERSION = 1

# Scalar feature columns that are real SQL columns.
_SCALAR_COLUMNS = [
    "distance_m", "elevation_gain_m", "moving_s", "elapsed_s", "gap_distance_m",
    "avg_hr", "max_hr", "avg_power_w_per_kg", "power_to_hr_per_kg",
]

_SUMMARY_COLUMNS = [
    "activity_id", "start_date", "sport_type", "has_streams",
    "distance_m", "moving_s",
]


class PostgresActivityRepository(ActivityRepository):
    def __init__(self, db: Database):
        self.db = db

    # --- Writes ------------------------------------------------------------

    def upsert_rows(self, athlete_id: int, rows: Sequence[Dict[str, Any]]) -> int:
        """Insert or replace feature rows.

        ``stream_object`` is left alone on conflict: the sync writes the blob path
        separately, and re-computing features must not forget where the streams are.
        """
        if not rows:
            return 0
        columns = ["athlete_id", "activity_id", "start_date", "sport_type",
                   "has_streams", *_SCALAR_COLUMNS, "features", "feature_version"]
        placeholders = ", ".join(["%s"] * len(columns))
        updates = ", ".join(
            f"{name} = excluded.{name}"
            for name in ("start_date", "sport_type", "has_streams",
                         *_SCALAR_COLUMNS, "features", "feature_version")
        )
        sql = (
            f"insert into activities ({', '.join(columns)}) "
            f"values ({placeholders}) "
            f"on conflict (athlete_id, activity_id) do update set "
            f"{updates}, updated_at = now()"
        )
        return self.db.execute_many(
            sql, [self._to_params(athlete_id, row) for row in rows]
        )

    @staticmethod
    def _to_params(athlete_id: int, row: Dict[str, Any]) -> tuple:
        generated = {
            name: _clean(row.get(name))
            for name in GENERATED_COLUMNS
        }
        return (
            athlete_id,
            int(row["activity_id"]),
            row["date"],
            str(row.get("sport_type") or ""),
            bool(row.get("has_streams", False)),
            *[_clean(row.get(name)) for name in _SCALAR_COLUMNS],
            json.dumps(generated),
            FEATURE_VERSION,
        )

    def set_stream_object(
        self, athlete_id: int, activity_id: int, object_path: Optional[str]
    ) -> None:
        self.db.execute(
            "update activities set stream_object = %s, updated_at = now() "
            "where athlete_id = %s and activity_id = %s",
            (object_path, athlete_id, activity_id),
        )

    # --- Reads -------------------------------------------------------------

    def summaries(self, athlete_id: int) -> List[Dict[str, Any]]:
        return self.db.fetch_all(
            f"select {', '.join(_SUMMARY_COLUMNS)} from activities "
            f"where athlete_id = %s order by start_date",
            (athlete_id,),
        )

    def rows(
        self, athlete_id: int, activity_ids: Optional[Sequence[int]] = None
    ) -> List[Dict[str, Any]]:
        selected = ", ".join([
            "activity_id", "start_date", "sport_type", "has_streams",
            *_SCALAR_COLUMNS, "features",
        ])
        if activity_ids is None:
            raw = self.db.fetch_all(
                f"select {selected} from activities where athlete_id = %s "
                f"order by start_date",
                (athlete_id,),
            )
        else:
            ids = [int(i) for i in activity_ids]
            if not ids:
                return []
            raw = self.db.fetch_all(
                f"select {selected} from activities "
                f"where athlete_id = %s and activity_id = any(%s) "
                f"order by start_date",
                (athlete_id, ids),
            )
        return [_to_feature_row(row) for row in raw]

    def known_ids(self, athlete_id: int) -> set:
        rows = self.db.fetch_all(
            "select activity_id from activities where athlete_id = %s "
            "and feature_version = %s",
            (athlete_id, FEATURE_VERSION),
        )
        return {int(row["activity_id"]) for row in rows}

    def date_range(self, athlete_id: int) -> Optional[Tuple[datetime, datetime]]:
        row = self.db.fetch_one(
            "select min(start_date) as oldest, max(start_date) as newest "
            "from activities where athlete_id = %s",
            (athlete_id,),
        )
        if not row or row["oldest"] is None:
            return None
        return row["oldest"], row["newest"]

    def stream_object(self, athlete_id: int, activity_id: int) -> Optional[str]:
        row = self.db.fetch_one(
            "select stream_object from activities "
            "where athlete_id = %s and activity_id = %s",
            (athlete_id, int(activity_id)),
        )
        return row["stream_object"] if row else None


def _to_feature_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """A SQL row back into the shape :mod:`src.domain.dataset.features` produces."""
    generated = row.get("features") or {}
    if isinstance(generated, str):
        generated = json.loads(generated)
    out: Dict[str, Any] = {
        "activity_id": int(row["activity_id"]),
        # The feature table calls it `date`; SQL calls it start_date.
        "date": _naive(row["start_date"]),
        "sport_type": row["sport_type"],
        "has_streams": bool(row["has_streams"]),
    }
    for name in _SCALAR_COLUMNS:
        out[name] = row.get(name)
    for key, _, _ in GRADIENT_BANDS:
        column = band_column(key)
        out[column] = generated.get(column)
    for label, _ in PR_DISTANCES:
        column = best_column(label)
        out[column] = generated.get(column)
    return out


def _naive(value: datetime) -> datetime:
    """Binning and window comparisons are all naive; strip tzinfo on the way out."""
    return value.replace(tzinfo=None) if value.tzinfo else value


def _clean(value: Any) -> Optional[float]:
    """NaN is not valid JSON and means "unknown" here, so it becomes NULL."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number
