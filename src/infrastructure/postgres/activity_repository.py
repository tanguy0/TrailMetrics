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
    FEATURE_VERSION,
    GENERATED_COLUMNS,
    band_column,
    best_column,
)
from src.domain.ports.storage import ActivityRepository
from src.domain.progress.models import GRADIENT_BANDS, PR_DISTANCES
from src.infrastructure.postgres.pool import Database

# The oldest ``feature_version`` whose stored stream blob can rebuild a row
# locally, with no Strava call — see
# :mod:`src.usecases.refeaturize_athlete_activities`.
#
# A blob is written by the same import that stamps the row, so the row's version
# also says what the blob contains. ``watts`` only started being written at v2,
# and a v1 archive decodes it as NaN rather than failing (see
# :func:`~src.infrastructure.storage.codec.decode_stream`) — which would quietly
# turn a metered ride into an aero-model estimate. So v1 rows go back to Strava;
# v2 and up carry every stream the featurizer reads and can be rebuilt offline.
# Raise this the next time a bump adds a stream, in the same edit.
MIN_LOCAL_REBUILD_VERSION = 2

# Scalar feature columns that are real SQL columns.
_SCALAR_COLUMNS = [
    "distance_m", "elevation_gain_m", "moving_s", "elapsed_s", "gap_distance_m",
    "avg_hr", "max_hr", "avg_power_w_per_kg", "power_to_hr_per_kg",
    "avg_power_w_measured", "avg_power_w_modelled", "power_to_hr_measured",
    "relative_effort",
]

# Text column carrying power's provenance ("measured" / "estimated"). Not in
# _SCALAR_COLUMNS: every entry there is run through `_clean()`'s float()
# coercion, which a string would fail — wired alongside `summary_polyline`
# instead, both below.
_TEXT_COLUMNS = ["power_source"]

# Athlete-entered, not part of any Strava sync — deliberately kept out of
# _SCALAR_COLUMNS/_TEXT_COLUMNS so upsert_rows (the sync path) never inserts or
# updates them; see set_rpe_feeling for the only way these are written.
_MANUAL_COLUMNS = ["rpe", "feeling"]

_SUMMARY_COLUMNS = [
    "activity_id", "start_date", "sport_type", "has_streams",
    "distance_m", "moving_s", "relative_effort",
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
                   "has_streams", *_SCALAR_COLUMNS, *_TEXT_COLUMNS,
                   "features", "feature_version", "summary_polyline"]
        placeholders = ", ".join(["%s"] * len(columns))
        updates = ", ".join(
            f"{name} = excluded.{name}"
            for name in ("start_date", "sport_type", "has_streams",
                         *_SCALAR_COLUMNS, *_TEXT_COLUMNS, "features", "feature_version")
        )
        # The route is *coalesced* rather than overwritten: a re-featurize passes no
        # polyline, and that must not erase one already fetched on demand.
        updates += (
            ", summary_polyline = coalesce(excluded.summary_polyline, "
            "activities.summary_polyline)"
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
            *[row.get(name) or None for name in _TEXT_COLUMNS],
            json.dumps(generated),
            FEATURE_VERSION,
            row.get("summary_polyline") or None,
        )

    def route_polyline(self, athlete_id: int, activity_id: int) -> Optional[str]:
        row = self.db.fetch_one(
            "select summary_polyline from activities "
            "where athlete_id = %s and activity_id = %s",
            (athlete_id, activity_id),
        )
        return row["summary_polyline"] if row else None

    def set_route_polyline(
        self, athlete_id: int, activity_id: int, polyline: Optional[str]
    ) -> None:
        self.db.execute(
            "update activities set summary_polyline = %s, updated_at = now() "
            "where athlete_id = %s and activity_id = %s",
            (polyline, athlete_id, activity_id),
        )

    def set_rpe_feeling(
        self, athlete_id: int, activity_id: int, *,
        rpe: Optional[int] = None, feeling: Optional[str] = None,
    ) -> None:
        fields: List[str] = []
        params: List[Any] = []
        if rpe is not None:
            fields.append("rpe = %s")
            params.append(rpe)
        if feeling is not None:
            fields.append("feeling = %s")
            params.append(feeling)
        if not fields:
            return
        fields.append("updated_at = now()")
        self.db.execute(
            f"update activities set {', '.join(fields)} "
            f"where athlete_id = %s and activity_id = %s",
            (*params, athlete_id, activity_id),
        )

    def set_relative_efforts(
        self, athlete_id: int, values: Sequence[Tuple[int, Optional[float]]]
    ) -> int:
        """Update Relative Effort on rows that already exist.

        This is what backfills the column on a history imported before it existed.
        Relative Effort rides on Strava's *activity list*, which a sync walks anyway,
        so refreshing every stored activity costs no extra request — whereas
        re-featurizing them all would cost one per activity and blow the rate limit.

        ``coalesce`` so a value Strava has stopped reporting (an activity whose heart
        rate was removed) does not erase the number we already have.
        """
        cleaned = [(_clean(value), athlete_id, int(activity_id))
                   for activity_id, value in values if _clean(value) is not None]
        if not cleaned:
            return 0
        return self.db.execute_many(
            "update activities "
            "set relative_effort = coalesce(%s, relative_effort), updated_at = now() "
            "where athlete_id = %s and activity_id = %s",
            cleaned,
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
            *_SCALAR_COLUMNS, *_TEXT_COLUMNS, *_MANUAL_COLUMNS, "features",
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

    def stale_activities(self, athlete_id: int) -> List[Dict[str, Any]]:
        """Activities whose row was computed by an older featurizer, oldest first.

        The complement of :meth:`known_ids`, and it selects what a *rebuild*
        needs rather than a whole feature row: where the streams are, the version
        that wrote them (see :data:`MIN_LOCAL_REBUILD_VERSION`), and the summary
        totals that are all a streamless activity's row was ever made of.
        """
        rows = self.db.fetch_all(
            "select activity_id, start_date, sport_type, has_streams, "
            "stream_object, feature_version, distance_m, elevation_gain_m, "
            "moving_s, relative_effort "
            "from activities where athlete_id = %s and feature_version <> %s "
            "order by start_date",
            (athlete_id, FEATURE_VERSION),
        )
        return [
            {
                "activity_id": int(row["activity_id"]),
                "start_date": _naive(row["start_date"]),
                "sport_type": row["sport_type"],
                "has_streams": bool(row["has_streams"]),
                "stream_object": row["stream_object"],
                "feature_version": int(row["feature_version"] or 0),
                "distance_m": _clean(row["distance_m"]),
                "elevation_gain_m": _clean(row["elevation_gain_m"]),
                "moving_s": _clean(row["moving_s"]),
                "relative_effort": _clean(row["relative_effort"]),
            }
            for row in rows
        ]

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
    for name in (*_SCALAR_COLUMNS, *_TEXT_COLUMNS, *_MANUAL_COLUMNS):
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
