"""The Home screen's summary: who the athlete is, and what their history adds up to.

One endpoint rather than several, because the screen is one screenful — a second
round-trip per widget would only add latency and failure modes.

Everything here is read off the stored feature rows, including the personal
records. That is worth stating plainly: ``best_<distance>`` is computed once per
activity at import time and kept in the ``features`` JSONB, so "current records"
is a minimum over columns, not a scan of per-second streams. A full history
therefore costs one query.

Numbers go out raw — metres, seconds, kilograms — and the browser formats them,
the same contract the chart IR uses.
"""

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence

from fastapi import APIRouter, Depends

from api.deps import current_athlete, get_activity_repository, get_token_service
from src.domain.dataset.features import best_column
from src.domain.dataset.sport import RUNNING_SPORT_TYPES
from src.domain.geo.polyline import decode as decode_polyline
from src.domain.ports.storage import Athlete
from src.domain.progress.models import PR_DISTANCES
from src.infrastructure.strava.strava_client import StravaClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/home", tags=["home"])

# The record ladder shown on the Home screen. A subset of PR_DISTANCES: the
# distances a running history is actually likely to contain, so the widget isn't
# mostly empty rows. Ultra distances stay available in the records plots.
HOME_PR_LABELS = ["5 km", "10 km", "Semi", "Marathon"]

_DAYS_PER_YEAR = 365.2425

# Fastest speed a stored effort may imply before it is treated as corrupt, in m/s.
#
# A record is a *minimum* over the whole history, which makes it the single value
# most exposed to a bad split: one activity whose cumulative distance jumped from
# GPS jitter yields an impossibly fast effort, and the minimum then reports that
# artefact forever. 10 m/s is 36 km/h — beyond the 100 m world record's average
# pace, so nothing genuine is ever discarded, and the marathon bound this implies
# (1h10) stays well clear of the real record.
#
# This is a display guard, not a fix: the bad `best_*` value stays in the feature
# row, and the honest repair belongs in the preprocessing that produced it.
_MAX_PLAUSIBLE_SPEED_MS = 10.0

_METRES_BY_LABEL = {label: metres for label, metres in PR_DISTANCES}


@router.get("/last-activity/route")
def last_activity_route(athlete: Athlete = Depends(current_athlete)) -> dict:
    """The latest activity's route as coordinates, for the map.

    Any sport — the map just traces wherever the athlete last went, unlike the
    running-only volume/PR stats in :func:`summary`.
    """
    rows = get_activity_repository().summaries(athlete.id)
    if not rows:
        return {"activity_id": None, "points": [], "source": "none"}
    return resolve_activity_route(athlete, int(rows[-1]["activity_id"]))


def resolve_activity_route(athlete: Athlete, activity_id: int) -> dict:
    """One activity's route as coordinates, for the map.

    Shared by the Home screen's latest-activity widget and the Training calendar's
    click-to-open session view — any activity, not just the latest.

    A network round-trip on purpose: routes were not stored before this feature
    existed, so an older activity may have none yet. When Strava has to be asked,
    the result is written back, so the call happens at most once per activity
    rather than on every view.

    A missing route is a normal outcome — treadmill runs and manual entries have
    none — so this returns an empty ``points`` list rather than a 404.
    """
    activities = get_activity_repository()
    encoded = activities.route_polyline(athlete.id, activity_id)
    source = "stored"

    if not encoded:
        try:
            stravalib_client = get_token_service().client_for(athlete.id)
            if stravalib_client is None:
                return {"activity_id": activity_id, "points": [],
                        "source": "unavailable"}
            encoded = StravaClient(stravalib_client).fetch_route_polyline(activity_id)
            source = "strava"
        except Exception as error:
            # An unreachable or rate-limited Strava costs the map, nothing else.
            logger.warning("could not fetch route for %s: %s", activity_id, error)
            return {"activity_id": activity_id, "points": [], "source": "unavailable"}
        # Cache even a confirmed absence? No: an empty string is indistinguishable
        # from "not fetched yet", and re-asking once per view of an indoor run is
        # cheaper than a schema flag for it.
        if encoded:
            activities.set_route_polyline(athlete.id, activity_id, encoded)

    points = decode_polyline(encoded) if encoded else []
    return {
        "activity_id": activity_id,
        "points": [[lat, lng] for lat, lng in points],
        "source": source if points else "none",
    }


@router.get("/summary")
def summary(athlete: Athlete = Depends(current_athlete)) -> dict:
    """Profile totals, body fields, current records and the latest activity.

    Volume totals, the PR ladder, and "latest activity" answer different
    questions. Cycling is now imported too (see
    ``sync_athlete_activities.DEFAULT_SPORT_TYPES``), but the totals and PR
    ladder were built for running and a ride's numbers aren't comparable to a
    run's; mixing them in here would silently corrupt "furthest run" into
    "furthest anything" the moment the athlete's next ride is longer than their
    longest run. Cycling gets its own read of the same history through the
    Analysis section instead. "Latest activity" has no such comparability
    problem — it is just whatever the athlete did most recently, in any sport.
    """
    all_rows = get_activity_repository().rows(athlete.id)
    rows = _running_only(all_rows)
    today = date.today()

    return {
        "profile": _profile(rows, athlete.weight_kg),
        "health": _health(athlete, rows, today),
        "records": _records(rows),
        "last_activity": _last_activity(all_rows, athlete.weight_kg),
    }


def _running_only(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("sport_type") in RUNNING_SPORT_TYPES]


def _profile(rows: Sequence[Dict[str, Any]], weight_kg: Optional[float] = None) -> Dict[str, Any]:
    """Volume totals and the two 'biggest ever' activities."""
    distances = [(_num(r.get("distance_m")), r) for r in rows]
    with_distance = [(value, r) for value, r in distances if value is not None]
    moving = [(_num(r.get("moving_s")), r) for r in rows]
    with_moving = [(value, r) for value, r in moving if value is not None]

    furthest = max(with_distance, key=lambda pair: pair[0], default=None)
    longest = max(with_moving, key=lambda pair: pair[0], default=None)

    return {
        "activity_count": len(rows),
        "oldest_activity": _iso(_row_date(rows[0])) if rows else None,
        "newest_activity": _iso(_row_date(rows[-1])) if rows else None,
        "total_distance_m": sum(value for value, _ in with_distance),
        "total_elevation_gain_m": sum(
            value for value in (_num(r.get("elevation_gain_m")) for r in rows)
            if value is not None
        ),
        "total_moving_s": sum(value for value, _ in with_moving),
        # "Furthest" is by distance, "longest" is by time — on hilly terrain they
        # are routinely different activities, so both are worth showing.
        "furthest_activity": _activity_card(furthest[1], weight_kg) if furthest else None,
        "longest_activity": _activity_card(longest[1], weight_kg) if longest else None,
    }


def _health(
    athlete: Athlete,
    rows: Sequence[Dict[str, Any]],
    today: date,
) -> Dict[str, Any]:
    """Body fields plus how long this history spans.

    ``experience_years`` measures from the first recorded activity to today, not to
    the last one: someone who ran for eight years and took this month off has eight
    years of experience, not none.
    """
    first = _row_date(rows[0]) if rows else None
    experience_days = (today - first.date()).days if first else None

    return {
        "age": athlete.age_on(today),
        "birthdate": athlete.birthdate.isoformat() if athlete.birthdate else None,
        "weight_kg": athlete.weight_kg,
        "height_cm": athlete.height_cm,
        "experience_years": (
            round(max(experience_days, 0) / _DAYS_PER_YEAR, 1)
            if experience_days is not None else None
        ),
        "first_activity": _iso(first),
    }


def _records(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fastest stored effort at each Home distance, with the run that set it.

    Distances no activity ever covered are omitted rather than returned empty — a
    row of dashes tells the reader nothing they cannot see from its absence. Efforts
    implying an impossible speed are dropped too; see
    :data:`_MAX_PLAUSIBLE_SPEED_MS` for why a minimum needs that guard.
    """
    records: List[Dict[str, Any]] = []
    for label in HOME_PR_LABELS:
        column = best_column(label)
        floor = _METRES_BY_LABEL[label] / _MAX_PLAUSIBLE_SPEED_MS
        efforts = [
            (value, row)
            for value, row in ((_num(r.get(column)), r) for r in rows)
            if value is not None and value >= floor
        ]
        if not efforts:
            continue
        seconds, row = min(efforts, key=lambda pair: pair[0])
        records.append({
            "label": label,
            "seconds": seconds,
            "set_on": _iso(_row_date(row)),
            "activity_id": row.get("activity_id"),
        })
    return records


def _last_activity(
    rows: Sequence[Dict[str, Any]], weight_kg: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """The most recent activity. Rows arrive oldest-first, so it is the last one."""
    return _activity_card(rows[-1], weight_kg) if rows else None


def _activity_card(row: Dict[str, Any], weight_kg: Optional[float] = None) -> Dict[str, Any]:
    """The fields the Home widgets and the training calendar show for one activity."""
    return {
        "activity_id": row.get("activity_id"),
        "date": _iso(_row_date(row)),
        "sport_type": row.get("sport_type"),
        "has_streams": bool(row.get("has_streams")),
        "distance_m": _num(row.get("distance_m")),
        "elevation_gain_m": _num(row.get("elevation_gain_m")),
        "moving_s": _num(row.get("moving_s")),
        "avg_hr": _num(row.get("avg_hr")),
        "avg_power_w": _avg_power_w(row, weight_kg),
        "power_source": row.get("power_source"),
    }


def _avg_power_w(row: Dict[str, Any], weight_kg: Optional[float]) -> Optional[float]:
    """Real watts, else cycling's modelled absolute figure, else running's modelled
    per-kg figure scaled by the current weight — mirrors the fallback in
    :func:`src.domain.dataset.features.apply_mass`, which this raw-dict-based
    endpoint doesn't go through."""
    measured = _num(row.get("avg_power_w_measured"))
    if measured is not None:
        return measured
    modelled_cycling = _num(row.get("avg_power_w_modelled"))
    if modelled_cycling is not None:
        return modelled_cycling
    per_kg = _num(row.get("avg_power_w_per_kg"))
    if per_kg is not None and weight_kg:
        return per_kg * weight_kg
    return None


def _row_date(row: Dict[str, Any]) -> Optional[datetime]:
    value = row.get("date")
    return value if isinstance(value, datetime) else None


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _num(value: Any) -> Optional[float]:
    """A float, or ``None`` for anything unusable — including NaN.

    Summary-only activities store NaN for the stream-derived columns, and NaN
    survives ``json.dumps`` as the literal ``NaN``, which ``JSON.parse`` rejects.
    Filtering here keeps that out of every response.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN != NaN
