"""Race plan ("Plan de course"): a pace profile for a course and a target time.

Public, like the blog: a visitor can upload a GPX and get a plan built on the
reference GAP curve, with no account and no database. A signed-in athlete gets the
same endpoint with their own curves on top — which is why auth here is *optional*
(:func:`_optional_athlete`) rather than a dependency that 401s.

A visitor's GPX is re-sent with every request rather than stored: it is a few
hundred kilobytes, parsing it is milliseconds next to the plan itself, and their
course then never has to live anywhere on our side. A signed-in athlete can also
*save* a plan (``/race-plans``, the page-like list): title, GPX and parameters are
stored, and planning a saved one names it by ``plan_id`` instead of re-uploading.

Personal curves are the only slow part — a first plan downloads and preprocesses
the athlete's recent runs — so each fitted curve is memoized in the athlete's
warm caches and persisted in ``plot_outputs``, keyed by the activity ids it was
fitted on: a new run makes a new key, exactly like a cached plot.
"""

import json
import logging
from typing import List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from api.deps import (
    current_athlete,
    current_athlete_id,
    data_source_for,
    get_athlete_repository,
    get_caches,
    get_plot_output_repository,
    get_race_plan_repository,
    language,
)
from src.domain.charts.ir import ChartData, PlotOutput, Trace
from src.domain.models.gap import GapCurve
from src.domain.ports.storage import Athlete
from src.domain.race_plan.gpx import GpxError, parse_gpx
from src.domain.race_plan.planner import PlanError, build_course
from src.translations import translate
from src.usecases.plan_race import (
    PlanRace,
    PlanRaceInput,
    curve_options,
    fit_personal_curve,
    running_activity_ids,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/race-plan", tags=["race-plan"])
# The saved plans, scoped to the signed-in athlete like any other of their data.
saved_router = APIRouter(prefix="/race-plans", tags=["race-plan"])

# A 170 km track at one point per second of a slow recording is well under this.
MAX_GPX_BYTES = 15 * 1024 * 1024
MAX_AID_STATIONS = 100
# Bump when the fitting recipe changes, so stored curves miss once.
CURVE_VERSION = 1


class AidStation(BaseModel):
    km: float
    name: str = Field(default="", max_length=80)


class PlanParams(BaseModel):
    target_time_s: float = Field(gt=0, le=10 * 86400)
    aid_stations: List[AidStation] = Field(default_factory=list, max_length=MAX_AID_STATIONS)
    # Seconds after midnight; only adds a wall-clock column.
    start_time_s: Optional[float] = Field(default=None, ge=0, lt=86400)
    curve: Optional[str] = None


class SavedPlanMeta(BaseModel):
    title: str = Field(default="", max_length=200)
    params: PlanParams


@router.get("/options")
def options(request: Request, lang: str = Depends(language)) -> dict:
    """Whether the caller is signed in, and which curves they can plan on."""
    signed_in = _optional_athlete(request) is not None
    return {"signed_in": signed_in, "curves": curve_options(signed_in, lang)}


@router.post("")
def plan(
    request: Request,
    params: str = Form(...),
    gpx: Optional[UploadFile] = File(None),
    plan_id: Optional[str] = Form(None),
    lang: str = Depends(language),
) -> dict:
    """Plan a course: an uploaded ``gpx``, or the stored GPX of saved ``plan_id``."""
    parsed = _parse(params, PlanParams)
    athlete = _optional_athlete(request)

    if gpx is not None:
        payload = _read_gpx(gpx, lang)
    elif plan_id and athlete is not None:
        payload = get_race_plan_repository(athlete.id).gpx(plan_id)
        if payload is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=translate("race_plan.error.no_gpx", lang))

    usecase = PlanRace(personal_curve=_personal_curves(athlete) if athlete else None)
    try:
        result = usecase.execute(PlanRaceInput(
            gpx=payload,
            target_time_s=parsed.target_time_s,
            aid_stations_km=[s.km for s in parsed.aid_stations],
            aid_station_names=[s.name for s in parsed.aid_stations],
            start_clock_s=parsed.start_time_s,
            curve=parsed.curve,
            lang=lang,
        ))
    except (GpxError, PlanError) as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail=translate(error.reason_key, lang)
        )

    return {
        **result.to_dict(),
        "signed_in": athlete is not None,
        "curves": curve_options(athlete is not None, lang),
    }


# --- Saved plans -------------------------------------------------------------

@saved_router.get("")
def list_saved(athlete: Athlete = Depends(current_athlete)) -> dict:
    return {"plans": get_race_plan_repository(athlete.id).list()}


@saved_router.post("", status_code=status.HTTP_201_CREATED)
def create_saved(
    meta: str = Form(...),
    gpx: UploadFile = File(...),
    lang: str = Depends(language),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    parsed = _parse(meta, SavedPlanMeta)
    payload = _read_gpx(gpx, lang)
    distance, gain = _course_stats(payload, lang)
    return get_race_plan_repository(athlete.id).create(
        parsed.title.strip(), (gpx.filename or "")[:200], payload,
        parsed.params.model_dump(), distance, gain,
    )


@saved_router.get("/{plan_id}")
def get_saved(plan_id: str, athlete: Athlete = Depends(current_athlete)) -> dict:
    saved = get_race_plan_repository(athlete.id).get(plan_id)
    if saved is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    return saved


@saved_router.patch("/{plan_id}")
def update_saved(
    plan_id: str,
    meta: str = Form(...),
    gpx: Optional[UploadFile] = File(None),
    lang: str = Depends(language),
    athlete: Athlete = Depends(current_athlete),
) -> dict:
    """Replace a saved plan's inputs; a new GPX only when one is uploaded."""
    parsed = _parse(meta, SavedPlanMeta)
    repository = get_race_plan_repository(athlete.id)
    if gpx is not None:
        payload = _read_gpx(gpx, lang)
    else:
        payload = repository.gpx(plan_id)
        if payload is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    distance, gain = _course_stats(payload, lang)
    updated = repository.update(
        plan_id,
        title=parsed.title.strip(),
        params=parsed.params.model_dump(),
        distance_m=distance,
        elevation_gain_m=gain,
        gpx=payload if gpx is not None else None,
        gpx_name=(gpx.filename or "")[:200] if gpx is not None else None,
    )
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    return updated


@saved_router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved(plan_id: str, athlete: Athlete = Depends(current_athlete)) -> None:
    get_race_plan_repository(athlete.id).delete(plan_id)


# --- Helpers -----------------------------------------------------------------

def _parse(raw: str, model):
    try:
        return model.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as error:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Bad params: {error}")


def _read_gpx(upload: UploadFile, lang: str) -> bytes:
    payload = upload.file.read(MAX_GPX_BYTES + 1)
    if len(payload) > MAX_GPX_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=translate("race_plan.error.gpx_too_large", lang),
        )
    return payload


def _course_stats(payload: bytes, lang: str) -> Tuple[float, float]:
    """``(distance, D+)`` of a GPX — and the check that it is plannable at all."""
    try:
        course = build_course(parse_gpx(payload))
    except GpxError as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail=translate(error.reason_key, lang)
        )
    return course.total_m, course.elevation_gain()[0]


def _optional_athlete(request: Request) -> Optional[Athlete]:
    """The signed-in athlete (view-as included), or ``None`` for a visitor."""
    try:
        athlete_id = current_athlete_id(request)
        return get_athlete_repository().get(athlete_id)
    except HTTPException:
        # Not signed in — or no database configured, which for this public
        # endpoint just means nobody can be.
        return None


def _personal_curves(athlete: Athlete):
    """``model → (curve, reason)`` for this athlete, memoized and persisted."""
    data = data_source_for(athlete)
    memo = get_caches(athlete.id).memo

    def provide(model: str) -> Tuple[Optional[GapCurve], Optional[str]]:
        ids = running_activity_ids(data)
        key = ("race_plan_curve", CURVE_VERSION, model, ids)
        if key in memo:
            return memo[key]

        signature = f"race_plan_curve|v{CURVE_VERSION}|{model}|{','.join(map(str, ids))}"
        repository = get_plot_output_repository(athlete.id)
        try:
            stored = repository.get(signature)
        except Exception as error:
            logger.warning("could not read stored race-plan curve: %s", error)
            stored = None
        if stored is not None and stored.charts and stored.charts[0].traces:
            trace = stored.charts[0].traces[0]
            result = (_curve_from_trace(trace), None)
            memo[key] = result
            return result

        result = fit_personal_curve(data, ids, model, memo)
        memo[key] = result
        if result[0] is not None:
            try:
                repository.put(signature, "race_plan_curve", _curve_as_output(result[0]))
            except Exception as error:
                logger.warning("could not store race-plan curve: %s", error)
        return result

    return provide


# A curve rides in ``plot_outputs`` as a one-trace chart: the table's payload is the
# chart IR, and a curve is exactly an x/y series.
def _curve_as_output(curve: GapCurve) -> PlotOutput:
    return PlotOutput(charts=[ChartData(traces=[Trace(
        name="gap_curve",
        x=[float(v) for v in curve.bin_centers],
        y=[float(v) for v in curve.means],
    )])])


def _curve_from_trace(trace: Trace) -> GapCurve:
    n = len(trace.x)
    return GapCurve(
        bin_centers=np.asarray(trace.x, dtype=float),
        means=np.asarray(trace.y, dtype=float),
        stds=np.zeros(n),
        counts=np.ones(n, dtype=int),
    )
