"""Race plan ("Plan de course"): a pace profile for a course and a target time.

Public, like the blog: a visitor can upload a GPX and get a plan built on the
reference GAP curve, with no account and no database. A signed-in athlete gets the
same endpoint with their own curves on top — which is why auth here is *optional*
(:func:`_optional_identity`) rather than a dependency that 401s.

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
from datetime import date
from typing import List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from api.deps import (
    current_account,
    current_athlete_id,
    data_source_for,
    get_athlete_repository,
    get_caches,
    get_coaching_repository,
    get_planned_item_repository,
    get_plot_output_repository,
    get_race_plan_repository,
    language,
    optional_account,
)
from src.domain.charts.ir import ChartData, PlotOutput, Trace
from src.domain.models.gap import GapCurve
from src.domain.durability.config import DEFAULT_CONFIG as DURABILITY_CONFIG
from src.domain.durability.model import RaceWeather
from src.domain.durability.personalization import AthleteDurabilityModel
from src.domain.ports.accounts import Account
from src.domain.ports.storage import Athlete
from src.domain.race_plan.gpx import GpxError, parse_gpx
from src.domain.race_plan.planner import PlanError, build_course
from src.domain.race_plan.preview import course_preview
from src.translations import translate
from src.domain.durability.history import (
    durability_activity_ids,
    fit_athlete_durability,
)
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
    # Durability (cost drift over a long effort). Optional so plans saved before it
    # existed still load; weather left empty means neutral conditions.
    durability: bool = True
    temperature_start_c: Optional[float] = Field(default=None, ge=-40, le=55)
    temperature_end_c: Optional[float] = Field(default=None, ge=-40, le=55)
    relative_humidity_pct: Optional[float] = Field(default=None, ge=0, le=100)

    def weather(self) -> RaceWeather:
        return RaceWeather(
            temperature_start_c=self.temperature_start_c,
            temperature_end_c=self.temperature_end_c,
            relative_humidity_pct=self.relative_humidity_pct,
        )


class SavedPlanMeta(BaseModel):
    title: str = Field(default="", max_length=200)
    params: PlanParams
    # The race's date and weight as an objective; both optional ("not said").
    event_date: Optional[date] = None
    importance: Optional[Literal["primary", "secondary"]] = None


@router.get("/options")
def options(request: Request, lang: str = Depends(language)) -> dict:
    """Whether the caller is signed in, and which curves they can plan on."""
    signed_in = _optional_identity(request)[1] is not None
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
    account, athlete = _optional_identity(request)

    if gpx is not None:
        payload = _read_gpx(gpx, lang)
    elif plan_id and account is not None:
        # Saved plans belong to the account, not to Strava: an account without
        # Strava can reopen its plans too (on the reference curves).
        payload = get_race_plan_repository(account.id).gpx(plan_id)
        if payload is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    else:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail=translate("race_plan.error.no_gpx", lang))

    usecase = PlanRace(
        personal_curve=_personal_curves(athlete) if athlete else None,
        durability_model=_durability_model(athlete) if athlete else None,
    )
    try:
        result = usecase.execute(PlanRaceInput(
            gpx=payload,
            target_time_s=parsed.target_time_s,
            aid_stations_km=[s.km for s in parsed.aid_stations],
            aid_station_names=[s.name for s in parsed.aid_stations],
            start_clock_s=parsed.start_time_s,
            curve=parsed.curve,
            lang=lang,
            durability=parsed.durability,
            weather=parsed.weather(),
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
def list_saved(account: Account = Depends(current_account)) -> dict:
    repository = get_race_plan_repository(account.id)
    plans = repository.list()
    # Plans saved before thumbnails existed get theirs once, here.
    for plan in plans:
        if plan["preview"] is None:
            plan["preview"] = _backfill_preview(repository, plan["id"])
    return {"plans": plans}


@saved_router.post("", status_code=status.HTTP_201_CREATED)
def create_saved(
    request: Request,
    meta: str = Form(...),
    gpx: UploadFile = File(...),
    lang: str = Depends(language),
    account: Account = Depends(current_account),
) -> dict:
    parsed = _parse(meta, SavedPlanMeta)
    payload = _read_gpx(gpx, lang)
    distance, gain, preview = _course_stats(payload, lang)
    repository = get_race_plan_repository(account.id)
    created = repository.create(
        parsed.title.strip(), (gpx.filename or "")[:200], payload,
        parsed.params.model_dump(), distance, gain, preview,
        event_date=parsed.event_date, importance=parsed.importance,
    )
    return _sync_goal(request, account, repository, created, lang)


@saved_router.get("/{plan_id}")
def get_saved(plan_id: str, account: Account = Depends(current_account)) -> dict:
    saved = get_race_plan_repository(account.id).get(plan_id)
    if saved is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    return saved


@saved_router.patch("/{plan_id}")
def update_saved(
    request: Request,
    plan_id: str,
    meta: str = Form(...),
    gpx: Optional[UploadFile] = File(None),
    lang: str = Depends(language),
    account: Account = Depends(current_account),
) -> dict:
    """Replace a saved plan's inputs; a new GPX only when one is uploaded."""
    parsed = _parse(meta, SavedPlanMeta)
    repository = get_race_plan_repository(account.id)
    if gpx is not None:
        payload = _read_gpx(gpx, lang)
    else:
        payload = repository.gpx(plan_id)
        if payload is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    distance, gain, preview = _course_stats(payload, lang)
    updated = repository.update(
        plan_id,
        title=parsed.title.strip(),
        params=parsed.params.model_dump(),
        distance_m=distance,
        elevation_gain_m=gain,
        preview=preview,
        event_date=parsed.event_date,
        importance=parsed.importance,
        gpx=payload if gpx is not None else None,
        gpx_name=(gpx.filename or "")[:200] if gpx is not None else None,
    )
    if updated is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    return _sync_goal(request, account, repository, updated, lang)


@saved_router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved(
    request: Request, plan_id: str, account: Account = Depends(current_account)
) -> None:
    repository = get_race_plan_repository(account.id)
    saved = repository.get(plan_id)
    athlete_id = request.state.account_athlete_id
    # The goal this plan put on the diary goes with it.
    if saved is not None and saved["goal_item_id"] and athlete_id is not None:
        get_planned_item_repository(athlete_id).delete(saved["goal_item_id"])
    repository.delete(plan_id)


# --- Helpers -----------------------------------------------------------------

def _sync_goal(request: Request, account: Account, repository, saved: dict, lang: str) -> dict:
    """Keep a coached athlete's diary goal in step with this plan; return the plan.

    Only with a date *and* an importance, only for an account that is coached, and
    only on the account's own diary (never a coachee's a coach is viewing as) —
    which needs Strava, since the diary is keyed by the Strava athlete. Clearing
    the date or importance leaves an existing goal where it is.
    """
    athlete_id = request.state.account_athlete_id
    if (
        not saved["event_date"]
        or not saved["importance"]
        or athlete_id is None
        or not get_coaching_repository().is_coached(account.id)
    ):
        return saved
    items = get_planned_item_repository(athlete_id)
    title = saved["title"] or translate("ui.race_plan.untitled", lang)
    when = date.fromisoformat(saved["event_date"])
    goal = None
    if saved["goal_item_id"]:
        goal = items.update(
            saved["goal_item_id"], date=when, end_date=when, title=title,
            importance=saved["importance"],
        )
    if goal is None:
        goal = items.create("goal", when, title, "", importance=saved["importance"])
        repository.set_goal_item(saved["id"], goal["id"])
        saved = {**saved, "goal_item_id": goal["id"]}
    return saved


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


def _course_stats(payload: bytes, lang: str) -> Tuple[float, float, dict]:
    """``(distance, D+, preview)`` of a GPX — and the check that it is plannable at all."""
    try:
        points = parse_gpx(payload)
        course = build_course(points)
    except GpxError as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, detail=translate(error.reason_key, lang)
        )
    return course.total_m, course.elevation_gain()[0], course_preview(points, course)


def _backfill_preview(repository, plan_id: str) -> Optional[dict]:
    """Compute and store a saved plan's missing thumbnail; ``None`` if it cannot be."""
    payload = repository.gpx(plan_id)
    if payload is None:
        return None
    try:
        points = parse_gpx(payload)
        preview = course_preview(points, build_course(points))
    except GpxError:
        return None
    repository.set_preview(plan_id, preview)
    return preview


def _optional_identity(request: Request) -> Tuple[Optional[Account], Optional[Athlete]]:
    """``(account, athlete)`` of the caller, view-as included; ``None`` for what is missing.

    A visitor has neither; an account without Strava has no athlete.
    """
    try:
        account = optional_account(request)
    except HTTPException:
        # No database configured, which for this public endpoint just means
        # nobody can be signed in.
        return None, None
    if account is None:
        return None, None
    try:
        return account, get_athlete_repository().get(current_athlete_id(request, account))
    except HTTPException:
        return account, None


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


def _durability_model(athlete: Athlete):
    """A provider of this athlete's durability model, memoized in their warm caches.

    Keyed by today's date and the past-year long runs it reads, so a new run — or a
    run ageing out of the one-year window — makes a new key.
    """
    data = data_source_for(athlete)
    memo = get_caches(athlete.id).memo

    def provide() -> AthleteDurabilityModel:
        today = date.today()
        ids = durability_activity_ids(data.summaries(), today, DURABILITY_CONFIG)
        key = ("race_plan_durability", DURABILITY_CONFIG.population.version, today, ids)
        if key not in memo:
            memo[key] = fit_athlete_durability(data, today, DURABILITY_CONFIG, memo=memo)
        return memo[key]

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
