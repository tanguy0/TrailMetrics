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

Personal models are the slow part — a fit downloads and preprocesses the
athlete's recent runs — so they are stored and only refitted on request
(:mod:`api.athlete_models`, the ``refit`` field). A saved plan also keeps its
result, written on save and on recompute, so it opens without being planned again.
"""

import json
import logging
from datetime import date
from typing import List, Literal, Optional, Tuple

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from api.athlete_models import AthleteModels
from api.deps import (
    current_account,
    current_athlete_id,
    get_athlete_repository,
    get_coaching_repository,
    get_planned_item_repository,
    get_race_plan_repository,
    language,
    optional_account,
)
from src.domain.durability.model import RaceWeather
from src.domain.ports.accounts import Account
from src.domain.ports.storage import Athlete
from src.domain.race_plan.gpx import GpxError, parse_gpx
from src.domain.race_plan.planner import PlanError, build_course
from src.domain.race_plan.preview import course_preview
from src.translations import translate
from src.usecases.plan_race import PlanRace, PlanRaceInput, curve_options

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/race-plan", tags=["race-plan"])
# The saved plans, scoped to the signed-in athlete like any other of their data.
saved_router = APIRouter(prefix="/race-plans", tags=["race-plan"])

# A 170 km track at one point per second of a slow recording is well under this.
MAX_GPX_BYTES = 15 * 1024 * 1024
MAX_AID_STATIONS = 100


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
    refit: bool = Form(False),
    lang: str = Depends(language),
) -> dict:
    """Plan a course: an uploaded ``gpx``, or the stored GPX of saved ``plan_id``.

    Never stored: saving is its own act. ``refit`` refits the athlete's models on
    their latest runs first (the Recompute button of a plan not yet saved).
    """
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

    return _with_options(_compute(parsed, payload, athlete, lang, refit), athlete, lang)


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
    refit: bool = Form(False),
    lang: str = Depends(language),
    account: Account = Depends(current_account),
) -> dict:
    """Save a new plan — its inputs, and its result as computed now."""
    parsed = _parse(meta, SavedPlanMeta)
    payload = _read_gpx(gpx, lang)
    distance, gain, preview = _course_stats(payload, lang)
    repository = get_race_plan_repository(account.id)
    created = repository.create(
        parsed.title.strip(), (gpx.filename or "")[:200], payload,
        parsed.params.model_dump(), distance, gain, preview,
        event_date=parsed.event_date, importance=parsed.importance,
    )
    saved = _sync_goal(request, account, repository, created, lang)
    return _store_result(request, repository, saved, parsed.params, payload, lang, refit)


@saved_router.get("/{plan_id}")
def get_saved(
    request: Request,
    plan_id: str,
    lang: str = Depends(language),
    account: Account = Depends(current_account),
) -> dict:
    """A saved plan with its stored result — planned once here only when it has
    none yet (saved before results were kept) or it is in another language."""
    repository = get_race_plan_repository(account.id)
    saved = repository.get(plan_id)
    if saved is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="race plan not found")
    stored = repository.result(plan_id)
    if stored is not None and stored["lang"] == lang:
        athlete = _optional_identity(request)[1]
        return {**saved, "result": _with_options(stored["result"], athlete, lang)}
    payload = repository.gpx(plan_id)
    params = PlanParams.model_validate(saved["params"])
    return _store_result(request, repository, saved, params, payload, lang, refit=False)


@saved_router.patch("/{plan_id}")
def update_saved(
    request: Request,
    plan_id: str,
    meta: str = Form(...),
    gpx: Optional[UploadFile] = File(None),
    refit: bool = Form(False),
    lang: str = Depends(language),
    account: Account = Depends(current_account),
) -> dict:
    """Replace a saved plan's inputs (a new GPX only when one is uploaded) and its
    result, computed on them now — on refitted models when ``refit`` (Recompute)."""
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
    saved = _sync_goal(request, account, repository, updated, lang)
    return _store_result(request, repository, saved, parsed.params, payload, lang, refit)


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

def _compute(parsed: PlanParams, payload: bytes, athlete: Optional[Athlete], lang: str,
             refit: bool = False) -> dict:
    """The plan of ``payload`` for ``parsed``, on the athlete's stored models if any."""
    models = AthleteModels(athlete, refit=refit) if athlete else None
    usecase = PlanRace(
        personal_curve=models.gap_curve if models else None,
        durability_model=models.durability if models else None,
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
    return result.to_dict()


def _with_options(result: dict, athlete: Optional[Athlete], lang: str) -> dict:
    """A plan as the screen takes it: plus who is asking and the curves they have."""
    return {
        **result,
        "signed_in": athlete is not None,
        "curves": curve_options(athlete is not None, lang),
    }


def _store_result(request: Request, repository, saved: dict, params: PlanParams,
                  payload: bytes, lang: str, refit: bool) -> dict:
    """Compute a saved plan, store the result with it, and return both.

    A plan that cannot be computed (a target the course rules out) is still saved —
    it just has no result until its inputs change.
    """
    athlete = _optional_identity(request)[1]
    try:
        result = _compute(params, payload, athlete, lang, refit)
    except HTTPException as error:
        logger.info("saved plan %s has no result: %s", saved["id"], error.detail)
        result = None
    computed_at = repository.set_result(saved["id"], result, lang if result else None)
    return {
        **saved,
        "computed_at": computed_at,
        "result": _with_options(result, athlete, lang) if result else None,
    }


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
