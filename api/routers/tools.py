"""The Tools tab (design/tagg/access.md § Outils): designed pages, not panel stacks.

* **Level assessment** — open to everyone; saving is an explicit act that needs
  Strava, and the saved estimate becomes the athlete's VMA (design/specs/level.md).
* **GAP profile** and **Durability profile** — Strava required. Each rates the
  runner against an average runner on the shared five-level scale
  (src/domain/assessment) and returns the chart it was read on, both computed
  from the same fitted models the race plan uses (and caches).
"""

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.deps import (
    STRAVA_NOT_CONNECTED,
    current_account,
    current_athlete,
    get_account_repository,
    get_athlete_repository,
    get_level_repository,
    language,
)
from api.routers.race_plan import _durability_model, _personal_curves
from src.domain.charts.ir import PlotOutput
from src.domain.durability.config import DEFAULT_CONFIG as DURABILITY_CONFIG
from src.domain.durability.personalization import POPULATION_ONLY
from src.domain.durability.assessment import assess as assess_durability
from src.domain.durability.assessment import profile_chart as profile_durability_chart
from src.domain.gap.assessment import assess as assess_gap, profile_chart
from src.domain.gap.reference_curves import balanced_runner
from src.domain.level import zones
from src.domain.level.estimate import LevelEstimate, LevelInputError, estimate, hr_max_or_none
from src.domain.ports.accounts import Account
from src.domain.ports.storage import Athlete
from src.translations import translate
from src.usecases.plan_race import PERSONAL_EFFICIENCY

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["tools"])



class EstimateRequest(BaseModel):
    method: str = Field(max_length=40)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    hr_max: Optional[float] = None


@router.get("/zones")
def zone_definitions() -> dict:
    """The zone tables — the one definition Home and the level tool both draw from."""
    return zones.definitions()


@router.post("/level/estimate")
def estimate_level(payload: EstimateRequest = Body(...), lang: str = Depends(language)) -> dict:
    """Estimate a VMA. Open to everyone and never saved: saving is its own act."""
    result, hr_max = _estimate(payload, lang)
    return {**_estimate_payload(result, hr_max, lang), "saved_at": None}


@router.post("/level/save")
def save_level(
    request: Request,
    payload: EstimateRequest = Body(...),
    lang: str = Depends(language),
    account: Account = Depends(current_account),
) -> dict:
    """Save an estimate as the account's level — its Home zones from then on.

    Recomputed from the inputs rather than taken from the client, so what is
    stored is what the tool computes. Needs Strava: the estimate becomes the
    athlete's VMA, and an account without Strava has no athlete to set it on.
    """
    if request.state.account_athlete_id is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=STRAVA_NOT_CONNECTED)
    result, hr_max = _estimate(payload, lang)
    body = _estimate_payload(result, hr_max, lang)
    saved = get_level_repository(account.id).save(
        result.method,
        {**payload.inputs, "hr_max": hr_max},
        {k: v for k, v in body.items() if k != "notes"},
    )
    _apply_to_athlete(account, result, hr_max)
    return {**body, "saved_at": saved["created_at"]}


def _estimate(payload: EstimateRequest, lang: str) -> Tuple[LevelEstimate, Optional[int]]:
    try:
        return estimate(payload.method, payload.inputs), hr_max_or_none(payload.hr_max)
    except LevelInputError as error:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=translate(f"ui.{error.key}", lang).format(**error.params),
        )


@router.get("/level/latest")
def latest_level(account: Account = Depends(current_account), lang: str = Depends(language)) -> dict:
    """The account's latest estimate (what Home's Zones card shows), or null."""
    return {"estimate": get_level_repository(account.id).latest()}


@router.get("/gap/summary")
def gap_summary(athlete: Athlete = Depends(current_athlete), lang: str = Depends(language)) -> dict:
    """The GAP profile: a level per terrain against the balanced runner.

    Read on the same personal curve the race plan uses (and caches), so the two
    tools always describe the same runner.
    """
    curve, reason = _personal_curves(athlete)(PERSONAL_EFFICIENCY)
    terrains = [t.to_dict() for t in assess_gap(curve, balanced_runner())]
    if curve is None:
        return {
            "available": False,
            "reason": translate(reason or "race_plan.reason.no_runs", lang),
            "terrains": terrains,
        }
    return {
        "available": True,
        "terrains": terrains,
        # The very curve the levels were read on, against the same reference.
        "chart": PlotOutput(charts=[profile_chart(curve, balanced_runner(), lang)]).to_dict()["charts"][0],
    }


@router.get("/durability/summary")
def durability_summary(
    athlete: Athlete = Depends(current_athlete), lang: str = Depends(language)
) -> dict:
    """The durability profile: a level per quality against the average runner.

    Read on the same fitted model the race plan uses (and caches).
    """
    model = _durability_model(athlete)()
    chart = profile_durability_chart(model, DURABILITY_CONFIG, lang)
    return {
        "available": model.confidence != POPULATION_ONLY,
        "qualities": [q.to_dict() for q in assess_durability(model)],
        "chart": PlotOutput(charts=[chart]).to_dict()["charts"][0] if chart else None,
    }


# --- Helpers -----------------------------------------------------------------

def _estimate_payload(result: LevelEstimate, hr_max: Optional[int], lang: str) -> dict:
    return {
        "method": result.method,
        "vma_kmh": result.vma_kmh,
        "vma_pace_s_per_km": result.vma_pace_s_per_km,
        "vdot": result.vdot,
        "confidence": result.confidence,
        "extras": result.extras,
        "zones": [vars(zone) for zone in result.zones],
        "hr_max": hr_max,
        "hr_zones": zones.hr_zone_ceilings(hr_max),
        "notes": [
            translate(f"ui.{note.key}", lang).format(**note.params) for note in result.notes
        ],
    }


def _apply_to_athlete(account: Account, result: LevelEstimate, hr_max: Optional[int]) -> None:
    """The latest estimate becomes the athlete's VMA (and HRmax when given) —
    the account's own athlete only, never one a coach is viewing. Paces set by
    hand on Home give way to it: the new estimate is the new reference."""
    athlete_id = get_account_repository().athlete_id_for(account.id)
    if athlete_id is None:
        return
    athletes = get_athlete_repository()
    athlete = athletes.get(athlete_id)
    if athlete is None:
        return
    athletes.set_zones(
        athlete.id,
        hr_zone1_end=athlete.hr_zone1_end,
        hr_zone2_end=athlete.hr_zone2_end,
        hr_zone3_end=athlete.hr_zone3_end,
        hr_zone4_end=athlete.hr_zone4_end,
        hr_max=hr_max if hr_max is not None else athlete.hr_max,
        vma_pace_s_per_km=result.vma_pace_s_per_km,
    )
    athletes.set_pace_overrides(athlete.id, None)
