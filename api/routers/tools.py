"""The Tools tab (design/tagg/access.md § Outils): designed pages, not panel stacks.

* **Level assessment** — open to everyone; saving is an explicit act that needs
  Strava, and the saved estimate becomes the athlete's VMA (design/specs/level.md).
* **Slope profile** and **Durability** — Strava required. Their charts are the
  existing ``gap_curve`` and ``durability_curve`` plots, rendered by the client
  through ``/render/panel`` like Home's; this router only adds the headline
  tiles, computed from the same fitted models the race plan uses (and caches).
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.deps import (
    STRAVA_NOT_CONNECTED,
    current_account,
    current_athlete,
    data_source_for,
    get_account_repository,
    get_athlete_repository,
    get_level_repository,
    language,
)
from api.routers.race_plan import _durability_model, _personal_curves
from src.domain.durability.config import DEFAULT_CONFIG as DURABILITY_CONFIG
from src.domain.durability.personalization import POPULATION_ONLY
from src.domain.gap.reference_curves import balanced_runner
from src.domain.level import zones
from src.domain.level.estimate import LevelEstimate, LevelInputError, estimate, hr_max_or_none
from src.domain.plots.durability_curve import projected_extra_cost
from src.domain.ports.accounts import Account
from src.domain.ports.storage import Athlete
from src.translations import translate
from src.domain.dataset.sport import RUNNING_SPORT_TYPES

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tools", tags=["tools"])

# Gradient the slope tiles are read at: ±10 % (the curve's x is m of climb per km).
SLOPE_M_PER_KM = 100.0
# The flat-equivalent pace is read over this recent window of runs.
FLAT_PACE_DAYS = 84
DURABILITY_TILES_H = (2.0, 4.0)


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
    """Headline numbers of the slope profile: what ±10 % costs, against the reference."""
    curve, reason = _personal_curves(athlete)("efficiency")
    if curve is None:
        return {"available": False, "reason": translate(reason or "race_plan.reason.no_runs", lang)}
    reference = balanced_runner()

    def read(c, x: float) -> float:
        order = np.argsort(c.bin_centers)
        return float(np.interp(x, np.asarray(c.bin_centers)[order], np.asarray(c.means)[order]))

    up, down = read(curve, SLOPE_M_PER_KM), read(curve, -SLOPE_M_PER_KM)
    ref_up, ref_down = read(reference, SLOPE_M_PER_KM), read(reference, -SLOPE_M_PER_KM)
    return {
        "available": True,
        # Speed adjusters: GAP/speed. Above 1 a slope costs speed, below 1 it gives.
        "uphill_factor": round(up, 3),
        "downhill_factor": round(down, 3),
        "uphill_vs_reference_pct": round((up - ref_up) / ref_up * 100, 1),
        "downhill_vs_reference_pct": round((down - ref_down) / ref_down * 100, 1),
        "flat_pace_s_per_km": _flat_pace(athlete),
        "slope_pct": SLOPE_M_PER_KM / 10,
    }


@router.get("/durability/summary")
def durability_summary(athlete: Athlete = Depends(current_athlete)) -> dict:
    """Headline numbers of durability: extra cost after 2 h and 4 h, and confidence."""
    model = _durability_model(athlete)()
    hours = np.array([0.0, *DURABILITY_TILES_H])
    extra = projected_extra_cost(model, DURABILITY_CONFIG, hours)
    population = projected_extra_cost(model, DURABILITY_CONFIG, hours, model.population)
    return {
        "confidence": model.confidence,
        "personal": model.confidence != POPULATION_ONLY,
        "extra_cost_pct": {f"{h:g}h": round(float(v), 1) for h, v in zip(DURABILITY_TILES_H, extra[1:])},
        "population_extra_cost_pct": {
            f"{h:g}h": round(float(v), 1) for h, v in zip(DURABILITY_TILES_H, population[1:])
        },
        "n_activities": model.n_activities,
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
    the account's own athlete only, never one a coach is viewing."""
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


def _flat_pace(athlete: Athlete) -> Optional[float]:
    """Gradient-adjusted pace over the recent runs: the pace on the flat equivalent."""
    data = data_source_for(athlete)
    since = date.today() - timedelta(days=FLAT_PACE_DAYS)
    ids: List[int] = [
        s.activity_id for s in data.summaries()
        if s.sport_type in RUNNING_SPORT_TYPES and s.start_date.date() >= since
    ]
    if not ids:
        return None
    frame = data.features(ids)
    if frame.empty or "gap_distance_m" not in frame:
        return None
    distance = float(frame["gap_distance_m"].fillna(0).sum())
    moving = float(frame.loc[frame["gap_distance_m"].notna(), "moving_s"].sum())
    return round(moving / distance * 1000, 1) if distance > 0 else None
