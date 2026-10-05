"""An athlete's fitted models as the Tools read them: stored, refitted on request.

A personal GAP curve and the durability model each read the athlete's recent
history stream by stream, so they are fitted once and kept in ``athlete_models``.
The GAP profile, the durability profile and the race plan all read that stored
fit — the three tools always describe the same runner — and a new run does not
change it: a refit happens when the athlete asks for one (the tools' Recompute
button), and otherwise only

* when there is no fit yet (or it was written by an older recipe), or
* when the stored fit had nothing personal in it (too few runs) and runs have
  come in since — there is nothing on screen to keep stable then.
"""

import logging
from datetime import date
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from api.deps import data_source_for, get_athlete_model_repository, get_caches
from src.domain.durability.config import DEFAULT_CONFIG as DURABILITY_CONFIG
from src.domain.durability.history import durability_activity_ids, fit_athlete_durability
from src.domain.durability.personalization import POPULATION_ONLY, AthleteDurabilityModel
from src.domain.models.gap import GapCurve
from src.domain.ports.storage import Athlete
from src.usecases.plan_race import fit_personal_curve, running_activity_ids

logger = logging.getLogger(__name__)

# Bump when a fitting recipe (or what is stored of it) changes: rows written by
# another version are ignored, so every athlete is refitted once.
GAP_VERSION = "v1"
DURABILITY_VERSION = f"{DURABILITY_CONFIG.population.version}|v1"
DURABILITY = "durability"


def gap_kind(model: str) -> str:
    return f"gap_curve:{model}"


class AthleteModels:
    """One athlete's stored models for one request; ``refit`` fits each anew, once."""

    def __init__(self, athlete: Athlete, refit: bool = False):
        self.refit = refit
        self.data = data_source_for(athlete)
        self.memo = get_caches(athlete.id).memo
        self.store = get_athlete_model_repository(athlete.id)
        # Fitted during this request: a second read must not refit again.
        self._fresh: Dict[str, Dict[str, Any]] = {}
        self._status: Dict[str, Dict[str, Any]] = {}

    def gap_curve(self, model: str) -> Tuple[Optional[GapCurve], Optional[str]]:
        """``(curve, reason)`` — a ``PersonalCurve`` for :class:`PlanRace`."""
        kind = gap_kind(model)
        ids = running_activity_ids(self.data)
        payload = self._load(kind, GAP_VERSION, ids, lambda p: p.get("curve") is not None)
        if payload is None:
            curve, reason = fit_personal_curve(self.data, ids, model, self.memo)
            payload = {"curve": _curve_payload(curve) if curve is not None else None,
                       "reason": reason}
            self._save(kind, GAP_VERSION, payload, ids)
        stored = payload.get("curve")
        return (_curve(stored) if stored else None), payload.get("reason")

    def durability(self) -> AthleteDurabilityModel:
        """The durability model — a ``DurabilityProvider`` for :class:`PlanRace`."""
        today = date.today()
        ids = durability_activity_ids(self.data.summaries(), today, DURABILITY_CONFIG)
        payload = self._load(DURABILITY, DURABILITY_VERSION, ids,
                             lambda p: p.get("confidence") != POPULATION_ONLY)
        if payload is not None:
            return AthleteDurabilityModel.from_store(payload)
        model = fit_athlete_durability(self.data, today, DURABILITY_CONFIG, memo=self.memo)
        self._save(DURABILITY, DURABILITY_VERSION, model.to_store(), ids)
        return model

    def status(self, kind: str) -> Dict[str, Any]:
        """When ``kind`` was fitted and how many runs are newer — after reading it."""
        return self._status.get(kind, {"computed_at": None, "new_runs": 0})

    # --- Storage -------------------------------------------------------------

    def _load(self, kind: str, version: str, ids: Sequence[int],
              personal: Callable[[Dict[str, Any]], bool]) -> Optional[Dict[str, Any]]:
        """The stored payload to use, or ``None`` when it must be fitted."""
        if kind in self._fresh:
            return self._fresh[kind]
        if self.refit:
            return None
        try:
            row = self.store.get(kind, version)
        except Exception as error:
            logger.warning("could not read stored %s model: %s", kind, error)
            return None
        if row is None:
            return None
        new_runs = len(set(ids) - set(row["activity_ids"]))
        if new_runs and not personal(row["payload"]):
            return None
        self._status[kind] = {"computed_at": _iso(row["computed_at"]), "new_runs": new_runs}
        return row["payload"]

    def _save(self, kind: str, version: str, payload: Dict[str, Any],
              ids: Sequence[int]) -> None:
        self._fresh[kind] = payload
        computed_at = None
        try:
            computed_at = self.store.put(kind, version, payload, ids)
        except Exception as error:
            logger.warning("could not store %s model: %s", kind, error)
        self._status[kind] = {"computed_at": _iso(computed_at), "new_runs": 0}


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if value is not None else None


def _curve_payload(curve: GapCurve) -> Dict[str, Any]:
    return {
        "bin_centers": curve.bin_centers.tolist(),
        "means": curve.means.tolist(),
        "stds": curve.stds.tolist(),
        "counts": curve.counts.tolist(),
    }


def _curve(raw: Dict[str, Any]) -> GapCurve:
    # ``float`` dtype turns the ``null``s NaNs were stored as back into NaN.
    return GapCurve(
        bin_centers=np.asarray(raw["bin_centers"], dtype=float),
        means=np.asarray(raw["means"], dtype=float),
        stds=np.asarray(raw["stds"], dtype=float),
        counts=np.asarray([c or 0 for c in raw["counts"]], dtype=int),
    )
