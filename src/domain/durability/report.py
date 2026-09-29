"""Durability diagnostics as chart IR pieces, shared by the race plan and the analysis plot.

Every durability output carries the same two things: the coefficients (population
prior vs what was applied, and how much of it is personal), and plain-language
notes on confidence, fallback reasons and the status of the defaults. They are
essential for trusting — and later calibrating — the model, so both surfaces show
them identically.
"""

from src.domain.charts.ir import CellFormat, Column, TableData
from src.domain.durability.config import (
    COMPONENTS,
    DOWNHILL,
    DURATION,
    PLACEHOLDER,
    PRE_RACE_LOAD,
    SEVERE_INTENSITY,
    THERMAL,
)
from src.domain.durability.personalization import AthleteDurabilityModel
from src.translations import translate



_UNITS = {DURATION: "1/h", SEVERE_INTENSITY: "1/h", DOWNHILL: "1/km", THERMAL: "1/h",
          PRE_RACE_LOAD: "—"}


def durability_table(model: AthleteDurabilityModel, profile, lang: str) -> TableData:
    """Coefficients (population vs applied), exposure and contribution at the finish."""
    rows = []
    for name in COMPONENTS + (PRE_RACE_LOAD,):
        if name == PRE_RACE_LOAD and PRE_RACE_LOAD not in profile.components:
            continue
        exposure = (profile.pre_race_exposure if name == PRE_RACE_LOAD
                    else float(profile.exposures[name][-1]))
        rows.append({
            "component": translate(f"durability.component.{name}", lang),
            "unit": _UNITS[name],
            "population": model.population.get(name),
            "applied": model.coefficients.get(name),
            "weight": model.personal_weight.get(name, 0.0) * 100,
            "exposure": exposure,
            "contribution": float(profile.components[name][-1]) * 100,
        })
    return TableData(
        title=translate("durability.table.title", lang),
        columns=[
            Column("component", translate("durability.col.component", lang)),
            Column("unit", translate("durability.col.unit", lang)),
            Column("population", translate("durability.col.population", lang),
                   CellFormat("number", 4)),
            Column("applied", translate("durability.col.applied", lang), CellFormat("number", 4)),
            Column("weight", translate("durability.col.weight", lang), CellFormat("number", 0, "%")),
            Column("exposure", translate("durability.col.exposure", lang), CellFormat("number", 2)),
            Column("contribution", translate("durability.col.contribution", lang),
                   CellFormat("number", 2, "%")),
        ],
        rows=rows,
        download_name="durability_model",
    )


def durability_notes(model: AthleteDurabilityModel, lang: str, solution=None) -> list:
    notes = [translate(f"durability.confidence.{model.confidence}", lang).format(
        runs=model.n_activities, hours=f"{model.hours:.0f}",
    )]
    for reason in model.reasons:
        notes.append(translate(f"durability.reason.{reason}", lang))
    if model.coefficients.status == PLACEHOLDER:
        notes.append(translate("durability.note.placeholder", lang).format(
            version=model.population.version))
    if solution is not None:
        if solution.reference.source == "target_time":
            notes.append(translate("durability.note.target_reference", lang))
        if solution.profile.clamped:
            notes.append(translate("durability.note.clamped", lang))
        if solution.fallback == "not_converged":
            notes.append(translate("durability.note.not_converged", lang))
        elif solution.fallback == "fresh":
            notes.append(translate("durability.note.fresh_fallback", lang))
    return notes
