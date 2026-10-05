"""The durability profile: how well a runner holds up, against the average runner.

Three qualities, one per coefficient the athlete's own runs can move (thermal is
never fitted, so it has no say):

* ``long_efforts`` — the ``duration`` coefficient: drift from time on feet;
* ``hard_efforts`` — ``severe_intensity``: the extra drift above threshold;
* ``descents`` — ``downhill``: the drift that metres of descent leave behind.

Each coefficient multiplies its exposure in log-cost, so for the same effort the
ratio athlete / population *is* the ratio of the drift that quality causes:
``(athlete / population − 1)`` as a percentage, rated on the shared scale. Above
zero they drift more than the average runner, which is worse.

A quality the athlete's runs barely inform — a personal weight (the variance the
data removed from the prior, 0–1) under :data:`MIN_PERSONAL_WEIGHT` — would only
echo the population back. It reads ``insufficient`` instead.
"""

from typing import List, Optional

import numpy as np

from src.domain.assessment import Assessment
from src.domain.charts.ir import Axis, AxisKind, ChartData, Trace, TraceKind
from src.domain.durability.config import DOWNHILL, DURATION, SEVERE_INTENSITY, DurabilityConfig
from src.domain.durability.personalization import POPULATION_ONLY, AthleteDurabilityModel
from src.domain.gap import theme
from src.translations import translate

QUALITIES = (
    ("long_efforts", DURATION),
    ("hard_efforts", SEVERE_INTENSITY),
    ("descents", DOWNHILL),
)

MIN_PERSONAL_WEIGHT = 0.3

# The projection's horizon: a little past the athlete's longest run, within bounds.
_HORIZON_H = (4.0, 12.0)


def assess(model: AthleteDurabilityModel) -> List[Assessment]:
    """One assessment per quality, in :data:`QUALITIES` order."""
    return [Assessment.of(key, _extra(model, name)) for key, name in QUALITIES]


def _extra(model: AthleteDurabilityModel, name: str) -> Optional[float]:
    if model.confidence == POPULATION_ONLY:
        return None
    if model.personal_weight.get(name, 0.0) < MIN_PERSONAL_WEIGHT:
        return None
    population = model.population.get(name)
    if population <= 0:
        return None
    return (model.coefficients.get(name) / population - 1.0) * 100.0


def profile_chart(
    model: AthleteDurabilityModel, config: DurabilityConfig, lang: str
) -> Optional[ChartData]:
    """Projected extra cost over a long steady run: the runner against the average.

    ``None`` without a personal fit — the two curves would be one. No title: it
    lives in the card (charts.md).
    """
    # Imported here: the plot module registers itself on import, and this module
    # is also imported by code that never draws.
    from src.domain.plots.durability_curve import projected_extra_cost

    if model.confidence == POPULATION_ONLY:
        return None
    longest = max((s.elapsed_s for s in model.segments), default=0.0) / 3600.0
    horizon = float(np.clip(np.ceil(longest + 1), *_HORIZON_H))
    hours = np.linspace(0.0, horizon, 97)

    def trace(coefficients, name, color, dash, width):
        return Trace(
            name=name,
            x=hours.round(3).tolist(),
            y=projected_extra_cost(model, config, hours, coefficients).round(2).tolist(),
            kind=TraceKind.LINE,
            color=color,
            dash=dash,
            width=width,
            hover_template="%{x:.1f} h<br>+%{y:.1f} %<extra>%{fullData.name}</extra>",
        )

    return ChartData(
        x_axis=Axis(title=translate("ui.durability_tool.chart.x", lang), kind=AxisKind.LINEAR,
                    tick_format=",.0f", suffix=" h", dtick=1),
        y_axis=Axis(title=translate("ui.durability_tool.chart.y", lang), kind=AxisKind.LINEAR,
                    tick_format=",.0f", suffix=" %"),
        traces=[
            trace(model.population, translate("ui.durability_tool.chart.average", lang),
                  theme.BALANCED_RUNNER, "--", 1.5),
            trace(model.coefficients, translate("ui.gap_tool.chart.you", lang),
                  theme.EFFICIENCY, "-", 2.4),
        ],
    )
