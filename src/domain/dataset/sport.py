"""Which broad sport family an activity belongs to.

Four families exist: **running** (the app's original domain — "Run", "TrailRun",
"VirtualRun"), **cycling** (every one of Strava's bike ``sport_type`` values —
a gravel or mountain ride is still "usual cycling" to every panel here, since
none of them distinguish bike sub-types), **hiking** ("Hike", "Walk"), and
**swimming** ("Swim"). A panel's data has to stay within one family — GAP and
the modelled power-per-kg are running biomechanics, calibrated to a "balanced
runner" reference curve that means nothing for a bike, a hike or a swim, and
even a mechanically sport-agnostic number like "fastest 10 km" is not a
comparable figure between a foot split, a bike split, a hiking split and a
swim split. See :mod:`src.domain.dataset.features` for where running-only
columns are guarded, and :mod:`src.usecases.resolve_panel_data` for where a
mixed selection is resolved down to one family.
"""

RUNNING_SPORT_TYPES = frozenset({"Run", "TrailRun", "VirtualRun"})
CYCLING_SPORT_TYPES = frozenset({
    "Ride", "MountainBikeRide", "GravelRide", "VirtualRide"
})
HIKING_SPORT_TYPES = frozenset({"Hike", "Walk"})
SWIMMING_SPORT_TYPES = frozenset({"Swim"})

RUNNING = "running"
CYCLING = "cycling"
HIKING = "hiking"
SWIMMING = "swimming"


def sport_family(sport_type: str) -> str:
    """``"cycling"``/``"hiking"``/``"swimming"`` for a recognised ride/hike/swim,
    ``"running"`` for everything else.

    Running is the fallback rather than "unknown" because it is what every
    activity in this app was before the other families existed — an
    oddly-typed or summary-only row should keep behaving exactly as it did.
    """
    if sport_type in CYCLING_SPORT_TYPES:
        return CYCLING
    if sport_type in HIKING_SPORT_TYPES:
        return HIKING
    if sport_type in SWIMMING_SPORT_TYPES:
        return SWIMMING
    return RUNNING
