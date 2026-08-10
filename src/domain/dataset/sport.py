"""Which broad sport family an activity belongs to.

Two families exist: **running** (the app's original domain — "Run", "TrailRun",
"VirtualRun") and **cycling** (every one of Strava's bike ``sport_type`` values —
a gravel or mountain ride is still "usual cycling" to every panel here, since
none of them distinguish bike sub-types). A panel's data has to stay within one
family — GAP and the modelled power-per-kg are running biomechanics, calibrated
to a "balanced runner" reference curve that means nothing for a bike, and even a
mechanically sport-agnostic number like "fastest 10 km" is not a comparable
figure between a foot split and a bike split. See
:mod:`src.domain.dataset.features` for where running-only columns are guarded,
and :mod:`src.usecases.resolve_panel_data` for where a mixed selection is
resolved down to one family.
"""

RUNNING_SPORT_TYPES = frozenset({"Run", "TrailRun", "VirtualRun"})
CYCLING_SPORT_TYPES = frozenset({
    "Ride", "MountainBikeRide", "GravelRide", "VirtualRide"
})

RUNNING = "running"
CYCLING = "cycling"


def sport_family(sport_type: str) -> str:
    """``"cycling"`` for a recognised ride, ``"running"`` for everything else.

    Running is the fallback rather than "unknown" because it is what every
    activity in this app was before cycling existed — an oddly-typed or
    summary-only row should keep behaving exactly as it did.
    """
    return CYCLING if sport_type in CYCLING_SPORT_TYPES else RUNNING
