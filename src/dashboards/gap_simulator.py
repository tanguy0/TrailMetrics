"""Built-in page: **Personalized GAP Simulator**.

Two panels over the same plot type, showing what its parameters buy you: the first
fits both models **once per calendar year**, so the athlete reads how their
gradient cost has moved across seasons; the second stratifies one model by
heart-rate band over the whole history, exposing how that cost changes with
intensity.

The old version of this page hard-coded a "time scales" editor to compare periods.
Here that is just the data source: one window per year in :func:`_year_windows`,
and every curve splits by window, coloured per window, with no extra code. Which is
the point — "a curve per year" needed a change to *this file only*, not to the plot.
"""

from datetime import date
from typing import List

from src.domain.spec.datasource import (
    ActivityFilter,
    DataSourceSpec,
    SourceMode,
    TimeWindow,
)
from src.domain.spec.pages import PageSpec, PanelSpec, PlotSpec
from src.translations import translate

GAP_SIMULATOR_KEY = "gap_simulator"

# Road runs are included, not excluded.
#
# The intuition says otherwise — this page is about gradient cost, so why keep runs
# with no gradient? Because both models are *calibrated against flat running*. The
# efficiency model normalises every bucket by the median efficiency of the flat band,
# and the auto-learning model can only learn an adjustment where a climbing section
# shares a heart rate with a flat one. Road runs are where those flat samples come
# from, so dropping them starves the very reference the curves are measured against.
#
# Measured on a 1,089-activity history: trail-only fitted the auto-learning model for
# 3 of 5 years, trail + road for 4 of 5, with roughly double the splits per year.
# VirtualRun stays out — a treadmill's altitude is not a gradient.
_DEFAULT_SPORTS = ["TrailRun", "Run"]

# How many recent years the per-year panel fits.
#
# Two independent reasons land on the same number. The group palette
# (`theme.TIME_SCALE_CYCLE`) holds five well-separated hues, so a sixth year would
# repeat a colour and two different years would look like one. And each year is a
# real model fit over per-second data, so the count is also the cost — an athlete
# with twelve years of history would wait for twelve fits to read a figure that no
# longer separates. Older years remain one edit away: duplicate the page and add
# the windows.
MAX_YEARS = 5


def _year_windows(oldest: date, newest: date, limit: int = MAX_YEARS) -> List[TimeWindow]:
    """One window per calendar year covered by ``[oldest, newest]``, newest last.

    Clipped to the athlete's actual range so the first and last windows do not claim
    months with no data behind them, and capped to the most recent ``limit`` years
    (see :data:`MAX_YEARS`).

    A year inside the range with no activities in it is *kept*. It resolves to an
    empty group, and the plot then says "no usable splits for 2021" — which is a
    finding. Dropping it would silently redraw the axis as though that year had
    never been part of the history.
    """
    years = range(oldest.year, newest.year + 1)
    windows = [
        TimeWindow(
            name=str(year),
            start=max(oldest, date(year, 1, 1)),
            end=min(newest, date(year, 12, 31)),
        )
        for year in years
    ]
    return windows[-limit:]


def build_gap_simulator(oldest: date, newest: date, lang: str = "en") -> PageSpec:
    def filters() -> ActivityFilter:
        return ActivityFilter(sport_types=list(_DEFAULT_SPORTS))

    per_year = DataSourceSpec(
        mode=SourceMode.WINDOWS,
        windows=_year_windows(oldest, newest),
        filters=filters(),
    )
    # The intensity panel deliberately stays on the whole history. Crossing years
    # with heart-rate bands multiplies the curves — five years × two bands is ten
    # lines in one colour family — and the question it answers ("does climbing cost
    # me more when I go hard?") is not a question about years.
    all_history = DataSourceSpec(
        mode=SourceMode.WINDOW,
        windows=[TimeWindow(
            name=translate("dash.window.all_history", lang),
            start=oldest, end=newest,
        )],
        filters=filters(),
    )

    return PageSpec(
        name=translate("page.gap.title", lang),
        description=translate("gap.intro", lang),
        icon="⛰️",
        builtin_key=GAP_SIMULATOR_KEY,
        panels=[
            PanelSpec(
                title=translate("dash.gap.panel.per_year", lang),
                description=translate("gap.caption.per_year", lang),
                source=per_year,
                plots=[PlotSpec(plot_type="gap_curve", params={
                    "models": ["efficiency", "auto_learning"],
                    "references": ["balanced_runner", "kilian"],
                    # Off with several years on one figure: a ±1σ ribbon per curve
                    # over five curves hides the curves.
                    "show_std": False,
                    "hr_bands": [],
                })],
            ),
            PanelSpec(
                title=translate("dash.gap.panel.intensity", lang),
                description=translate("gap.caption.intensity", lang),
                source=all_history,
                plots=[PlotSpec(plot_type="gap_curve", params={
                    "models": ["efficiency"],
                    "references": [],
                    "show_std": False,
                    # Named bands replace the two extra figures the page used to
                    # hard-code, and stay editable.
                    "hr_bands": [
                        {"name": translate("gap.intensity.low", lang),
                         "hr_min": 120, "hr_max": 150},
                        {"name": translate("gap.intensity.high", lang),
                         "hr_min": 160, "hr_max": 190},
                    ],
                    "efficiency_band_min_samples": 50,
                })],
            ),
        ],
    )
