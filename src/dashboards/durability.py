"""Built-in page: **Durability**.

How much more does running cost you after two, three, five hours — and is that
more or less than a typical runner? The page explains what is measured, shows the
long runs it is measured on, and ends with the fitted durability curve: the same
model the race plan bends its pacing along.

Everything reads **the past year only**. The window is set to the twelve months
before seeding, and the durability plot additionally ignores anything older than
365 days from the day it is rendered — so even as this stored page ages, it never
fits on stale runs (it will say so when the window no longer covers recent ones).
"""

from datetime import date, timedelta

from src.domain.spec.datasource import (
    ActivityFilter,
    DataSourceSpec,
    SourceMode,
    TimeWindow,
)
from src.domain.spec.pages import PageSpec, PanelSpec, PlotSpec
from src.translations import translate

DURABILITY_KEY = "durability"

LOOKBACK_DAYS = 365
# Outdoor runs only: a treadmill's altitude is not a gradient.
_SPORTS = ["Run", "TrailRun"]
# The same threshold as the model's own "long run" filter.
_MIN_RUN_MINUTES = 45


def build_durability(oldest: date, newest: date, lang: str = "en") -> PageSpec:
    today = date.today()
    window = TimeWindow(
        name=translate("dash.durability.window", lang),
        start=today - timedelta(days=LOOKBACK_DAYS),
        end=today,
    )

    def past_year() -> DataSourceSpec:
        return DataSourceSpec(
            mode=SourceMode.WINDOW,
            windows=[TimeWindow(window.name, window.start, window.end)],
            filters=ActivityFilter(sport_types=list(_SPORTS)),
        )

    return PageSpec(
        name=translate("page.durability.title", lang),
        description=translate("durability.intro", lang),
        icon="🔋",
        builtin_key=DURABILITY_KEY,
        panels=[
            PanelSpec(
                title=translate("dash.durability.panel.method", lang),
                source=past_year(),
                plots=[
                    PlotSpec(plot_type="text_block", params={
                        "text": translate("dash.durability.text.what", lang),
                        "variant": "lede",
                    }),
                    PlotSpec(plot_type="text_block", params={
                        "text": translate("dash.durability.text.how", lang),
                        "variant": "body",
                    }),
                ],
            ),
            PanelSpec(
                title=translate("dash.durability.panel.long_runs", lang),
                description=translate("dash.durability.panel.long_runs.help", lang),
                source=past_year(),
                columns=2,
                plots=[
                    # The longest run of each month: durability is only
                    # measured on long runs, so this is how much evidence there is.
                    PlotSpec(plot_type="metric_trend", params={
                        "metric": "moving_time",
                        "aggregation": "max",
                        "granularity": "month",
                        "x_mode": "calendar",
                        "chart": "bar",
                    }),
                    PlotSpec(plot_type="metric_distribution", params={
                        "metric": "moving_time",
                        "bins": 20,
                    }),
                ],
            ),
            PanelSpec(
                title=translate("dash.durability.panel.curve", lang),
                description=translate("dash.durability.panel.curve.help", lang),
                source=past_year(),
                plots=[PlotSpec(plot_type="durability_curve", params={
                    "lookback_days": LOOKBACK_DAYS,
                    "min_run_minutes": _MIN_RUN_MINUTES,
                    "bin_minutes": 20,
                    "show_observed": True,
                })],
            ),
        ],
    )
