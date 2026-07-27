"""Built-in page: **Race Comparator**.

One panel, because the whole page is about a single hand-picked set of workouts:
the stats table and every evolution trace must describe *the same* selection. This
is exactly the shape the panel model implies — one data source, several plots — and
it removes the old page's problem of a selector bolted onto the top of the file.

The activity list starts empty: the user picks the races, then all five plots
follow.
"""

from datetime import date

from src.domain.spec.datasource import ActivityFilter, DataSourceSpec, SourceMode
from src.domain.spec.pages import PageSpec, PanelSpec, PlotSpec
from src.translations import translate

RACE_COMPARATOR_KEY = "race_comparator"

# The four signals the page originally shipped, as one plot type each.
_SIGNALS = ["gap_pace", "power", "heartrate", "power_to_hr"]


def build_race_comparator(oldest: date, newest: date, lang: str = "en") -> PageSpec:
    plots = [
        PlotSpec(plot_type="data_table", params={
            "rows": "activity",
            "columns": [
                "distance_km", "elevation_gain_m", "moving_time",
                "avg_pace", "avg_gap_pace", "avg_power_w",
            ],
            "highlight_best": True,
            "sort_by": "date",
            "descending": True,
        }),
    ]
    plots += [
        PlotSpec(plot_type="stream_evolution", params={
            "signal": signal,
            "x_axis": "time",
            "max_series": 4,
        })
        for signal in _SIGNALS
    ]

    return PageSpec(
        name=translate("page.races.title", lang),
        description=translate("races.intro", lang).format(max=4),
        icon="🏁",
        builtin_key=RACE_COMPARATOR_KEY,
        panels=[
            PanelSpec(
                title=translate("dash.races.panel.selection", lang),
                description=translate("races.select.subheader", lang),
                # Hand-picked activities — the mode this page exists for.
                source=DataSourceSpec(
                    mode=SourceMode.ACTIVITIES,
                    activity_ids=[],
                    selection_label=translate("dash.races.selection_label", lang),
                    filters=ActivityFilter(),
                ),
                plots=plots,
                columns=2,
            ),
        ],
    )
