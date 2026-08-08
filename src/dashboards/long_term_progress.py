"""Built-in page: **Long-Term Progress**.

The page that motivated the whole refactor. Its "seasons" were a bespoke editor
plus an overlay/continuous switch plus a grey out-of-season curve — all of which is
now just a data source in *several time windows* mode with ``x_mode="elapsed"``.
Calendar years are the default windows, and the user can redefine them as
"Marathon block" / "Summer base" without the page knowing.

Four panels: records, volume, terrain, efficiency. The volume and terrain panels
share the same year windows so their curves are directly comparable.
"""

from datetime import date
from typing import List

from src.domain.spec.datasource import ActivityFilter, DataSourceSpec, SourceMode, TimeWindow
from src.domain.spec.pages import PageSpec, PanelSpec, PlotSpec
from src.translations import translate

LONG_TERM_PROGRESS_KEY = "long_term_progress"


def _year_windows(oldest: date, newest: date) -> List[TimeWindow]:
    """One window per calendar year covered by the history."""
    return [
        TimeWindow(name=str(year), start=date(year, 1, 1), end=date(year, 12, 31))
        for year in range(oldest.year, newest.year + 1)
    ]


def build_long_term_progress(oldest: date, newest: date, lang: str = "en") -> PageSpec:
    def whole_history() -> DataSourceSpec:
        return DataSourceSpec(
            mode=SourceMode.WINDOW,
            windows=[TimeWindow(
                name=translate("dash.window.all_history", lang),
                start=oldest, end=newest,
            )],
            filters=ActivityFilter(),
        )

    def years() -> DataSourceSpec:
        return DataSourceSpec(
            mode=SourceMode.WINDOWS,
            windows=_year_windows(oldest, newest),
            filters=ActivityFilter(),
        )

    return PageSpec(
        name=translate("page.ltp.title", lang),
        description=translate("ltp.intro", lang),
        icon="📈",
        builtin_key=LONG_TERM_PROGRESS_KEY,
        panels=[
            # 1. Records — all-time, so one window over the whole history. One
            #    chart and one table, not a real side-by-side pair, so this is
            #    columns=1 like Terrain and Efficiency below — a `columns=2`
            #    grid left the chart at half width with nothing to its right,
            #    since the table forces itself onto its own full-width row.
            PanelSpec(
                title=translate("ltp.section.records", lang),
                description=translate("ltp.section.records.help", lang),
                source=whole_history(),
                columns=1,
                plots=[
                    PlotSpec(plot_type="pr_progression", params={
                        "distances": ["5 km", "10 km", "Semi", "Marathon"],
                        "display": "pace",
                        "extend_to_last": True,
                    }),
                    PlotSpec(plot_type="records_table", params={"per_group": False}),
                ],
            ),
            # 2. Volume — year windows aligned to their own start, cumulative.
            PanelSpec(
                title=translate("dash.ltp.panel.volume", lang),
                description=translate("param.x_mode.help", lang),
                source=years(),
                columns=2,
                plots=[
                    PlotSpec(plot_type="metric_trend", params={
                        "metric": "distance_km",
                        "aggregation": "sum",
                        "granularity": "week",
                        "x_mode": "elapsed",
                        "cumulative": True,
                        "chart": "step",
                        "markers": False,
                        "show_totals": True,
                    }),
                    PlotSpec(plot_type="metric_trend", params={
                        "metric": "elevation_gain_m",
                        "aggregation": "sum",
                        "granularity": "week",
                        "x_mode": "elapsed",
                        "cumulative": True,
                        "chart": "step",
                        "markers": False,
                        "show_totals": True,
                    }),
                ],
            ),
            # 3. Terrain — average gradient trend beside the band breakdown.
            PanelSpec(
                title=translate("dash.ltp.panel.terrain", lang),
                description=translate("ltp.gradient_map.help", lang),
                source=years(),
                columns=1,
                plots=[
                    PlotSpec(plot_type="metric_trend", params={
                        "metric": "avg_gradient_pct",
                        "granularity": "month",
                        "x_mode": "elapsed",
                        "chart": "line",
                        "markers": True,
                        "show_totals": True,
                    }),
                    PlotSpec(plot_type="gradient_map", params={
                        "granularity": "month",
                    }),
                ],
            ),
            # 4. Efficiency — one continuous timeline, smoothed; the signal is
            #    noisy week to week and only readable as a trend.
            PanelSpec(
                title=translate("ltp.section.power_hr", lang),
                description=translate("ltp.section.power_hr.help", lang),
                source=whole_history(),
                columns=1,
                plots=[
                    PlotSpec(plot_type="metric_trend", params={
                        "metric": "power_to_hr",
                        "aggregation": "mean",
                        "granularity": "week",
                        "x_mode": "calendar",
                        "chart": "line",
                        "markers": True,
                        "smooth_rolling": 3,
                    }),
                ],
            ),
        ],
    )
