"""Built-in page: **Personalized GAP Simulator**.

Two panels over the same kind of source, showing what the plot's parameters buy
you: the first fits both models across the whole history, the second stratifies
the same fit by heart-rate band to expose how the athlete's gradient cost changes
with intensity.

The old version of this page hard-coded a "time scales" editor to compare periods.
Here that is just the data source: switch a panel to *several time windows* and
every curve splits by window, coloured per window, with no extra code.
"""

from datetime import date

from src.domain.spec.datasource import (
    ActivityFilter,
    DataSourceSpec,
    SourceMode,
    TimeWindow,
)
from src.domain.spec.pages import PageSpec, PanelSpec, PlotSpec
from src.translations import translate

GAP_SIMULATOR_KEY = "gap_simulator"

# Trail runs only by default: the gradient cost is what this page is about, and
# flat road runs add samples that all land in the same bucket.
_DEFAULT_SPORTS = ["TrailRun"]


def build_gap_simulator(oldest: date, newest: date, lang: str = "en") -> PageSpec:
    def source() -> DataSourceSpec:
        return DataSourceSpec(
            mode=SourceMode.WINDOW,
            windows=[TimeWindow(
                name=translate("dash.window.all_history", lang),
                start=oldest, end=newest,
            )],
            filters=ActivityFilter(sport_types=list(_DEFAULT_SPORTS)),
        )

    return PageSpec(
        name=translate("page.gap.title", lang),
        description=translate("gap.intro", lang),
        icon="⛰️",
        builtin_key=GAP_SIMULATOR_KEY,
        panels=[
            PanelSpec(
                title=translate("dash.gap.panel.curves", lang),
                description=translate("gap.caption.main", lang),
                source=source(),
                plots=[PlotSpec(plot_type="gap_curve", params={
                    "models": ["efficiency", "auto_learning"],
                    "references": ["balanced_runner", "kilian"],
                    "show_std": True,
                    "hr_bands": [],
                })],
            ),
            PanelSpec(
                title=translate("dash.gap.panel.intensity", lang),
                description=translate("gap.caption.intensity", lang),
                source=source(),
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
