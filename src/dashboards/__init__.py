"""The example pages that ship with the app.

Each is a plain :class:`~src.domain.spec.pages.PageSpec` assembled from the same
panels, plots and parameters a user gets in the builder — no privileged code path,
no bespoke rendering. That constraint is the point: if an example needs something
the builder can't express, the builder is missing a feature.

They are built as functions of the loaded date range rather than constants,
because a default time window that doesn't cover the athlete's history would show
an empty page on first open.
"""

from datetime import date
from typing import Callable, Dict, List, Optional

from src.dashboards.gap_simulator import GAP_SIMULATOR_KEY, build_gap_simulator
from src.dashboards.long_term_progress import (
    LONG_TERM_PROGRESS_KEY,
    build_long_term_progress,
)
from src.dashboards.race_comparator import (
    RACE_COMPARATOR_KEY,
    build_race_comparator,
)
from src.domain.spec.pages import PageSpec

PageBuilder = Callable[[date, date, str], PageSpec]

BUILDERS: Dict[str, PageBuilder] = {
    GAP_SIMULATOR_KEY: build_gap_simulator,
    RACE_COMPARATOR_KEY: build_race_comparator,
    LONG_TERM_PROGRESS_KEY: build_long_term_progress,
}


def build(
    key: str, oldest: date, newest: date, lang: str = "en"
) -> Optional[PageSpec]:
    """The built-in page named ``key``, covering ``[oldest, newest]``."""
    builder = BUILDERS.get(key)
    return builder(oldest, newest, lang) if builder else None


def build_all(oldest: date, newest: date, lang: str = "en") -> List[PageSpec]:
    return [builder(oldest, newest, lang) for builder in BUILDERS.values()]


__all__ = [
    "BUILDERS",
    "GAP_SIMULATOR_KEY",
    "LONG_TERM_PROGRESS_KEY",
    "RACE_COMPARATOR_KEY",
    "build",
    "build_all",
]
