"""Plot registry. Importing this package registers every built-in plot type.

Import order is the order plot types appear in the "add plot" picker, so it runs
from the most commonly wanted (a trend over time) to the most specialised (a
fitted GAP model) and ends with the raw table.
"""

from src.domain.plots.base import (  # noqa: F401
    CHEAP,
    EXPENSIVE,
    SERIES_BY_ACTIVITY,
    SERIES_BY_GROUP,
    PlotDefinition,
    all_plots,
    by_level,
    get,
    group_color,
    register,
    series_color,
)

# Each import has the side effect of registering its plot type(s).
from src.domain.plots import metric_trend      # noqa: F401
from src.domain.plots import gradient_map      # noqa: F401
from src.domain.plots import records           # noqa: F401
from src.domain.plots import stream_evolution  # noqa: F401
from src.domain.plots import gap_curve         # noqa: F401
from src.domain.plots import metric_scatter    # noqa: F401
from src.domain.plots import metric_distribution  # noqa: F401
from src.domain.plots import data_table        # noqa: F401
# Content blocks last: they carry no data, so they belong at the end of the picker.
from src.domain.plots import text_block        # noqa: F401
from src.domain.plots import image_block       # noqa: F401

__all__ = [
    "CHEAP",
    "EXPENSIVE",
    "SERIES_BY_ACTIVITY",
    "SERIES_BY_GROUP",
    "PlotDefinition",
    "all_plots",
    "by_level",
    "get",
    "group_color",
    "register",
    "series_color",
]
