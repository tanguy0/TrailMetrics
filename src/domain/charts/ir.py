"""Chart IR — what a plot *computes*, before anything is drawn.

Every plot in :mod:`src.domain.plots` returns a :class:`PlotOutput`: a pure,
JSON-serializable description of traces, axes and tables. It never returns a
Plotly figure. One renderer (:mod:`src.domain.charts.plotly`) turns the IR into
figures. Two renderers consume it — one in Python for notebooks, one in
TypeScript for the web app — so the compute side stays UI-agnostic.

Two consequences worth knowing:

* **Plot definitions describe data, not looks.** Palette, fonts, axis chrome and
  hover styling live once in the renderer, so a new plot type inherits the whole
  Trail / Earthy look for free.
* **Strings in the IR are already translated.** ``compute`` receives the active
  language and puts final text in here, which keeps the renderer dumb and makes
  the payload directly displayable.

Values stay primitive on purpose. Durations are plain **seconds** (the renderer
maps them onto a time axis so ticks read ``m:ss``) and dates are ``datetime`` or
ISO-8601 strings, so :meth:`PlotOutput.to_dict` can hand the whole thing to
``json.dumps``.
"""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


class TraceKind(str, Enum):
    """How one series is drawn."""

    LINE = "line"
    STEP = "step"          # staircase (hold, then jump) — cumulative series
    BAR = "bar"
    SCATTER = "scatter"    # markers only
    AREA = "area"          # filled to zero, or stacked via ``stack_group``


class AxisKind(str, Enum):
    """How axis values should be interpreted."""

    LINEAR = "linear"
    DATE = "date"
    DURATION = "duration"  # seconds; rendered as clock ticks (m:ss / h:mm:ss)
    CATEGORY = "category"


@dataclass
class Axis:
    """One axis: its label, how to read its values, and how to tick it."""

    title: str = ""
    kind: AxisKind = AxisKind.LINEAR
    # Pace-like axes read better upside down (faster = higher).
    reversed: bool = False
    # Plotly d3-format for hover/ticks on numeric axes (e.g. ",.0f"); for DURATION
    # axes this is a strftime pattern (e.g. "%M:%S").
    tick_format: Optional[str] = None
    suffix: Optional[str] = None
    range: Optional[List[float]] = None
    # Force a tick every ``dtick`` units (used by the elapsed-months overlay).
    dtick: Optional[float] = None


@dataclass
class Trace:
    """One series. ``x``/``y`` may contain ``None`` to break the line into runs."""

    name: str
    x: List[Any] = field(default_factory=list)
    y: List[Any] = field(default_factory=list)
    kind: TraceKind = TraceKind.LINE
    color: Optional[str] = None
    # matplotlib-style code ("-", "--", "-.", ":"); mapped to a Plotly dash.
    dash: str = "-"
    width: float = 2.4
    markers: bool = False
    marker_size: float = 5.0
    opacity: float = 1.0
    # Non-empty groups traces into a stack (gradient map, mileage by band).
    stack_group: Optional[str] = None
    # A ±band drawn as a translucent ribbon around ``y`` (GAP ±1σ).
    band_upper: Optional[List[Any]] = None
    band_lower: Optional[List[Any]] = None
    # Pre-formatted per-point strings, referenced from ``hover_template`` as
    # ``%{customdata}`` — how durations/paces get human hovers on a numeric axis.
    hover_text: Optional[List[str]] = None
    hover_template: Optional[str] = None
    legend_group: Optional[str] = None
    show_legend: bool = True


@dataclass
class ChartData:
    """One figure: axes plus the traces drawn on them."""

    title: str = ""
    x_axis: Axis = field(default_factory=Axis)
    y_axis: Axis = field(default_factory=Axis)
    traces: List[Trace] = field(default_factory=list)
    height: int = 460
    # "closest" | "x unified" — the latter suits stacked areas.
    hover_mode: str = "closest"
    # Caption rendered under the figure.
    caption: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not any(t.y for t in self.traces)


@dataclass
class CellFormat:
    """How to render one table column's values."""

    kind: str = "text"  # text | number | integer | duration | pace | date | percent
    decimals: int = 1
    suffix: str = ""


@dataclass
class Column:
    key: str
    label: str
    format: CellFormat = field(default_factory=CellFormat)
    # "max" / "min" highlights the best value in the column; None disables it.
    highlight: Optional[str] = None


@dataclass
class TableData:
    """A table of already-computed values, with optional best-value highlighting."""

    title: str = ""
    columns: List[Column] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    # Basename offered for the CSV download (no extension).
    download_name: str = "table"
    caption: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.rows


@dataclass
class PlotOutput:
    """Everything one plot produced: figures, tables and any caveats to show.

    ``notes`` carries already-translated messages — "power needs your weight",
    "not enough samples in this HR band" — which the app renders as info boxes so
    a partial result explains itself instead of silently showing nothing.
    """

    charts: List[ChartData] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            all(c.is_empty for c in self.charts)
            and all(t.is_empty for t in self.tables)
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready payload (enums → values, datetimes → ISO strings)."""
        return _jsonable(asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and value != value:  # NaN → null
        return None
    return value


def empty_output(note: str) -> PlotOutput:
    """A result that produced nothing, carrying the reason why."""
    return PlotOutput(notes=[note])


def as_list(values: Sequence[Any]) -> List[Any]:
    """Coerce numpy arrays / pandas series to a plain list for the IR."""
    if values is None:
        return []
    return [v for v in list(values)]
