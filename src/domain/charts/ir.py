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

from dataclasses import asdict, dataclass, field, fields
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
    # Tint the axis title and ticks to match the series measured against it. Set on
    # dual-axis charts, where "which axis is this line on?" is otherwise a guess.
    color: Optional[str] = None


@dataclass
class Trace:
    """One series. ``x``/``y`` may contain ``None`` to break the line into runs."""

    name: str
    x: List[Any] = field(default_factory=list)
    y: List[Any] = field(default_factory=list)
    kind: TraceKind = TraceKind.LINE
    color: Optional[str] = None
    # Which y-axis this series is measured against: "y" (left) or "y2" (right).
    # Only meaningful when the chart defines a ``y2_axis``.
    axis: str = "y"
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
    """One figure: axes plus the traces drawn on them.

    ``y2_axis`` opts the figure into a **second, right-hand y-axis**, for the case
    where one chart has to carry two quantities in different units — distance and
    climb per week, heart rate against pace within a run. Traces then choose their
    axis via :attr:`Trace.axis`.

    Use it only when the comparison is the point. Two scales mean the reader cannot
    trust where the series cross — that crossing is an artefact of how each axis
    happens to be scaled, not a fact about the data. Where the quantities share a
    unit, put them on one axis; where the shapes matter more than the relationship,
    two charts are honest and a second axis is not.
    """

    title: str = ""
    x_axis: Axis = field(default_factory=Axis)
    y_axis: Axis = field(default_factory=Axis)
    # Present only for dual-unit charts; ``None`` leaves the figure single-axis.
    y2_axis: Optional[Axis] = None
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
class TextBlock:
    """Prose inside a panel, so a page can carry its own commentary.

    A page is a document, and a document that can only hold figures forces its
    reasoning into chart titles. This is the smallest thing that fixes that: text
    the author typed, rendered as-is.

    ``text`` is **not** translated — unlike every other string in the IR it comes
    from the athlete, not from :mod:`src.translations`, so it is passed through
    verbatim in whatever language they wrote it.
    """

    text: str = ""
    # "body" | "lede" | "heading" | "quote" — presentation intent, resolved by the
    # renderer, so the author picks meaning rather than a font size.
    variant: str = "body"
    align: str = "left"   # left | center
    tone: str = "none"    # none | forest | terracotta | sunrise | plum

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass
class ImageBlock:
    """One image inside a panel — a course profile, a photo, a screenshot.

    ``src`` is a URL the browser can load: either an external address or an
    app-relative ``/assets/{id}`` path for a file the athlete uploaded. Keeping it
    a plain URL rather than embedded bytes is what stops a page document from
    growing to megabytes.
    """

    src: str = ""
    alt: str = ""
    caption: Optional[str] = None
    # Share of the panel's width, 10–100. A photo rarely wants the full column.
    width_pct: int = 100
    align: str = "left"  # left | center

    @property
    def is_empty(self) -> bool:
        return not self.src.strip()


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
    """Everything one plot produced: figures, tables, prose, images, caveats.

    ``notes`` carries already-translated messages — "power needs your weight",
    "not enough samples in this HR band" — which the app renders as info boxes so
    a partial result explains itself instead of silently showing nothing.

    ``texts`` and ``images`` are what let a page hold its own commentary. They ride
    in the same envelope as charts on purpose: a plot type that produces prose is
    then an ordinary registry entry, edited by the same generated form and stored
    in the same document, rather than a second kind of panel content.
    """

    charts: List[ChartData] = field(default_factory=list)
    tables: List[TableData] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    texts: List[TextBlock] = field(default_factory=list)
    images: List[ImageBlock] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return (
            all(c.is_empty for c in self.charts)
            and all(t.is_empty for t in self.tables)
            and all(t.is_empty for t in self.texts)
            and all(i.is_empty for i in self.images)
        )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready payload (enums → values, datetimes → ISO strings)."""
        return _jsonable(asdict(self))

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PlotOutput":
        """Rebuild an output from :meth:`to_dict` — the inverse, and load-bearing.

        Without it, a computed output can only ever be cached in the process that
        produced it. With it, an expensive fit survives a restart in Postgres and a
        page opens with the curve already drawn (see
        ``src.infrastructure.postgres.plot_output_repository``).

        Lossy in exactly one way, deliberately: a ``datetime`` x-value comes back as
        the ISO string it was serialized to. Both renderers treat a date axis's
        values as opaque, so nothing downstream can tell.
        """
        raw = raw or {}
        return PlotOutput(
            charts=[_chart_from_dict(c) for c in (raw.get("charts") or [])],
            tables=[_table_from_dict(t) for t in (raw.get("tables") or [])],
            notes=[str(n) for n in (raw.get("notes") or [])],
            texts=[_dataclass_from_dict(TextBlock, t) for t in (raw.get("texts") or [])],
            images=[
                _dataclass_from_dict(ImageBlock, i) for i in (raw.get("images") or [])
            ],
        )


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


def _dataclass_from_dict(cls, raw: Optional[Dict[str, Any]]):
    """Build ``cls`` from a payload, ignoring keys it no longer declares.

    Tolerating unknown keys is what lets a cached output outlive a change to the
    IR: an entry written by an older version loads with its extra fields dropped
    rather than raising, exactly as :func:`src.domain.spec.params.coerce` does for
    stored plot parameters.
    """
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (raw or {}).items() if k in known})


def _axis_from_dict(raw: Optional[Dict[str, Any]]) -> Axis:
    axis = _dataclass_from_dict(Axis, raw)
    axis.kind = AxisKind(axis.kind)
    return axis


def _chart_from_dict(raw: Dict[str, Any]) -> ChartData:
    chart = _dataclass_from_dict(ChartData, raw)
    chart.x_axis = _axis_from_dict(raw.get("x_axis"))
    chart.y_axis = _axis_from_dict(raw.get("y_axis"))
    chart.y2_axis = (
        _axis_from_dict(raw["y2_axis"]) if raw.get("y2_axis") is not None else None
    )
    chart.traces = []
    for payload in raw.get("traces") or []:
        trace = _dataclass_from_dict(Trace, payload)
        trace.kind = TraceKind(trace.kind)
        chart.traces.append(trace)
    return chart


def _table_from_dict(raw: Dict[str, Any]) -> TableData:
    table = _dataclass_from_dict(TableData, raw)
    table.columns = []
    for payload in raw.get("columns") or []:
        column = _dataclass_from_dict(Column, payload)
        column.format = _dataclass_from_dict(CellFormat, payload.get("format"))
        table.columns.append(column)
    table.rows = [dict(row) for row in (raw.get("rows") or [])]
    return table


def empty_output(note: str) -> PlotOutput:
    """A result that produced nothing, carrying the reason why."""
    return PlotOutput(notes=[note])


def as_list(values: Sequence[Any]) -> List[Any]:
    """Coerce numpy arrays / pandas series to a plain list for the IR."""
    if values is None:
        return []
    return [v for v in list(values)]
