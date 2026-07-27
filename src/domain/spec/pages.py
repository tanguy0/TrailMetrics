"""Page / panel / plot specs — a page *is* data.

    PageSpec ──< PanelSpec ──< PlotSpec
                    └── DataSourceSpec

This is the load-bearing idea of the app: a page is a serializable document, not
code. Once that holds, "the user builds a page by hand", "the app ships example
pages" and "pages live in a database" are the same mechanism — a built-in page is
just a :class:`PageSpec` constructed in :mod:`src.dashboards` instead of loaded
from a repository, and every page renders through one path.

``schema_version`` is stamped on save so a future migration can upgrade stored
documents; plot parameters survive schema drift on their own via
:func:`src.domain.spec.params.coerce`.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from src.domain.spec.datasource import DataSourceSpec

SCHEMA_VERSION = 1


def new_id(prefix: str) -> str:
    """Short stable id — keys widgets, survives reordering, safe in a URL."""
    return f"{prefix}_{uuid4().hex[:8]}"


@dataclass
class PlotSpec:
    """One plot inside a panel: a registry key plus the parameters chosen for it."""

    plot_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    # Overrides the plot type's default heading when set.
    title: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("plot"))
    collapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "plot_type": self.plot_type,
            "params": dict(self.params),
            "title": self.title,
            "collapsed": self.collapsed,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PlotSpec":
        return PlotSpec(
            plot_type=str(raw["plot_type"]),
            params=dict(raw.get("params") or {}),
            title=raw.get("title"),
            id=str(raw.get("id") or new_id("plot")),
            collapsed=bool(raw.get("collapsed", False)),
        )


@dataclass
class PanelSpec:
    """One data source and every plot built on it."""

    title: str = ""
    source: DataSourceSpec = field(default_factory=DataSourceSpec)
    plots: List[PlotSpec] = field(default_factory=list)
    # 1 or 2 — how many plots sit side by side.
    columns: int = 1
    description: str = ""
    id: str = field(default_factory=lambda: new_id("panel"))
    collapsed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "source": self.source.to_dict(),
            "plots": [p.to_dict() for p in self.plots],
            "columns": self.columns,
            "collapsed": self.collapsed,
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PanelSpec":
        return PanelSpec(
            title=str(raw.get("title") or ""),
            description=str(raw.get("description") or ""),
            source=DataSourceSpec.from_dict(raw.get("source") or {}),
            plots=[PlotSpec.from_dict(p) for p in (raw.get("plots") or [])],
            columns=int(raw.get("columns") or 1),
            id=str(raw.get("id") or new_id("panel")),
            collapsed=bool(raw.get("collapsed", False)),
        )


@dataclass
class PageSpec:
    """A whole page: an ordered list of panels.

    ``builtin_key`` is set on the example pages that ship with the app (the GAP
    simulator, race comparator and long-term progress). They render through the
    same path as user pages but are not editable in place — the user duplicates
    one to get an editable copy, which is what makes them useful as examples.
    """

    name: str
    description: str = ""
    panels: List[PanelSpec] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("page"))
    builtin_key: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    icon: str = "📊"

    @property
    def is_builtin(self) -> bool:
        return self.builtin_key is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "builtin_key": self.builtin_key,
            "panels": [p.to_dict() for p in self.panels],
        }

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PageSpec":
        return PageSpec(
            name=str(raw.get("name") or "Untitled"),
            description=str(raw.get("description") or ""),
            panels=[PanelSpec.from_dict(p) for p in (raw.get("panels") or [])],
            id=str(raw.get("id") or new_id("page")),
            builtin_key=raw.get("builtin_key"),
            schema_version=int(raw.get("schema_version") or SCHEMA_VERSION),
            icon=str(raw.get("icon") or "📊"),
        )

    def copy_as_custom(self, name: str) -> "PageSpec":
        """An editable clone with fresh ids and no built-in link.

        This is the "duplicate this example" action: the user gets a real page of
        their own, pre-filled with a working analysis they can pull apart.
        """
        clone = PageSpec.from_dict(self.to_dict())
        clone.name = name
        clone.builtin_key = None
        clone.id = new_id("page")
        for panel in clone.panels:
            panel.id = new_id("panel")
            for plot in panel.plots:
                plot.id = new_id("plot")
        return clone
