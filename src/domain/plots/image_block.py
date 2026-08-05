"""An image in a panel — a course profile, a photo, a screenshot from elsewhere.

Like :mod:`src.domain.plots.text_block`, it reads no activity data. It stores a
**URL**, never bytes: either an external address, or ``/assets/{id}`` for a file
the athlete uploaded through ``POST /assets``. A page document therefore stays a
few kilobytes however many images it carries, which matters because the whole spec
is re-sent on every autosave.
"""

from typing import Any, Dict, List

from src.domain.charts.ir import ImageBlock, PlotOutput
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.plots.base import PlotDefinition, register
from src.domain.spec.params import (
    Choice,
    ParamSpec,
    choice,
    image,
    integer,
    text,
)

PARAMS: List[ParamSpec] = [
    image("src", "param.image.src", "", help_key="param.image.src.help"),
    text("caption", "param.image.caption", ""),
    # Not decoration: an image with no text alternative is invisible to a screen
    # reader, and the author is the only one who knows what it shows.
    text("alt", "param.image.alt", "", help_key="param.image.alt.help"),
    integer("width_pct", "param.image.width", 100, min=10, max=100, step=5,
            help_key="param.image.width.help"),
    choice("align", "param.image.align", "left", choices=[
        Choice("left", "param.align.left"),
        Choice("center", "param.align.center"),
    ]),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    caption = str(params.get("caption") or "").strip()
    return PlotOutput(images=[ImageBlock(
        src=str(params.get("src") or "").strip(),
        alt=str(params.get("alt") or "").strip(),
        caption=caption or None,
        width_pct=_width(params.get("width_pct")),
        align=str(params.get("align") or "left"),
    )])


def _width(raw: Any) -> int:
    """Clamp rather than reject: a stored 0 or 400 should still render something."""
    try:
        return max(10, min(100, int(raw)))
    except (TypeError, ValueError):
        return 100


register(PlotDefinition(
    key="image_block",
    label_key="plot.image_block.label",
    description_key="plot.image_block.description",
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=PARAMS,
    requires_data=False,
    category_key="plotcat.content",
))
