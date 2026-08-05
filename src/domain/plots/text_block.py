"""A block of prose in a panel — the page's own commentary.

The one plot type that reads no data at all. It exists because a page is a
document: an analysis that cannot say *what it found* pushes its reasoning into
chart titles, and the reader has to reconstruct the argument from the figures.

Registered as an ordinary plot type rather than a new kind of panel content, which
is what keeps it free: it is edited by the same generated form, stored in the same
document, moved and deleted by the same panel controls.
"""

from typing import Any, Dict, List

from src.domain.charts.ir import PlotOutput, TextBlock
from src.domain.dataset.resolved import DataLevel, ResolvedPanelData
from src.domain.plots.base import PlotDefinition, register
from src.domain.spec.params import Choice, ParamSpec, choice, textarea

PARAMS: List[ParamSpec] = [
    textarea("text", "param.text.body", "", help_key="param.text.body.help"),
    choice("variant", "param.text.variant", "body", choices=[
        Choice("body", "param.text.variant.body"),
        Choice("lede", "param.text.variant.lede"),
        Choice("heading", "param.text.variant.heading"),
        Choice("quote", "param.text.variant.quote"),
    ]),
    choice("align", "param.text.align", "left", choices=[
        Choice("left", "param.align.left"),
        Choice("center", "param.align.center"),
    ]),
    # The accents the rest of the product uses, so a highlighted note looks like it
    # belongs to the same app rather than to this plot type.
    choice("tone", "param.text.tone", "none", choices=[
        Choice("none", "param.tone.none"),
        Choice("forest", "param.tone.forest"),
        Choice("terracotta", "param.tone.terracotta"),
        Choice("sunrise", "param.tone.sunrise"),
        Choice("plum", "param.tone.plum"),
    ]),
]


def compute(resolved: ResolvedPanelData, params: Dict[str, Any]) -> PlotOutput:
    """Pass the author's text through.

    Empty text is *not* an error and gets no note: a block that was just added is
    empty by definition, and telling the author their empty box is empty adds
    nothing. The editor shows the field right above it.
    """
    return PlotOutput(texts=[TextBlock(
        text=str(params.get("text") or ""),
        variant=str(params.get("variant") or "body"),
        align=str(params.get("align") or "left"),
        tone=str(params.get("tone") or "none"),
    )])


register(PlotDefinition(
    key="text_block",
    label_key="plot.text_block.label",
    description_key="plot.text_block.description",
    # Nominally ACTIVITY: the level says which dataset the resolver must build, and
    # `requires_data=False` means none is built. ACTIVITY is the cheapest answer.
    level=DataLevel.ACTIVITY,
    compute=compute,
    params=PARAMS,
    requires_data=False,
    category_key="plotcat.content",
))
