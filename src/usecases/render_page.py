"""Render a page spec: resolve each panel, compute each plot.

The single path from "a page is this document" to "here are the figures". Both the
built-in example pages and anything the user builds go through it, which is what
keeps the examples honest — if this path can't express a page, the page can't ship.

What it deliberately does *not* do is draw anything. It returns chart IR, which the
API hands straight to the browser as JSON — so presentation can change entirely
without touching this file.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.domain.charts.ir import PlotOutput
from src.domain.dataset.features import FEATURE_VERSION
from src.domain.dataset.resolved import ResolvedPanelData
from src.domain.ports.activity_data import ActivityDataSource
from src.domain.plots import base as plot_registry
from src.domain.plots.base import EXPENSIVE, PlotDefinition
from src.domain.spec.pages import PageSpec, PanelSpec, PlotSpec
from src.domain.spec.params import coerce
from src.translations import translate
from src.usecases.base import UseCase
from src.usecases.resolve_panel_data import ResolvePanelData, ResolvePanelDataInput


class OutputCache:
    """Finished plot outputs, keyed by render signature.

    A class rather than a bare dict so the store can be *replaced*: the API backs it
    with Postgres as well as memory, which is what lets an expensive fit survive a
    restart. Kept in the use-case layer because the renderer is the only thing that
    knows when an output is complete — but the interface is deliberately two methods,
    so an implementation needs nothing from this module.

    ``set`` takes the plot type alongside the output. It is not part of the key; it
    is there so a persistent implementation can record *what* it stored, which is
    the difference between a debuggable cache table and an opaque one.
    """

    def __init__(self, store: Optional[Dict[str, PlotOutput]] = None):
        self._store: Dict[str, PlotOutput] = store if store is not None else {}

    def get(self, signature: str) -> Optional[PlotOutput]:
        return self._store.get(signature)

    def set(self, signature: str, plot_type: str, output: PlotOutput) -> None:
        self._store[signature] = output


@dataclass
class RenderContext:
    """Everything a render needs beyond the page itself.

    ``data`` says where activities come from (in-memory streams, or a database);
    the two caches are injected and expected to outlive one render: ``memo`` holds
    streams, per-second series and fitted models, ``output_cache`` holds finished
    plot outputs keyed by their inputs. Together they are why changing one
    parameter re-renders instantly instead of re-crunching a decade of runs.
    """

    data: Optional[ActivityDataSource] = None
    lang: str = "en"
    mass_kg: Optional[float] = None
    memo: Dict[Any, Any] = field(default_factory=dict)
    output_cache: OutputCache = field(default_factory=OutputCache)
    # When set, an expensive plot with no cached result is reported as *pending*
    # instead of computed, so the editor can offer an explicit refresh.
    defer_expensive: bool = False
    # Plot ids the user explicitly asked to compute this run.
    force_compute: set = field(default_factory=set)
    # Recompute everything and overwrite what was cached. This is the "recompute"
    # action: a cache keyed on its inputs is right almost always, and the exception
    # is a fit the athlete wants to run again anyway (new data at the edge of a
    # window, a changed body weight, a suspect curve). Cheaper than any scheme for
    # guessing when that is true.
    refresh: bool = False


@dataclass
class PlotResult:
    spec: PlotSpec
    definition: Optional[PlotDefinition]
    output: PlotOutput = field(default_factory=PlotOutput)
    # Human-readable failure, shown in place of the plot rather than crashing.
    error: Optional[str] = None
    # Expensive and not computed yet — the editor shows a refresh button.
    pending: bool = False
    # Parameters after defaults were filled in; the form edits these.
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return self.spec.title or ""


@dataclass
class PanelResult:
    spec: PanelSpec
    resolved: Optional[ResolvedPanelData] = None
    plots: List[PlotResult] = field(default_factory=list)
    error: Optional[str] = None


class RenderPage(UseCase):
    """Resolve and compute a whole page, one panel at a time."""

    def __init__(self, resolver: Optional[ResolvePanelData] = None):
        self.resolver = resolver or ResolvePanelData()

    def execute(self, page: PageSpec, context: RenderContext) -> List[PanelResult]:
        return [self.render_panel(panel, context) for panel in page.panels]

    def render_panel(self, panel: PanelSpec, context: RenderContext) -> PanelResult:
        definitions = {
            plot.id: plot_registry.get(plot.plot_type) for plot in panel.plots
        }
        needs_streams = any(
            d.requires_streams for d in definitions.values() if d is not None
        )

        if context.data is None:
            return PanelResult(spec=panel, error="no activity data source configured")

        try:
            resolved = self.resolver.execute(ResolvePanelDataInput(
                source=panel.source,
                data=context.data,
                lang=context.lang,
                mass_kg=context.mass_kg,
                memo=context.memo,
                require_streams=needs_streams,
                selection_fallback_label=(
                    panel.title.strip() or translate("panel.selection", context.lang)
                ),
            ))
        except Exception as error:  # a bad spec must not take the page down
            return PanelResult(spec=panel, error=str(error))

        results = [
            self.render_plot(panel, plot, definitions.get(plot.id), resolved, context)
            for plot in panel.plots
        ]
        return PanelResult(spec=panel, resolved=resolved, plots=results)

    def render_plot(
        self,
        panel: PanelSpec,
        plot: PlotSpec,
        definition: Optional[PlotDefinition],
        resolved: ResolvedPanelData,
        context: RenderContext,
    ) -> PlotResult:
        if definition is None:
            return PlotResult(
                spec=plot, definition=None,
                error=translate("plot.unknown_type", context.lang).format(
                    type=plot.plot_type),
            )

        params = coerce(definition.params, plot.params)
        result = PlotResult(spec=plot, definition=definition, params=params)

        if resolved.is_empty and definition.requires_data:
            result.output = PlotOutput(
                notes=[translate("panel.no_activities", context.lang)]
            )
            return result
        if definition.requires_weight and context.mass_kg is None:
            result.output = PlotOutput(
                notes=[translate("races.weight_needed", context.lang)]
            )
            return result

        signature = plot_signature(panel, plot, params, resolved, context, definition)
        cached = None if context.refresh else context.output_cache.get(signature)
        if cached is not None:
            result.output = cached
            return result

        deferred = (
            context.defer_expensive
            and definition.cost == EXPENSIVE
            and plot.id not in context.force_compute
            and not context.refresh
        )
        if deferred:
            result.pending = True
            return result

        try:
            output = definition.compute(resolved, params)
        except Exception as error:
            result.error = f"{type(error).__name__}: {error}"
            return result

        context.output_cache.set(signature, plot.plot_type, output)
        result.output = output
        return result


# Bump when a change to any `compute()` function changes a plot's *output* for
# inputs that would otherwise hash the same — a new default line width, a
# recolored trace, a smoothing tweak — so the persisted cache (`plot_outputs`,
# keyed on `plot_signature`) can't go on serving a pre-change render forever
# just because nothing in `payload` below happens to have changed. Bumping
# this invalidates every athlete's cached output once, on the next render
# after deploy; leave it alone for changes that don't touch a plot's output
# (an unrelated bug fix, a docstring, a new plot type).
# v3: the auto-learning GAP model is fitted on one weighted row per split instead
# of one row per (split, matching flat split) pair. The loss is algebraically
# identical, but XGBoost's histogram sketch bins a different row multiset, so a
# curve can shift by ~1-2% — enough that a cached pre-change render must not sit
# alongside a fresh one on the same page.
# v4: line widths are 30% thinner everywhere; the width lives in the cached IR,
# so old rows would draw thick lines next to fresh thin ones.
RENDER_VERSION = 4


def plot_signature(
    panel: PanelSpec,
    plot: PlotSpec,
    params: Dict[str, Any],
    resolved: ResolvedPanelData,
    context: RenderContext,
    definition: Optional[PlotDefinition] = None,
) -> str:
    """Stable cache key: everything that can change a plot's output.

    Includes the resolved activity ids rather than the source spec alone, so a
    freshly loaded history invalidates naturally while re-rendering the same page
    twice does not recompute anything. `RENDER_VERSION` covers a change to how a
    plot is drawn; `FEATURE_VERSION` covers a change to what it is drawn *from*.
    That second one matters because a re-featurize rewrites the numbers under
    activity ids that are otherwise unchanged — without it, a bump would fix
    every stored row and the athlete would still be shown the old chart from
    `plot_outputs`, for as long as the page's activity set stayed the same.

    A rating the athlete types is that same problem in miniature — same ids, new
    values — but far too frequent to answer by dropping the store the way a
    re-featurize does (see ``_run_sync`` in ``api/routers/activities.py``): every
    session rated would cost the next reader an XGBoost fit. So the plots that
    read ratings say so (``PlotDefinition.reads_ratings``) and get a digest of
    them in their key instead, leaving every other plot's cache untouched.
    """
    payload = {
        "render_version": RENDER_VERSION,
        "feature_version": FEATURE_VERSION,
        "plot_type": plot.plot_type,
        "params": params,
        "source": panel.source.to_dict(),
        "activities": sorted(resolved.activity_ids),
        "groups": [(g.label, sorted(g.activity_ids)) for g in resolved.groups],
        "mass_kg": context.mass_kg,
        "lang": context.lang,
    }
    if definition is not None and definition.reads_ratings:
        payload["ratings"] = _ratings_digest(resolved)
    return json.dumps(payload, sort_keys=True, default=str)


def _ratings_digest(resolved: ResolvedPanelData) -> str:
    """A fingerprint of every rating the athlete has entered.

    Hashed rather than listed: the raw form is three values per rated activity,
    which for a season of running is kilobytes inside a key that is itself stored
    and compared.
    """
    rated = sorted(
        (s.activity_id, s.rpe, s.feeling)
        for s in resolved.all_summaries()
        if s.rpe is not None or s.feeling is not None
    )
    return hashlib.sha1(
        json.dumps(rated, default=str).encode()
    ).hexdigest()
