"""Give every athlete the analyses the product ships with.

The three built-in analyses used to be generated on every request and served
read-only. That made them examples rather than tools: the race comparator is defined
by a hand-picked set of workouts, and a page nobody can edit can never be given one.

So they are seeded as **real stored pages** instead — same table, same editor, same
save path as anything the athlete builds. They differ in exactly one way: they carry a
``builtin_key`` and cannot be deleted.

Seeding is idempotent and never overwrites. Once an athlete has their GAP analysis,
that row is *theirs*; a later seed must not reset the windows they chose or the plots
they added. The one consequence worth knowing is that a default is built from the date
range known **at seed time** — see :func:`ensure_default_analyses` for why that is the
right trade now that the page is editable.
"""

import logging
from dataclasses import dataclass
from datetime import date
from typing import List

from src import dashboards
from src.domain.ports.page_repository import PageRepository
from src.domain.spec.pages import PageSpec
from src.usecases.base import UseCase

logger = logging.getLogger(__name__)


@dataclass
class EnsureDefaultAnalysesInput:
    pages: PageRepository
    # The athlete's data range, used to build windows that actually contain runs.
    oldest: date
    newest: date
    lang: str = "en"


class EnsureDefaultAnalyses(UseCase):
    """Create any default analysis this athlete is missing."""

    def execute(self, params: EnsureDefaultAnalysesInput) -> List[PageSpec]:
        """Seed the missing defaults and return what was created (often nothing).

        Built from ``[oldest, newest]`` at the moment of seeding. That range is frozen
        into the page — the GAP analysis gets one window per year *of the history that
        existed then* — which is correct for a page the athlete now owns: extending it
        later would silently rewrite a document they may have edited. Importing new
        years means editing the windows, which is one click in the data-source editor,
        or duplicating the analysis.
        """
        existing = params.pages.default_keys()
        created: List[PageSpec] = []

        for key, builder in dashboards.BUILDERS.items():
            if key in existing:
                continue
            page = builder(params.oldest, params.newest, params.lang)
            try:
                params.pages.save(page)
            except Exception as error:
                # The unique index is the real guard: a concurrent first page-load can
                # lose this race, and losing it means the page already exists, which is
                # the outcome we wanted.
                logger.info("default analysis %s already seeded: %s", key, error)
                continue
            created.append(page)

        return created
