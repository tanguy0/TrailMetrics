"""The process-wide budget behind every athlete's render memo.

The memo is what makes the panel editor usable: a decoded per-second stream, the
smoothed series derived from it and a fitted model are all expensive enough that
recomputing them on each parameter tweak would be unbearable, so they are kept
between requests (see :mod:`api.deps`).

Kept *how long* is the question this module answers. An unbounded memo is fine
until an athlete opens an analysis over their whole history: the GAP page's
second panel selects every activity ever recorded, so every one of their streams
is decoded and pinned — a thousand activities is a few hundred megabytes that is
never handed back. Multiply by the athletes a worker has served and the container
is killed on some later, entirely innocent request.

So entries are held under a **single byte budget for the whole process**, not one
per athlete. That distinction is the point: memory is a property of the container,
and a per-athlete bound multiplied by the number of cached athletes is not a bound
on anything that matters. Eviction is global LRU, so a busy athlete keeps their
warm state and an idle one loses it, which is the trade the caches exist to make.

Evicting is always *safe* — an entry is a memoized pure computation, so a miss
costs time and never correctness. That is what lets the budget be enforced
bluntly, mid-render if need be.
"""

import ctypes
import logging
import os
import sys
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterator, MutableMapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _budget_bytes() -> int:
    """Total bytes the render memo may hold across every athlete.

    Deliberately a fraction of a small container rather than a guess at the
    machine: the default has to be safe on Railway's smallest useful plan, and
    raising it is one environment variable on a plan that can afford it.
    """
    try:
        megabytes = int(os.environ.get("MEMO_BUDGET_MB", "") or 192)
    except ValueError:
        megabytes = 192
    return max(megabytes, 16) * 1024 * 1024


# Deep enough to reach the arrays that actually cost something — a fitted model's
# retained test set is object -> __dict__ -> ndarray — without walking a whole
# object graph on every insert.
_MAX_DEPTH = 6


def estimate_bytes(value: Any, _depth: int = 0, _seen: Optional[set] = None) -> int:
    """Roughly what ``value`` costs to keep, counting the arrays that dominate.

    Approximate on purpose. The entries worth bounding are numpy arrays and
    containers of them, which this counts exactly; everything else is small
    enough that ``sys.getsizeof`` is a fine answer, and being wrong about it only
    shifts an eviction boundary.
    """
    if _seen is None:
        _seen = set()
    marker = id(value)
    if marker in _seen:
        return 0  # already counted, or a cycle
    _seen.add(marker)

    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, (str, bytes, bytearray)):
        return sys.getsizeof(value)
    if _depth >= _MAX_DEPTH:
        return sys.getsizeof(value)

    try:
        if isinstance(value, (list, tuple, set, frozenset)):
            return sys.getsizeof(value) + sum(
                estimate_bytes(item, _depth + 1, _seen) for item in value
            )
        if isinstance(value, dict):
            return sys.getsizeof(value) + sum(
                estimate_bytes(key, _depth + 1, _seen)
                + estimate_bytes(item, _depth + 1, _seen)
                for key, item in value.items()
            )
        contents = getattr(value, "__dict__", None)
        if isinstance(contents, dict):
            return sys.getsizeof(value) + sum(
                estimate_bytes(item, _depth + 1, _seen) for item in contents.values()
            )
        slots = getattr(value, "__slots__", None)
        if slots:
            return sys.getsizeof(value) + sum(
                estimate_bytes(getattr(value, name), _depth + 1, _seen)
                for name in slots
                if hasattr(value, name)
            )
    except Exception:
        # Sizing must never be the thing that fails a render.
        pass
    return sys.getsizeof(value)


class MemoStore:
    """Every athlete's memoized values, under one byte budget, evicted LRU."""

    def __init__(self, budget: Optional[int] = None):
        self.budget = budget if budget is not None else _budget_bytes()
        # (athlete_id, key) -> (value, estimated bytes)
        self._entries: "OrderedDict[Tuple[int, Any], Tuple[Any, int]]" = OrderedDict()
        self._total = 0
        # Handlers are sync and run in FastAPI's threadpool, so several renders
        # touch this at once. The lock guards the store's own bookkeeping; it does
        # not make "check then compute then set" atomic, and does not need to —
        # two threads racing to memoize the same key both compute the same value.
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return self._total

    def get(self, athlete_id: int, key: Any, default: Any = None) -> Any:
        with self._lock:
            entry = self._entries.get((athlete_id, key))
            if entry is None:
                return default
            self._entries.move_to_end((athlete_id, key))
            return entry[0]

    def contains(self, athlete_id: int, key: Any) -> bool:
        with self._lock:
            return (athlete_id, key) in self._entries

    def set(self, athlete_id: int, key: Any, value: Any) -> None:
        size = estimate_bytes(value)
        with self._lock:
            existing = self._entries.pop((athlete_id, key), None)
            if existing is not None:
                self._total -= existing[1]
            # An entry larger than the whole budget is not worth evicting
            # everything else for; it is served to this caller and not kept.
            if size > self.budget:
                logger.info(
                    "memo entry of %.1f MB exceeds the %.1f MB budget; not caching it",
                    size / 1e6, self.budget / 1e6,
                )
                return
            self._entries[(athlete_id, key)] = (value, size)
            self._total += size
            self._evict_to_budget()

    def discard(self, athlete_id: int, key: Any) -> None:
        with self._lock:
            existing = self._entries.pop((athlete_id, key), None)
            if existing is not None:
                self._total -= existing[1]

    def discard_athlete(self, athlete_id: int) -> None:
        """Drop everything memoized for one athlete — what a sync invalidates."""
        with self._lock:
            for entry_key in [k for k in self._entries if k[0] == athlete_id]:
                self._total -= self._entries.pop(entry_key)[1]

    def keys_for(self, athlete_id: int) -> list:
        with self._lock:
            return [key for owner, key in self._entries if owner == athlete_id]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._total = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._total,
                "budget_bytes": self.budget,
            }

    def _evict_to_budget(self) -> None:
        """Drop least-recently-used entries until back under budget. Caller holds the lock."""
        dropped = 0
        while self._total > self.budget and self._entries:
            _, (_, size) = self._entries.popitem(last=False)
            self._total -= size
            dropped += 1
        if dropped:
            logger.info(
                "memo evicted %d entries, now %.1f MB of %.1f MB",
                dropped, self._total / 1e6, self.budget / 1e6,
            )


class AthleteMemo(MutableMapping):
    """One athlete's ``dict``-shaped view onto the shared store.

    The domain takes its memo as a plain mapping and only ever does ``in``, get
    and set on it (see :meth:`ResolvedPanelData.memo`), so the budget can be
    imposed here without the plot layer knowing there is one.
    """

    def __init__(self, store: MemoStore, athlete_id: int):
        self._store = store
        self._athlete_id = athlete_id

    def __contains__(self, key: Any) -> bool:
        return self._store.contains(self._athlete_id, key)

    def __getitem__(self, key: Any) -> Any:
        missing = object()
        value = self._store.get(self._athlete_id, key, missing)
        if value is missing:
            raise KeyError(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        self._store.set(self._athlete_id, key, value)

    def __delitem__(self, key: Any) -> None:
        self._store.discard(self._athlete_id, key)

    def __iter__(self) -> Iterator[Any]:
        # A snapshot: an eviction on another thread must not invalidate iteration.
        return iter(self._store.keys_for(self._athlete_id))

    def __len__(self) -> int:
        return len(self._store.keys_for(self._athlete_id))


def release_heap() -> None:
    """Ask glibc to return free heap pages to the OS.

    Freeing a large allocation inside the process does not necessarily shrink its
    RSS: glibc keeps the pages in its arenas for reuse, and the platform only sees
    the high-water mark. That is how a container gets OOM-killed on a *later*,
    smaller request than the one that actually caused the spike — which made these
    kills look random and unrelated to what triggered them.

    Worth calling after a deliberately large, one-off piece of work (a precompute
    pass) and nowhere else: it walks the arenas, so it is not free, and on a normal
    request the memory is about to be reused anyway.

    A no-op off glibc (macOS, musl), which is fine — this is an optimisation, and
    the platforms without it are not the one being OOM-killed.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass
