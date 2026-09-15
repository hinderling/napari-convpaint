"""Bounded, FE-pluggable feature cache for the interactive annotate→predict loop.

The expensive step in interactive segmentation is feature extraction; when the
same image (or z-slice / movie frame) is segmented repeatedly while refining
scribbles, its features can be reused instead of recomputed. This module holds a
generic, feature-extractor-agnostic cache: it stores an *opaque payload* defined
by each FE (e.g. DINO patch tokens — tiny and lossless to upsample), keyed by
`(img_id, slice, FE-signature)`, and owns everything storage-related — LRU
eviction and, crucially, a memory budget so caching a 100-slice stack or a
300-frame movie can never grow unbounded and crash the kernel.

Triage principle (never OOM): before storing an entry, its size is checked
against a live budget = `min(configured cap, available_RAM − headroom)`. If it
does not fit, the least-recently-used entries are evicted; if it still does not
fit (a single payload larger than the budget), it is simply not cached and the
caller recomputes. The cache never exceeds the budget, so it degrades to
recomputation rather than pushing the system into swap.

FE-specific behaviour lives in the FeatureExtractor (see the
`cacheable_repr_and_features` / `features_from_cacheable` /
`supports_feature_cache` protocol on the base class): the cache never needs to know what is inside a
payload.
"""
from __future__ import annotations

import functools
import threading
from collections import OrderedDict

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover - psutil is optional
    _HAVE_PSUTIL = False


# Fallback absolute cap when available-RAM cannot be queried (no psutil).
_DEFAULT_MAX_BYTES = 2 * 1024 ** 3  # 2 GiB
# Keep at least this fraction of currently-available RAM free (never consume it
# all with cache), as a safety headroom against OOM.
_DEFAULT_HEADROOM_FRAC = 0.25


def _payload_nbytes(payload) -> int:
    """Best-effort byte size of an opaque payload (array, list/tuple of arrays,
    or anything exposing .nbytes, recursing into lists/tuples/dicts).
    Unknown → 0 (treated as free)."""
    if payload is None:
        return 0
    if hasattr(payload, "nbytes"):
        return int(payload.nbytes)
    if isinstance(payload, (list, tuple)):
        return sum(_payload_nbytes(p) for p in payload)
    if isinstance(payload, dict):
        return sum(_payload_nbytes(v) for v in payload.values())
    return 0


def _locked(method):
    """Run `method` under the cache's re-entrant lock. The cache is shared
    across threads (e.g. the napari GUI thread changing limits or clearing
    while a worker thread is inside get/put), so every public entry point must
    hold the lock; private helpers are only called from within one."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class FeatureCache:
    """LRU feature cache bounded by a memory budget.

    Thread-safe: all public methods take a re-entrant lock, so a worker thread
    can extract/cache while the GUI thread clears the cache or changes limits.

    Parameters
    ----------
    max_bytes : int or None
        Hard cap on the cache's own size. None → an automatic cap derived from
        system RAM (a quarter of total, or `_DEFAULT_MAX_BYTES` without psutil).
    headroom_frac : float
        Fraction of *currently available* RAM to always keep free. The live
        budget is `available_RAM * (1 - headroom_frac)`; entries are never added
        (and are evicted) to respect it, so the cache cannot trigger OOM.
    enabled : bool
        Master switch; when False, get() always misses and put() is a no-op.
    """

    def __init__(self, max_bytes: int | None = None,
                 headroom_frac: float = _DEFAULT_HEADROOM_FRAC,
                 enabled: bool = True):
        self._lock = threading.RLock()
        self._store: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (payload, nbytes)
        self._total_bytes = 0
        self._headroom_frac = float(headroom_frac)
        self.enabled = bool(enabled)
        if max_bytes is None:
            if _HAVE_PSUTIL:
                max_bytes = int(psutil.virtual_memory().total * 0.25)
            else:
                max_bytes = _DEFAULT_MAX_BYTES
        self._max_bytes = int(max_bytes)
        self.hits = 0
        self.misses = 0
    # -- budget helpers ----------------------------------------------------

    def _available_bytes(self) -> int:
        if _HAVE_PSUTIL:
            return int(psutil.virtual_memory().available)
        # No psutil: rely solely on the configured cap (assume plenty free).
        return self._max_bytes

    def _fits(self, nbytes: int, available: int | None = None) -> bool:
        """Whether adding `nbytes` keeps the cache under its cap AND leaves the
        configured headroom of currently-available RAM free. Pass `available`
        to reuse one RAM snapshot across repeated checks (e.g. put's eviction
        loop) instead of re-querying psutil per call."""
        if self._total_bytes + nbytes > self._max_bytes:
            return False
        if available is None:
            available = self._available_bytes()
        # available RAM already accounts for the cache's current allocation, so
        # only the *new* bytes reduce it further.
        return available - nbytes >= self._headroom_frac * available

    # -- public API --------------------------------------------------------

    @_locked
    def get(self, key):
        """Return the cached payload for `key`, or None."""
        if not self.enabled:
            return None
        item = self._store.get(key)
        if item is not None:
            self._store.move_to_end(key)  # most-recently-used
            self.hits += 1
            return item[0]
        self.misses += 1
        return None

    @_locked
    def put(self, key, payload, nbytes: int | None = None):
        """Store `payload` under `key` if it fits the budget; else evict LRU and
        retry. A payload that can never fit is not cached (the caller recomputes)."""
        if not self.enabled or payload is None:
            return
        if nbytes is None:
            nbytes = _payload_nbytes(payload)
        # Overwrite of an existing key: drop the old size first.
        if key in self._store:
            self._total_bytes -= self._store.pop(key)[1]
        # A single payload larger than the whole cap can never be held.
        if nbytes > self._max_bytes:
            return
        # Evict least-recently-used until the new entry fits. One RAM snapshot
        # serves the whole loop: eviction only increases availability, so the
        # snapshot errs conservative.
        available = self._available_bytes()
        while self._store and not self._fits(nbytes, available):
            self._evict_one()
        if not self._fits(nbytes, available):
            # The live headroom refuses it even with the cache empty.
            return
        # The key is absent at this point (popped above if present), so
        # assignment appends at the MRU end.
        self._store[key] = (payload, nbytes)
        self._total_bytes += nbytes

    def _evict_one(self):
        key, (payload, nbytes) = self._store.popitem(last=False)  # LRU = oldest
        self._total_bytes -= nbytes

    # -- invalidation / limits --------------------------------------------

    @_locked
    def clear(self):
        """Drop all entries. Entries are content-addressed and never go stale;
        clearing only frees memory."""
        self._store.clear()
        self._total_bytes = 0

    @_locked
    def set_max_bytes(self, max_bytes: int):
        """Change the cap in place, evicting LRU entries if over."""
        self._max_bytes = int(max_bytes)
        while self._store and self._total_bytes > self._max_bytes:
            self._evict_one()

    @_locked
    def set_enabled(self, enabled: bool):
        """Enable/disable in place; disabling clears the cache to free space."""
        self.enabled = bool(enabled)
        if not self.enabled:
            self.clear()

    @property
    def nbytes(self) -> int:
        return self._total_bytes

    @_locked
    def __len__(self):
        return len(self._store)

    @_locked
    def stats(self) -> dict:
        return {
            "entries": len(self._store),
            "bytes": self._total_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "max_bytes": self._max_bytes,
        }
