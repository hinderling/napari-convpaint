"""Peak-RSS sampling during a code block.

``tracemalloc`` only accounts for Python-object allocations; the memory that
dominates Convpaint (numpy feature stacks, torch tensors, the classifier) is
native. So we sample the whole process's resident set size (RSS) on a background
thread and report the peak increase over the block.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

import psutil


class _PeakRSSSampler:
    def __init__(self, interval_s: float = 0.02):
        self._proc = psutil.Process()
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.baseline_bytes = self._proc.memory_info().rss
        self.peak_bytes = self.baseline_bytes

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            time.sleep(self._interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        # One final read in case the peak landed between samples.
        rss = self._proc.memory_info().rss
        if rss > self.peak_bytes:
            self.peak_bytes = rss

    @property
    def peak_increase_mb(self) -> float:
        return (self.peak_bytes - self.baseline_bytes) / 1e6

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / 1e6


@contextmanager
def track_peak_rss(interval_s: float = 0.02):
    """Context manager yielding a sampler; read ``.peak_increase_mb`` after."""
    sampler = _PeakRSSSampler(interval_s=interval_s)
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()
