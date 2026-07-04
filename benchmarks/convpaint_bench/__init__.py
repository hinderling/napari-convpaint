"""Convpaint performance benchmark harness.

Downloads ground-truth segmentation data on the fly (torchvision), generates
humanlike scribbles from it (scribbles_creator, optional GPL dep), runs
Convpaint train+predict, and measures speed / memory / accuracy.

Nothing here is part of the shipped ``napari_convpaint`` package.
"""
from .datasets import Sample, load_samples, DATASETS
from .scribbles import make_scribbles, scribble_coverage
from .metrics import compute_metrics
from .runner import RunConfig, RunResult, run_sample

__all__ = [
    "Sample", "load_samples", "DATASETS",
    "make_scribbles", "scribble_coverage",
    "compute_metrics",
    "RunConfig", "RunResult", "run_sample",
]
