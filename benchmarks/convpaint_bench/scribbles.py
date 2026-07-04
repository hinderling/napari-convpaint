"""Thin wrapper around the Convpaint paper's scribble generator.

The scribble-generation algorithm (skeleton + line drawing from a ground-truth
mask, covering a target fraction of pixels) is the one used in the Convpaint
paper: ``quasar1357/scribbles_creator`` by Roman Schwob (a co-author).

It is **GPL-v3** and therefore is *not* vendored into this BSD-3 repository. It is
an optional, benchmark-only dependency installed separately:

    pip install "git+https://github.com/quasar1357/scribbles_creator.git"

Its base install pulls only numpy/scikit-image/scipy (all already Convpaint deps);
only the unused ``scribbles_testing`` submodule has heavier requirements.
"""
from __future__ import annotations

import numpy as np

_IMPORT_ERROR_MSG = (
    "scribbles_creator is required for scribble generation but is not installed.\n"
    "Install it (benchmark-only, GPL-v3, kept out of the repo) with:\n"
    '    pip install "git+https://github.com/quasar1357/scribbles_creator.git"'
)


def _patch_scribbles_creator_bug(module):
    """Work around an upstream bug in scribbles_creator's ``square`` helper.

    The published version defines ``def square(side): return rectangle(n, n)`` —
    the parameter is ``side`` but the body references an undefined ``n``, so any
    scribble mode that dilates/closes with a square structuring element crashes
    with ``NameError: name 'n' is not defined``. We replace it with the correct
    implementation at import time. (Reported upstream; remove once fixed.)
    """
    square = getattr(module, "square", None)
    if square is None:
        return
    try:
        square(1)
    except NameError:
        from skimage.morphology import rectangle
        module.square = lambda side: rectangle(side, side)


def make_scribbles(ground_truth: np.ndarray, max_perc: float = 0.2,
                   seed: int = 1, **kwargs) -> np.ndarray:
    """Generate a scribble annotation covering ~``max_perc`` percent of the
    labelled pixels of ``ground_truth``.

    Parameters
    ----------
    ground_truth : uint8 [H, W], 0 = unannotated, classes from 1.
    max_perc : target percentage of labelled pixels to cover with scribbles.
    seed : seed for the (numpy-global) RNG the generator uses, for reproducibility.
    kwargs : forwarded to ``create_even_scribbles`` (mode, class_dist, ...).

    Returns
    -------
    scribbles : uint8 [H, W], same label convention (0 = unannotated).
    """
    try:
        import scribbles_creator
        from scribbles_creator import create_even_scribbles
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(_IMPORT_ERROR_MSG) from exc

    _patch_scribbles_creator_bug(scribbles_creator)

    # The generator draws on the numpy-global RNG; seed it for reproducibility.
    np.random.seed(seed)
    scribbles = create_even_scribbles(ground_truth, max_perc=max_perc, **kwargs)
    return np.asarray(scribbles, dtype=np.uint8)


def scribble_coverage(scribbles: np.ndarray, ground_truth: np.ndarray) -> float:
    """Fraction of labelled GT pixels that the scribbles annotate (in percent)."""
    labelled = np.count_nonzero(scribbles)
    total = np.count_nonzero(ground_truth)
    return 100.0 * labelled / total if total else 0.0
