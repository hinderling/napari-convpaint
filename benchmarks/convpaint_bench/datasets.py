"""On-the-fly ground-truth datasets for Convpaint benchmarking.

No image data is stored in the repository. Datasets are downloaded on first use
via torchvision (already a Convpaint dependency) into a local cache directory
that is git-ignored (see ``benchmarks/.gitignore``).

Ground truth follows the convention used by the Convpaint paper's benchmark
(``scribbles_creator``): a semantic mask where 0 = unannotated, and class labels
start at 1. For the datasets here we produce a binary foreground/background mask
(background = 1, foreground = 2), matching the paper's semantic-segmentation setup
(instances are merged into a single foreground class).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default cache directory for downloaded datasets (git-ignored).
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Sample:
    """A single benchmark image and its semantic ground truth.

    image : uint8 array, [H, W, 3] (RGB).
    gt    : uint8 array, [H, W], values in {1 (background), 2 (foreground)}.
            0 is reserved for "unannotated" by the scribble convention.
    name  : identifier, used for reporting and caching.
    """
    image: np.ndarray
    gt: np.ndarray
    name: str


def _binarize_pet_trimap(trimap: np.ndarray) -> np.ndarray:
    """Oxford-IIIT Pet trimaps use 1 = pet (foreground), 2 = background,
    3 = boundary/unclassified. Map to the scribble convention: background = 1,
    foreground = 2. The boundary ring (3) is assigned to background so every
    pixel has a definite label (the benchmark scores full-image predictions)."""
    gt = np.full(trimap.shape, 1, dtype=np.uint8)  # default background
    gt[trimap == 1] = 2  # pet -> foreground
    # trimap == 2 (background) and trimap == 3 (boundary) stay background (1)
    return gt


def oxford_pet_samples(num_images: int = 20, split: str = "test",
                       data_dir: Path | None = None, seed: int = 0):
    """Yield ``Sample`` objects from the Oxford-IIIT Pet dataset.

    Binary semantic segmentation (pet vs. background) on natural RGB images,
    which suits the default VGG16 feature extractor (RGB input). Downloads
    ~800 MB on first use into ``data_dir``.

    Parameters
    ----------
    num_images : how many images to draw (deterministically sampled).
    split : "trainval" or "test".
    data_dir : cache directory (defaults to benchmarks/data, git-ignored).
    seed : RNG seed for the image subset selection.
    """
    from torchvision.datasets import OxfordIIITPet

    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    ds = OxfordIIITPet(
        root=str(data_dir),
        split=split,
        target_types="segmentation",
        download=True,
    )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(num_images, len(ds)), replace=False)
    indices.sort()

    for idx in indices:
        img_pil, trimap_pil = ds[int(idx)]
        image = np.asarray(img_pil.convert("RGB"), dtype=np.uint8)
        trimap = np.asarray(trimap_pil, dtype=np.uint8)
        gt = _binarize_pet_trimap(trimap)
        yield Sample(image=image, gt=gt, name=f"oxfordpet_{split}_{int(idx):04d}")


# Registry so the CLI can select a dataset by name; each entry is a generator
# factory taking (num_images, data_dir, seed).
DATASETS = {
    "oxford_pet": oxford_pet_samples,
}


def load_samples(dataset: str = "oxford_pet", num_images: int = 20,
                 data_dir: Path | None = None, seed: int = 0):
    """Load ``num_images`` samples from the named dataset (materialized list)."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Available: {list(DATASETS)}")
    return list(DATASETS[dataset](num_images=num_images, data_dir=data_dir, seed=seed))
