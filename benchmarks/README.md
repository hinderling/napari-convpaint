# Convpaint performance benchmark

A small, self-contained harness to measure Convpaint's **segmentation speed,
memory use, and accuracy**, and to compare candidate performance improvements
(tiling, dask, chunking, data passing) without degrading accuracy.

Nothing here is part of the shipped `napari_convpaint` package — it lives under
`benchmarks/` and is not installed.

## How it works

It reproduces the evaluation approach from the Convpaint paper: take a
fully-annotated **ground-truth** mask, generate **humanlike scribbles** from it,
train Convpaint on the scribbles, predict the whole image, and score the
prediction against the ground truth.

- **Ground truth is downloaded on the fly** — no image data is stored in the
  repo. The default dataset is [Oxford-IIIT Pet][pet], fetched via
  `torchvision` (already a Convpaint dependency) into `benchmarks/data/`
  (git-ignored). Trimaps are binarized to semantic foreground/background
  (background = 1, foreground = 2), matching the paper's semantic setup.
- **Scribbles** are generated with [`scribbles_creator`][sc] — the exact tool
  from the paper (by co-author Roman Schwob). It is **GPL-v3**, so it is *not*
  vendored into this BSD-3 repo; install it separately (see below). Its base
  install only needs numpy/scikit-image/scipy, all already Convpaint deps.

## Setup

```bash
# in the convpaint env
pip install "git+https://github.com/quasar1357/scribbles_creator.git"
```

`torchvision`, `psutil`, `dask`, and `scikit-image` come with Convpaint / its
test extras.

## Usage

```bash
# 10-image baseline with the default VGG16 feature extractor
python benchmarks/run_baseline.py --num-images 10

# compare whole-image vs. tiled vs. dask-tiled prediction on the same images
python benchmarks/run_baseline.py --num-images 10 --compare-tiling --out benchmarks/results/tiling.json

# vary scribble coverage (percent of GT pixels annotated) or FE input downsampling
python benchmarks/run_baseline.py --num-images 10 --scribble-perc 0.5 --downsample 2
```

Each run prints per-image timings and a means summary:

```
config       n_ok  train_s  predict_s  total_s  peak_mem_mb  fg_iou  mean_iou  pixel_acc
whole-image    10    ...        ...       ...        ...       ...      ...        ...
tiled          10    ...        ...       ...        ...       ...      ...        ...
tiled+dask     10    ...        ...       ...        ...       ...      ...        ...
```

## What is measured

| Metric | Meaning |
| --- | --- |
| `train_s` | `ConvpaintModel.train` on the scribbled pixels (FE on annotated pixels + classifier fit) |
| `predict_s` | `ConvpaintModel.segment` on the full image (FE on all pixels + classifier predict) |
| `peak_mem_mb` | peak process RSS **increase** during the cycle, sampled on a background thread (captures native numpy/torch memory, unlike `tracemalloc`) |
| `fg_iou` / `mean_iou` | foreground / mean-over-classes IoU of the prediction vs. GT |
| `pixel_acc` | fraction of pixels labelled correctly |

## Layout

```
benchmarks/
  run_baseline.py            # CLI entry point
  convpaint_bench/
    datasets.py              # on-the-fly GT loaders (torchvision) -> Sample
    scribbles.py             # wrapper around scribbles_creator (optional GPL dep)
    metrics.py               # IoU / Dice / pixel accuracy
    memtrack.py              # background peak-RSS sampler
    runner.py                # one train+predict cycle, timed & measured
  data/                      # downloaded datasets (git-ignored)
  results/                   # JSON outputs (git-ignored)
```

[pet]: https://www.robots.ox.ac.uk/~vgg/data/pets/
[sc]: https://github.com/quasar1357/scribbles_creator
