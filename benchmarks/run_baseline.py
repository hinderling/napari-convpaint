#!/usr/bin/env python
"""Run the Convpaint segmentation benchmark and report speed / memory / accuracy.

Downloads ground truth on the fly (torchvision), generates scribbles from it
(scribbles_creator), runs Convpaint train+predict per image, and prints an
aggregate table. Optionally writes per-image results to JSON.

Examples
--------
    # 10-image baseline with the default VGG16 FE
    python benchmarks/run_baseline.py --num-images 10

    # compare whole-image vs. tiled and dask-tiled prediction
    python benchmarks/run_baseline.py --num-images 10 --compare-tiling

Requires (benchmark-only):
    pip install "git+https://github.com/quasar1357/scribbles_creator.git"
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# The catboost/torch libomp double-load aborts the process on macOS unless this
# is set before either is imported; set it here so users don't have to remember.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Make the harness importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from convpaint_bench.datasets import load_samples  # noqa: E402
from convpaint_bench.runner import RunConfig, run_sample  # noqa: E402


def _fmt_table(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = "  ".join(str(c).rjust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(str(r.get(c, "")).rjust(widths[c]) for c in cols) for r in rows)
    return f"{line}\n{sep}\n{body}"


def _summarize(results, label: str) -> dict:
    ok = [r for r in results if r.error is None]
    def mean(attr):
        vals = [getattr(r, attr) for r in ok]
        return statistics.mean(vals) if vals else float("nan")
    def mean_metric(key):
        vals = [r.metrics[key] for r in ok if key in r.metrics]
        return statistics.mean(vals) if vals else float("nan")
    return {
        "config": label,
        "n_ok": len(ok),
        "n_err": len(results) - len(ok),
        "train_s": round(mean("train_time_s"), 3),
        "predict_s": round(mean("predict_time_s"), 3),
        "total_s": round(mean("total_time_s"), 3),
        "peak_mem_mb": round(mean("peak_mem_increase_mb"), 1),
        "fg_iou": round(mean_metric("fg_iou"), 4),
        "mean_iou": round(mean_metric("mean_iou"), 4),
        "pixel_acc": round(mean_metric("pixel_accuracy"), 4),
    }


def _configs(args) -> list[tuple[str, RunConfig]]:
    base = dict(
        fe_name=args.fe,
        channel_mode="rgb" if args.fe in ("vgg16", "dinov2") else "rgb",
        scalings=tuple(args.scalings),
        scribble_perc=args.scribble_perc,
        image_downsample=args.downsample,
        upscale=args.upscale,
    )
    if args.compare_tiling:
        return [
            ("whole-image", RunConfig(**base, tile_image=False)),
            ("tiled", RunConfig(**base, tile_image=True, use_dask=False)),
            ("tiled+dask", RunConfig(**base, tile_image=True, use_dask=True)),
        ]
    return [("baseline", RunConfig(**base, tile_image=args.tile, use_dask=args.dask))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="oxford_pet")
    ap.add_argument("--num-images", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="image-subset selection seed")
    ap.add_argument("--fe", default="vgg16", help="feature extractor name")
    ap.add_argument("--scalings", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--scribble-perc", type=float, default=1.0,
                    help="target scribble coverage, percent of GT pixels")
    ap.add_argument("--downsample", type=int, default=1, help="FE input downsample factor")
    ap.add_argument("--upscale", type=int, default=1,
                    help="resize image+GT by this factor (reach the large-image regime)")
    ap.add_argument("--tile", action="store_true", help="model-side tiled prediction")
    ap.add_argument("--dask", action="store_true", help="dask-parallel tiled prediction")
    ap.add_argument("--compare-tiling", action="store_true",
                    help="run whole-image vs tiled vs tiled+dask and compare")
    ap.add_argument("--out", type=Path, default=None, help="write per-image results as JSON")
    args = ap.parse_args()

    print(f"Loading {args.num_images} images from {args.dataset} ...", flush=True)
    samples = load_samples(args.dataset, num_images=args.num_images, seed=args.seed)
    print(f"Loaded {len(samples)} images.\n", flush=True)

    configs = _configs(args)
    summaries = []
    all_results = {}
    for label, config in configs:
        print(f"=== config: {label} ===", flush=True)
        results = []
        for i, s in enumerate(samples):
            r = run_sample(s, config)
            results.append(r)
            tag = r.error if r.error else f"fg_iou={r.metrics.get('fg_iou', float('nan')):.3f}"
            print(f"  [{i+1}/{len(samples)}] {s.name}: "
                  f"train={r.train_time_s:.2f}s predict={r.predict_time_s:.2f}s "
                  f"mem={r.peak_mem_increase_mb:.0f}MB {tag}", flush=True)
        all_results[label] = [r.flat() for r in results]
        summaries.append(_summarize(results, label))
        print()

    print("=== SUMMARY (means) ===")
    print(_fmt_table(summaries))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"args": vars(args) | {"out": str(args.out)},
             "summaries": summaries, "results": all_results}, indent=2, default=str))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
