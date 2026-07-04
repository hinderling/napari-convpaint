"""Run one Convpaint train+predict cycle on a benchmark sample and measure it.

Measures, per image:
  - train time (feature extraction on scribbled pixels + classifier fit)
  - predict time (feature extraction on the full image + classifier predict)
  - peak process RSS during the cycle (MB)
  - accuracy of the full-image prediction vs. ground truth

The image is passed to the model channel-first (RGB -> [3, H, W]), matching what
the widget does internally, and ``channel_mode='rgb'`` is set so the default
VGG16 feature extractor uses its ImageNet path.
"""
from __future__ import annotations

import contextlib
import gc
import io
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from .datasets import Sample
from .scribbles import make_scribbles, scribble_coverage
from .metrics import compute_metrics
from .memtrack import track_peak_rss


@dataclass
class RunConfig:
    """Knobs for one benchmark configuration."""
    fe_name: str = "vgg16"
    channel_mode: str = "rgb"
    scalings: tuple = (1, 2)
    layers: tuple = (0,)             # FE layer indices (VGG16)
    scribble_perc: float = 1.0       # target scribble coverage (percent of GT px)
    tile_image: bool = False         # model-side tiling for prediction
    use_dask: bool = False           # dask-parallel tiled prediction
    image_downsample: int = 1        # FE input downsampling
    upscale: int = 1                 # resize image+GT by this factor (large-image regime)
    seed: int = 1                    # scribble RNG seed


@dataclass
class RunResult:
    name: str
    config: dict
    scribble_coverage_pct: float
    train_time_s: float
    predict_time_s: float
    total_time_s: float
    peak_mem_increase_mb: float
    n_pixels: int
    metrics: dict = field(default_factory=dict)
    error: str | None = None

    def flat(self) -> dict:
        """Flatten for tabular reporting."""
        d = {
            "name": self.name,
            "scribble_pct": round(self.scribble_coverage_pct, 3),
            "train_s": round(self.train_time_s, 3),
            "predict_s": round(self.predict_time_s, 3),
            "total_s": round(self.total_time_s, 3),
            "peak_mem_mb": round(self.peak_mem_increase_mb, 1),
            "n_pixels": self.n_pixels,
        }
        for k, v in self.metrics.items():
            d[k] = round(v, 4) if isinstance(v, float) else v
        if self.error:
            d["error"] = self.error
        return d


def _to_channel_first(image: np.ndarray, channel_mode: str) -> np.ndarray:
    """Match the widget's ``_get_data_channel_first`` for the shapes we use."""
    if channel_mode == "rgb":
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f"RGB mode expects [H, W, >=3], got {image.shape}")
        return np.moveaxis(image[:, :, :3], -1, 0)  # [3, H, W]
    # single-channel: take a 2D plane
    if image.ndim == 3:
        image = image[:, :, 0]
    return image


def _build_model(config: RunConfig):
    from napari_convpaint.convpaint_model import ConvpaintModel

    model = ConvpaintModel(fe_name=config.fe_name)
    params = dict(
        channel_mode=config.channel_mode,
        fe_scalings=list(config.scalings),
        tile_image=config.tile_image,
        image_downsample=config.image_downsample,
    )
    model.set_params(**params)
    if config.fe_name == "vgg16":
        # Select FE layers by index -> layer keys. Setting fe_layers on an
        # existing model normally warns (it resets training); harmless here
        # because we set it before the first train, so silence it.
        layer_keys = model.get_fe_layer_keys()
        if layer_keys:
            model.set_params(
                fe_layers=[layer_keys[i] for i in config.layers],
                ignore_warnings=True,
            )
    return model


@contextlib.contextmanager
def _quiet_stdout():
    """Suppress CatBoost's per-iteration training log (printed to stdout)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _upscale_sample(sample: Sample, factor: int) -> Sample:
    """Resize image (bilinear) and GT (nearest) by an integer factor, to reach
    the large-image regime where feature-stack memory and tiling actually matter.
    Nearest-neighbour on the GT keeps labels exact."""
    if factor == 1:
        return sample
    from skimage.transform import resize
    h, w = sample.gt.shape
    new_hw = (h * factor, w * factor)
    image = resize(sample.image, (*new_hw, sample.image.shape[2]),
                   order=1, preserve_range=True, anti_aliasing=True).astype(np.uint8)
    gt = resize(sample.gt, new_hw, order=0, preserve_range=True,
                anti_aliasing=False).astype(np.uint8)
    return Sample(image=image, gt=gt, name=f"{sample.name}_x{factor}")


def run_sample(sample: Sample, config: RunConfig | None = None) -> RunResult:
    """Train on generated scribbles, predict the whole image, measure it."""
    config = config or RunConfig()
    sample = _upscale_sample(sample, config.upscale)

    gt = sample.gt
    scribbles = make_scribbles(gt, max_perc=config.scribble_perc, seed=config.seed)
    coverage = scribble_coverage(scribbles, gt)

    image_cf = _to_channel_first(sample.image, config.channel_mode)
    n_pixels = int(gt.shape[-1] * gt.shape[-2])

    gc.collect()
    result_error = None
    metrics = {}
    seg = None
    with track_peak_rss() as mem:
        try:
            model = _build_model(config)

            with _quiet_stdout():
                t0 = time.perf_counter()
                model.train(image_cf, scribbles)
                t_train = time.perf_counter() - t0

                t1 = time.perf_counter()
                seg = model.segment(image_cf, use_dask=config.use_dask)
                t_predict = time.perf_counter() - t1

            seg = np.asarray(seg)
            # segment() drops the channel dim -> [H, W]; align to gt for scoring.
            seg = seg.reshape(gt.shape)
            metrics = compute_metrics(seg, gt)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the sweep
            result_error = f"{type(exc).__name__}: {exc}"
            t_train = t_predict = float("nan")

    return RunResult(
        name=sample.name,
        config=asdict(config),
        scribble_coverage_pct=coverage,
        train_time_s=t_train,
        predict_time_s=t_predict,
        total_time_s=(t_train + t_predict) if result_error is None else float("nan"),
        peak_mem_increase_mb=mem.peak_increase_mb,
        n_pixels=n_pixels,
        metrics=metrics,
        error=result_error,
    )
