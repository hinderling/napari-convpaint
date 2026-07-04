"""Segmentation accuracy metrics for the benchmark.

All metrics compare a full-image prediction against the semantic ground truth
(both with class labels starting at 1). Averaging over classes gives the mean
metrics; for the binary foreground/background datasets used here the
foreground-class IoU/Dice is usually the most informative single number.
"""
from __future__ import annotations

import numpy as np


def _class_labels(gt: np.ndarray) -> list[int]:
    return [int(c) for c in np.unique(gt) if c != 0]


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    """Fraction of pixels whose predicted label matches the GT (over labelled px)."""
    mask = gt != 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(pred[mask] == gt[mask]))


def per_class_iou(pred: np.ndarray, gt: np.ndarray) -> dict[int, float]:
    """Intersection-over-union per class."""
    out = {}
    for c in _class_labels(gt):
        pred_c = pred == c
        gt_c = gt == c
        inter = np.count_nonzero(pred_c & gt_c)
        union = np.count_nonzero(pred_c | gt_c)
        out[c] = inter / union if union else float("nan")
    return out


def per_class_dice(pred: np.ndarray, gt: np.ndarray) -> dict[int, float]:
    """Dice coefficient (F1) per class."""
    out = {}
    for c in _class_labels(gt):
        pred_c = pred == c
        gt_c = gt == c
        inter = np.count_nonzero(pred_c & gt_c)
        denom = np.count_nonzero(pred_c) + np.count_nonzero(gt_c)
        out[c] = 2 * inter / denom if denom else float("nan")
    return out


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """All metrics for one image, as a flat dict of floats.

    Includes per-class and mean IoU/Dice plus pixel accuracy. The mean is over
    classes present in the GT (ignoring NaNs)."""
    iou = per_class_iou(pred, gt)
    dice = per_class_dice(pred, gt)
    result = {
        "pixel_accuracy": pixel_accuracy(pred, gt),
        "mean_iou": float(np.nanmean(list(iou.values()))) if iou else float("nan"),
        "mean_dice": float(np.nanmean(list(dice.values()))) if dice else float("nan"),
    }
    for c, v in iou.items():
        result[f"iou_class_{c}"] = v
    for c, v in dice.items():
        result[f"dice_class_{c}"] = v
    # Foreground convention: class 2 is foreground in the binary datasets.
    if 2 in iou:
        result["fg_iou"] = iou[2]
        result["fg_dice"] = dice[2]
    return result
