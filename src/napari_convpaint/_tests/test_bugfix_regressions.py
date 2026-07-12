"""Regression tests for the bug-fix sweep on the performance branch.

Each test pins one of the fixed bugs so it cannot silently return:
tiling block math, downsample-aware tile alignment, feature-cache disk
routing / spill opt-out / FE-state cache keys, memory-mode img_ids
alignment, chunked classifier prediction, and nnlayers empty-selection
defaults. Uses the gaussian FE throughout (no weight downloads, fast)."""

import numpy as np
import pytest

from napari_convpaint import convpaint_model as cpm_mod
from napari_convpaint.convpaint_model import ConvpaintModel
from napari_convpaint.feature_cache import FeatureCache


def _mb(n):
    return np.zeros(int(n * 1024 * 1024), dtype=np.uint8)


def _trained_gaussian(im_size=64, scalings=None):
    cp = ConvpaintModel('gaussian')
    if scalings is not None:
        cp.set_params(fe_scalings=scalings, ignore_warnings=True)
    img = np.random.RandomState(0).rand(1, im_size, im_size).astype(np.float32)
    annot = np.zeros((1, im_size, im_size), dtype=np.uint8)
    annot[0, :10, :10] = 1
    annot[0, -10:, -10:] = 2
    cp.train(img, annot)
    return cp


# --------------------------------------------------------------------------
# Tiled prediction block math
# --------------------------------------------------------------------------

def test_tiled_predict_exact_multiple_no_extra_blocks():
    """An image side that is an exact multiple of the block size must not
    produce an extra (empty/wasted) tile per axis."""
    cp = _trained_gaussian()
    cp.set_params(tile_image=True, ignore_warnings=True)

    calls = []
    orig = cp._predict_image

    def counting(image, **kwargs):
        calls.append(np.asarray(image).shape)
        return orig(image, **kwargs)

    cp._predict_image = counting
    # gaussian: alignment=1, block 1000 -> maxblock 1000; H=W=2000 exact multiple
    img = np.random.RandomState(1).rand(2000, 2000).astype(np.float32)
    probas = cp._predict(img)
    assert len(calls) == 4, f"expected 2x2 tiles, got {len(calls)}: {calls}"
    assert probas.shape[-2:] == (2000, 2000)
    # No tile may be empty
    assert all(s[-1] > 0 and s[-2] > 0 for s in calls)


def test_tiled_equals_whole_image_with_downsample():
    """Auto-/forced tiling with image_downsample > 1 must produce the same
    output as the whole-image pass (tile origins on the downsample grid)."""
    cp = _trained_gaussian()
    cp.set_params(image_downsample=3, ignore_warnings=True)
    img = np.random.RandomState(2).rand(1602, 1602).astype(np.float32)

    cp.set_params(tile_image=True, ignore_warnings=True)
    tiled = cp._predict(img)

    cp.set_params(tile_image=False, ignore_warnings=True)
    old_min_side = cpm_mod.AUTO_TILE_MIN_SIDE
    cpm_mod.AUTO_TILE_MIN_SIDE = 10_000  # force the whole-image path
    try:
        whole = cp._predict(img)
    finally:
        cpm_mod.AUTO_TILE_MIN_SIDE = old_min_side

    assert tiled.shape == whole.shape
    # Tolerance sits above float32 accumulation noise (~1e-5, from computing
    # the same gaussian on different-width arrays) but far below the ~0.3 (bad
    # alignment) and ~0.02 (missing interpolation halo) seams this test pins.
    np.testing.assert_allclose(tiled, whole, rtol=0, atol=1e-4)


def test_tile_block_math_covers_all_sizes():
    """The kept regions of the tile loop must partition the image exactly for
    any padding/scaling/downsample combination (incl. margin >= block size),
    using the REAL geometry code (_tile_geometry), not a copy of it."""
    cp = ConvpaintModel('gaussian')

    def check(H, padding, scalings, downsample):
        cp.fe_model.padding = padding
        cp.set_params(fe_scalings=scalings, image_downsample=downsample,
                      ignore_warnings=True)
        maxblock, margin, nrows, _ = cp._tile_geometry((H, H))
        kept = []
        for row in range(nrows):
            min_row = max(0, row * maxblock - margin)
            min_row_ind = 0 if min_row == 0 else min_row + margin
            max_row_ind = min(min_row_ind + maxblock, H)
            kept.append((min_row_ind, max_row_ind))
        assert kept[0][0] == 0 and kept[-1][1] == H, (H, padding, scalings, downsample)
        assert all(a2 > a1 for a1, a2 in kept)
        assert all(k1[1] == k2[0] for k1, k2 in zip(kept, kept[1:]))

    for H in range(200, 4000, 379):
        for padding in (0, 12, 50, 130):
            for scalings in ([1], [1, 2], [1, 2, 4, 8]):
                for downsample in (1, 3, 7):
                    check(H, padding, scalings, downsample)


def test_tile_margin_floor_survives_downsample():
    """Padding-0 FEs must keep (at least) the legacy 50px overlap even with
    image_downsample > 1 — the fallback must apply before downsample scaling."""
    cp = ConvpaintModel('gaussian')
    cp.fe_model.padding = 0
    cp.set_params(image_downsample=4, ignore_warnings=True)
    _, margin, _, _ = cp._tile_geometry((2000, 2000))
    assert margin >= 50 * 4, f"margin {margin} lost the 50px floor under downsampling"


def test_fe_alignment_includes_downsample():
    cp = ConvpaintModel('gaussian')
    cp.set_params(fe_scalings=[1, 2], ignore_warnings=True)
    base = cp._get_fe_alignment(cp._param)
    cp.set_params(image_downsample=3, ignore_warnings=True)
    assert cp._get_fe_alignment(cp._param) == base * 3
    # Upscaling must not change the alignment
    cp.set_params(image_downsample=-3, ignore_warnings=True)
    assert cp._get_fe_alignment(cp._param) == base


# --------------------------------------------------------------------------
# Feature cache: storage tiers and keys
# --------------------------------------------------------------------------

def test_cache_oversized_payload_goes_to_disk():
    c = FeatureCache(max_bytes=1024 * 1024, headroom_frac=0.0,
                     disk_max_bytes=64 * 1024 * 1024)
    try:
        payload = _mb(2)  # 2 MB > 1 MB RAM cap
        c.put(('big',), payload)
        assert len(c) == 0
        assert c.stats()['disk_entries'] == 1
        got = c.get(('big',))
        assert got is not None and got.nbytes == payload.nbytes
    finally:
        c.close()


def test_cache_spill_ok_false_never_touches_disk():
    c = FeatureCache(max_bytes=1024 * 1024, headroom_frac=0.0,
                     disk_max_bytes=64 * 1024 * 1024)
    try:
        c.put(('a',), _mb(0.6), spill_ok=False)
        c.put(('b',), _mb(0.6))  # evicts 'a' -> must be dropped, not spilled
        assert c.stats()['disk_entries'] == 0
        assert c.get(('a',)) is None
        # An oversized no-spill payload is not cached anywhere
        c.put(('huge',), _mb(2), spill_ok=False)
        assert c.get(('huge',)) is None
        assert c.stats()['disk_entries'] == 0
    finally:
        c.close()


def test_hookmodel_opts_out_of_disk_spill():
    from napari_convpaint.feature_extractor import FeatureExtractor
    assert FeatureExtractor.cache_spill_to_disk(object()) is True
    from napari_convpaint.feature_extractors.nnlayers import Hookmodel
    assert Hookmodel.cache_spill_to_disk(object()) is False


def test_cache_key_includes_gaussian_sigma():
    cp = ConvpaintModel('gaussian')
    sig_before = cp._fe_cache_signature(cp._param)
    cp.fe_model.sigma = cp.fe_model.sigma + 1
    sig_after = cp._fe_cache_signature(cp._param)
    assert sig_before != sig_after


def test_cache_key_includes_jafar_scalings_state():
    """The signature must change when the FE reports different extra state
    (the mechanism JAFAR uses for jafar_scalings)."""
    cp = ConvpaintModel('gaussian')
    sig1 = cp._fe_cache_signature(cp._param)
    orig = cp.fe_model.cache_extra_state
    cp.fe_model.cache_extra_state = lambda p: ('jafar_scalings', (1, 8))
    try:
        sig2 = cp._fe_cache_signature(cp._param)
    finally:
        cp.fe_model.cache_extra_state = orig
    assert sig1 != sig2


# --------------------------------------------------------------------------
# Memory-mode img_ids alignment
# --------------------------------------------------------------------------

def test_memory_mode_img_ids_follow_kept_images():
    cp = ConvpaintModel('gaussian')
    rng = np.random.RandomState(4)
    img_a = rng.rand(1, 32, 32).astype(np.float32)
    img_b = rng.rand(1, 32, 32).astype(np.float32)
    empty = np.zeros((1, 32, 32), dtype=np.uint8)
    annot_b = np.zeros((1, 32, 32), dtype=np.uint8)
    annot_b[0, :8, :8] = 1
    annot_b[0, -8:, -8:] = 2
    out = cp._get_features([img_a, img_b], annotations=[empty, annot_b],
                           restore_input_form=False, memory_mode=True,
                           img_ids=['A', 'B'])
    features, annots, coords, img_ids, scale = out
    assert len(features) == len(img_ids)
    assert set(img_ids) == {'B'}, f"ids misaligned: {img_ids}"


# --------------------------------------------------------------------------
# Chunked classifier prediction
# --------------------------------------------------------------------------

def test_clf_predict_chunked_matches_single_shot():
    cp = _trained_gaussian(48)
    feats = np.random.RandomState(5).rand(4, 1300, 900).astype(np.float32)
    full = cp.classifier.predict_proba(
        np.moveaxis(feats, 0, -1).reshape(-1, feats.shape[0]))
    chunked = cp._clf_predict(feats, return_proba=True)
    assert np.array_equal(np.moveaxis(full, -1, 0), chunked)


# --------------------------------------------------------------------------
# nnlayers empty-selection defaults
# --------------------------------------------------------------------------

def test_nnlayers_empty_selection_resets_properties():
    pytest.importorskip('torchvision')
    from napari_convpaint.feature_extractors.nnlayers import Hookmodel
    fe = Hookmodel(model_name='vgg16')
    assert fe.padding > 0
    fe.selected_layers = []
    fe._compute_nn_properties()
    assert fe.padding == 0
    assert fe.patch_size == 1
    assert fe.has_global_context is False
