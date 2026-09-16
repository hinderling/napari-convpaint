"""Tests for the bounded feature cache (storage/eviction/budget logic)."""
import numpy as np

from napari_convpaint.feature_cache import FeatureCache


def _arr(mb):
    """A float32 array of approximately `mb` megabytes."""
    n = int(mb * 1e6 / 4)
    return np.zeros(n, dtype=np.float32)


def test_hit_and_miss():
    c = FeatureCache(max_bytes=100 * 10**6)
    assert c.get(("img", 0, "sig")) is None
    payload = _arr(1)
    c.put(("img", 0, "sig"), payload)
    got = c.get(("img", 0, "sig"))
    assert got is payload
    assert c.stats()["hits"] == 1
    assert c.stats()["misses"] == 1


def test_lru_eviction_by_cap():
    # Cap ~2.5 MB; each entry ~1 MB -> at most 2 fit, oldest evicted.
    c = FeatureCache(max_bytes=int(2.5 * 10**6))
    c.put(("a",), _arr(1))
    c.put(("b",), _arr(1))
    assert len(c) == 2
    c.put(("c",), _arr(1))  # evicts "a" (LRU)
    assert len(c) == 2
    assert c.get(("a",)) is None
    assert c.get(("b",)) is not None
    assert c.get(("c",)) is not None


def test_lru_touch_on_get_protects_entry():
    c = FeatureCache(max_bytes=int(2.5 * 10**6))
    c.put(("a",), _arr(1))
    c.put(("b",), _arr(1))
    assert c.get(("a",)) is not None  # touch "a" -> now "b" is LRU
    c.put(("c",), _arr(1))            # should evict "b", not "a"
    assert c.get(("a",)) is not None
    assert c.get(("b",)) is None


def test_single_oversize_payload_is_not_cached():
    c = FeatureCache(max_bytes=1 * 10**6)
    c.put(("big",), _arr(5))  # 5 MB into a 1 MB cap -> skipped, not cached
    assert len(c) == 0
    assert c.get(("big",)) is None


def test_overwrite_updates_size():
    c = FeatureCache(max_bytes=100 * 10**6)
    c.put(("k",), _arr(1))
    b0 = c.nbytes
    c.put(("k",), _arr(3))  # replace with a bigger payload
    assert c.nbytes > b0
    assert len(c) == 1


def test_clear():
    c = FeatureCache(max_bytes=100 * 10**6)
    c.put(("a",), _arr(1))
    c.put(("b",), _arr(1))
    c.clear()
    assert len(c) == 0
    assert c.nbytes == 0


def test_disabled_cache_is_noop():
    c = FeatureCache(max_bytes=100 * 10**6, enabled=False)
    c.put(("a",), _arr(1))
    assert c.get(("a",)) is None
    assert len(c) == 0


def test_list_payload_size_accounted():
    c = FeatureCache(max_bytes=int(2.5 * 10**6))
    c.put(("a",), [_arr(1), _arr(1)])  # ~2 MB as a list of arrays
    assert len(c) == 1
    c.put(("b",), _arr(1))  # pushes over 2.5 MB -> evicts "a"
    assert c.get(("a",)) is None


def test_model_feature_cache_identical_and_reuses():
    """With the cache on, a re-extraction of the same image reuses features and
    produces bit-identical output vs the cache off."""
    import warnings
    from napari_convpaint.convpaint_model import ConvpaintModel

    rng = np.random.default_rng(0)
    img = rng.random((64, 64), dtype=np.float32)
    annot = np.zeros((64, 64), dtype=np.uint8)
    annot[10:20, 10:20] = 1
    annot[40:50, 40:50] = 2

    def run(enable):
        m = ConvpaintModel(fe_name="gaussian_features")
        m.set_params(channel_mode="single")
        if enable:
            m.enable_feature_cache()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.train(img, annot)
            seg1 = np.asarray(m.segment(img))
            seg2 = np.asarray(m.segment(img))
        return m, seg1, seg2

    m_off, off1, off2 = run(False)
    m_on, on1, on2 = run(True)
    assert np.array_equal(off1, on1)
    assert np.array_equal(off2, on2)
    # cache actually stored and served something
    assert m_on._feature_cache.stats()["hits"] >= 1


# --- integration with the model-level cache protocol -----------------------

def test_cache_key_follows_user_params():
    """The key is built from the user's params (not the FE-enforced ones), so a
    change of e.g. fe_scalings changes the key even for FEs that enforce their own."""
    from napari_convpaint.convpaint_model import ConvpaintModel
    cp = ConvpaintModel('gaussian')
    sig_before = cp._fe_signature()
    cp.set_params(fe_scalings=[1, 2])
    assert cp._fe_signature() != sig_before


def test_cached_prediction_bit_identical_and_hits():
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.RandomState(0)
    img = rng.rand(1, 96, 96).astype(np.float32)
    annot = np.zeros((1, 96, 96), dtype=np.uint8)
    annot[0, :12, :12] = 1
    annot[0, -12:, -12:] = 2
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel('gaussian')
        fc = cp.enable_feature_cache(max_bytes=64 * 1024 * 1024)
        cp.train(img, annot)
        seg_first = cp.segment(img)
        hits_before = fc.stats()['hits']
        seg_second = cp.segment(img)
        assert fc.stats()['hits'] > hits_before          # second pass hits
        assert np.array_equal(seg_first, seg_second)
        # uncached model produces the identical segmentation
        cp2 = ConvpaintModel('gaussian')
        cp2.train(img, annot)
        assert np.array_equal(seg_second, cp2.segment(img))


def test_nn_fe_cache_hit_matches_fresh_and_uses_torch_payload():
    """NN FEs keep their native features on-device (torch); the cache payload
    is cast to numpy for storage but remembers it was torch, so hits are
    lifted back and reconstructed with the SAME torch backend as fresh
    extractions. Guards against a hit/miss backend split (skimage vs torch)
    which would make cache-enabled extraction both slow (CPU rescale) and
    potentially non-identical to fresh results."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.RandomState(0)
    img = rng.rand(1, 64, 64).astype(np.float32)
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel(fe_name='vgg16')
        cp.set_params(fe_scalings=[1, 2])
        feat_off = cp.get_feature_image(img)             # cache disabled: fresh
        fc = cp.enable_feature_cache(max_bytes=512 * 10**6)
        feat_miss = cp.get_feature_image(img)            # miss: fills cache
        feat_hit = cp.get_feature_image(img)             # hit: from payload
    assert fc.stats()['hits'] >= 1
    assert np.array_equal(feat_off, feat_miss), "cache-on (miss) differs from cache-off"
    assert np.array_equal(feat_miss, feat_hit), "cache hit differs from miss"
    payload = next(iter(fc._store.values()))[0]
    assert payload['was_torch'] is True
    for features, _, _ in payload['scales']:             # stored form is numpy
        assert all(isinstance(f, np.ndarray) for f in features)


def test_numpy_fe_payload_stays_numpy_and_identical():
    """Numpy-native FEs (e.g. gaussian) must NOT be lifted to torch on a hit —
    their fresh path is skimage, and hit/miss must keep sharing it."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.RandomState(0)
    img = rng.rand(1, 96, 96).astype(np.float32)
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel(fe_name='gaussian_features')
        feat_off = cp.get_feature_image(img)
        fc = cp.enable_feature_cache(max_bytes=256 * 10**6)
        feat_miss = cp.get_feature_image(img)
        feat_hit = cp.get_feature_image(img)
    assert np.array_equal(feat_off, feat_miss)
    assert np.array_equal(feat_miss, feat_hit)
    payload = next(iter(fc._store.values()))[0]
    assert payload['was_torch'] is False


def test_annotation_tiles_are_not_cached():
    """Training with tile_annotations extracts tiles cut around the scribbles; they
    never repeat and cannot serve a prediction, so they must not enter the cache.
    Untiled training and prediction of the same plane share one entry."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.RandomState(0)
    img = rng.rand(1, 96, 96).astype(np.float32)
    annot = np.zeros((1, 96, 96), dtype=np.uint8)
    annot[0, :12, :12] = 1
    annot[0, -12:, -12:] = 2
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel('gaussian')
        cp.set_params(tile_annotations=True)
        fc = cp.enable_feature_cache(max_bytes=64 * 1024 * 1024)
        cp.train(img, annot)
        assert len(fc) == 0                  # annotation tiles were not cached
        cp.segment(img)
        assert len(fc) == 1                  # the whole plane is
        cp.set_params(tile_annotations=False)
        hits_before = fc.stats()['hits']
        cp.train(img, annot)
        assert fc.stats()['hits'] > hits_before  # untiled training hits the plane entry
        assert len(fc) == 1


def test_planes_are_the_unit_of_reuse():
    """A stack, its single planes and (flattened) training planes share per-plane entries,
    and results are bit-identical with and without the cache."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.RandomState(0)
    stack = rng.rand(3, 64, 64).astype(np.float32)   # [Z, H, W], single channel
    annot = np.zeros((3, 64, 64), dtype=np.uint8)
    annot[1, :10, :10] = 1
    annot[1, -10:, -10:] = 2
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        # (no normalization, so that a plane is prepared identically alone and within its stack)
        cp_off = ConvpaintModel('gaussian')
        cp_off.set_params(tile_annotations=False, normalize=1)
        cp_off.train(stack, annot)
        seg_off = cp_off.segment(stack)

        cp = ConvpaintModel('gaussian')
        cp.set_params(tile_annotations=False, normalize=1)
        fc = cp.enable_feature_cache(max_bytes=64 * 1024 * 1024)
        cp.train(stack, annot)                         # extracts (and keeps) the annotated plane only
        assert len(fc) == 1
        seg1 = cp.segment(stack)                       # 1 plane reused, 2 extracted and kept
        assert len(fc) == 3 and fc.stats()['hits'] == 1
        seg2 = cp.segment(stack)                       # all planes reused
        assert fc.stats()['hits'] == 4
        seg_plane = cp.segment(stack[2])               # a single plane of the stack is reused too
        assert fc.stats()['hits'] == 5 and len(fc) == 3
    assert np.array_equal(seg1, seg_off) and np.array_equal(seg2, seg_off)
    assert np.array_equal(seg_plane, seg_off[2])


def test_3d_context_fe_is_reused_per_stack():
    """For an FE with 3D context, the whole stack (as passed) is the unit of reuse."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    rng = np.random.RandomState(0)
    stack = rng.rand(3, 64, 64).astype(np.float32)
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel('gaussian')
        cp.set_params(normalize=1)
        cp.fe_model.has_3d_context = True
        fc = cp.enable_feature_cache(max_bytes=64 * 1024 * 1024)
        f1 = cp.get_feature_image(stack)
        assert len(fc) == 1                            # one entry for the stack, not three
        f2 = cp.get_feature_image(stack)
        assert fc.stats()['hits'] == 1
        cp.get_feature_image(stack[1])                 # a single plane is another unit
        assert fc.stats()['hits'] == 1 and len(fc) == 2
    assert np.array_equal(f1, f2)
