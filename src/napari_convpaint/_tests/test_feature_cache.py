"""Tests for the bounded feature cache (storage/eviction/budget logic)."""
import numpy as np

from napari_convpaint.feature_cache import FeatureCache


def _arr(mb):
    """A float32 array of approximately `mb` megabytes."""
    n = int(mb * 1e6 / 4)
    return np.zeros(n, dtype=np.float32)


def test_hit_and_miss():
    c = FeatureCache(max_bytes=100 * 10**6, headroom_frac=0.0)
    assert c.get(("img", 0, "sig")) is None
    payload = _arr(1)
    c.put(("img", 0, "sig"), payload)
    got = c.get(("img", 0, "sig"))
    assert got is payload
    assert c.stats()["hits"] == 1
    assert c.stats()["misses"] == 1


def test_lru_eviction_by_cap():
    # Cap ~2.5 MB; each entry ~1 MB -> at most 2 fit, oldest evicted.
    c = FeatureCache(max_bytes=int(2.5 * 10**6), headroom_frac=0.0)
    c.put(("a",), _arr(1))
    c.put(("b",), _arr(1))
    assert len(c) == 2
    c.put(("c",), _arr(1))  # evicts "a" (LRU)
    assert len(c) == 2
    assert c.get(("a",)) is None
    assert c.get(("b",)) is not None
    assert c.get(("c",)) is not None


def test_lru_touch_on_get_protects_entry():
    c = FeatureCache(max_bytes=int(2.5 * 10**6), headroom_frac=0.0)
    c.put(("a",), _arr(1))
    c.put(("b",), _arr(1))
    assert c.get(("a",)) is not None  # touch "a" -> now "b" is LRU
    c.put(("c",), _arr(1))            # should evict "b", not "a"
    assert c.get(("a",)) is not None
    assert c.get(("b",)) is None


def test_single_oversize_payload_is_not_cached():
    c = FeatureCache(max_bytes=1 * 10**6, headroom_frac=0.0)
    c.put(("big",), _arr(5))  # 5 MB into a 1 MB cap -> skipped, not cached
    assert len(c) == 0
    assert c.get(("big",)) is None


def test_overwrite_updates_size():
    c = FeatureCache(max_bytes=100 * 10**6, headroom_frac=0.0)
    c.put(("k",), _arr(1))
    b0 = c.nbytes
    c.put(("k",), _arr(3))  # replace with a bigger payload
    assert c.nbytes > b0
    assert len(c) == 1


def test_clear():
    c = FeatureCache(max_bytes=100 * 10**6, headroom_frac=0.0)
    c.put(("a",), _arr(1))
    c.put(("b",), _arr(1))
    c.clear()
    assert len(c) == 0
    assert c.nbytes == 0


def test_disabled_cache_is_noop():
    c = FeatureCache(max_bytes=100 * 10**6, headroom_frac=0.0, enabled=False)
    c.put(("a",), _arr(1))
    assert c.get(("a",)) is None
    assert len(c) == 0


def test_list_payload_size_accounted():
    c = FeatureCache(max_bytes=int(2.5 * 10**6), headroom_frac=0.0)
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
            m.enable_feature_cache(True)
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


def test_disk_spillover_serves_ram_evicted_entries():
    """RAM-evicted entries spill to disk and are served from there (bit-identical)."""
    c = FeatureCache(max_bytes=int(2.5 * 10**6), headroom_frac=0.0,
                     disk_max_bytes=100 * 10**6)
    a = _arr(1); b = _arr(1); d = _arr(1)
    c.put(("a",), a); c.put(("b",), b)  # RAM full (2 entries)
    c.put(("c",), d)  # evicts "a" from RAM -> spills to disk
    assert len(c) == 2 and c.stats()["disk_entries"] == 1
    got = c.get(("a",))  # RAM miss -> disk hit
    assert got is not None and np.array_equal(got, a)  # round-trips bit-identical
    assert c.stats()["disk_hits"] == 1


def test_disk_lru_eviction_and_total_miss():
    c = FeatureCache(max_bytes=int(1.5 * 10**6), headroom_frac=0.0,
                     disk_max_bytes=int(1.5 * 10**6))  # RAM holds 1, disk holds 1
    c.put(("a",), _arr(1)); c.put(("b",), _arr(1))  # a -> disk, b in RAM
    c.put(("c",), _arr(1))  # b -> disk (evicts a from disk), c in RAM
    assert c.get(("a",)) is None      # a fell off disk entirely -> recompute
    assert c.get(("b",)) is not None  # b on disk
    assert c.get(("c",)) is not None  # c in RAM


def test_disk_disabled_by_default():
    c = FeatureCache(max_bytes=int(1.5 * 10**6), headroom_frac=0.0)  # no disk
    c.put(("a",), _arr(1)); c.put(("b",), _arr(1))  # a evicted, dropped (no disk)
    assert c.get(("a",)) is None and c.stats()["disk_entries"] == 0


def test_clear_removes_disk_tier_and_tempdir():
    import os
    c = FeatureCache(max_bytes=int(1.5 * 10**6), headroom_frac=0.0,
                     disk_max_bytes=100 * 10**6)
    c.put(("a",), _arr(1)); c.put(("b",), _arr(1))  # a on disk
    disk_dir = c._disk_dir
    assert disk_dir is not None and os.path.isdir(disk_dir)
    c.clear()
    assert c.stats()["disk_entries"] == 0 and c.disk_nbytes == 0
    c.close()
    assert not os.path.isdir(disk_dir)  # temp dir removed


def test_disk_bytes_never_exceeds_cap():
    """Stress: many puts must never push the disk tier over its byte cap."""
    cap = int(3.5 * 10**6)  # ~3 entries of 1 MB
    c = FeatureCache(max_bytes=int(1.5 * 10**6), headroom_frac=0.0, disk_max_bytes=cap)
    for i in range(20):
        c.put((i,), _arr(1))
        assert c.disk_nbytes <= cap  # invariant holds after every put
    c.close()


# --- integration with the model-level cache protocol -----------------------

def test_oversized_payload_goes_to_disk_tier():
    from napari_convpaint.feature_cache import FeatureCache
    c = FeatureCache(max_bytes=1024 * 1024, headroom_frac=0.0,
                     disk_max_bytes=64 * 1024 * 1024)
    try:
        payload = np.zeros(2 * 1024 * 1024, dtype=np.uint8)  # 2 MB > 1 MB RAM cap
        c.put(('big',), payload)
        assert len(c) == 0
        assert c.stats()['disk_entries'] == 1
        got = c.get(('big',))
        assert got is not None and got.nbytes == payload.nbytes
    finally:
        c.close()


def test_spill_ok_false_never_touches_disk():
    from napari_convpaint.feature_cache import FeatureCache
    c = FeatureCache(max_bytes=1024 * 1024, headroom_frac=0.0,
                     disk_max_bytes=64 * 1024 * 1024)
    try:
        c.put(('a',), np.zeros(600 * 1024, dtype=np.uint8), spill_ok=False)
        c.put(('b',), np.zeros(600 * 1024, dtype=np.uint8))  # evicts 'a' -> dropped
        assert c.stats()['disk_entries'] == 0
        assert c.get(('a',)) is None
        c.put(('huge',), np.zeros(2 * 1024 * 1024, dtype=np.uint8), spill_ok=False)
        assert c.get(('huge',)) is None
        assert c.stats()['disk_entries'] == 0
    finally:
        c.close()


def test_hookmodel_opts_out_of_disk_spill():
    from napari_convpaint.feature_extractor import FeatureExtractor
    assert FeatureExtractor.cache_spill_to_disk(object()) is True
    from napari_convpaint.feature_extractors.nnlayers import Hookmodel
    assert Hookmodel.cache_spill_to_disk(object()) is False


def test_cache_key_includes_fe_instance_state():
    from napari_convpaint.convpaint_model import ConvpaintModel
    cp = ConvpaintModel('gaussian')
    sig_before = cp._fe_cache_signature(cp._param)
    cp.fe_model.sigma = cp.fe_model.sigma + 1
    assert cp._fe_cache_signature(cp._param) != sig_before
    # generic hook: any change in reported extra state must change the key
    orig = cp.fe_model.cache_extra_state
    cp.fe_model.cache_extra_state = lambda: ('jafar_scalings', (1, 8))
    try:
        assert cp._fe_cache_signature(cp._param) != sig_before
    finally:
        cp.fe_model.cache_extra_state = orig


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
        # peek semantics
        assert cp._predict(rng.rand(1, 96, 96).astype(np.float32), cache_only=True) is None
        assert cp._predict(img, cache_only=True) is not None
        # uncached model produces the identical segmentation
        cp2 = ConvpaintModel('gaussian')
        cp2.train(img, annot)
        assert np.array_equal(seg_second, cp2.segment(img))


def test_thread_safety_under_concurrent_use():
    """Hammer the cache from worker threads while the "GUI" thread clears it and
    changes limits (exactly what the napari widget does during a threaded op).
    Correctness bar: no exceptions and consistent bookkeeping afterwards."""
    import threading

    c = FeatureCache(max_bytes=int(3 * 10**6), headroom_frac=0.0,
                     disk_max_bytes=int(5 * 10**6))
    errors = []
    start = threading.Barrier(5)

    def worker(tid):
        try:
            start.wait()
            for i in range(200):
                key = ("img", tid, i % 7)
                if c.get(key) is None:
                    c.put(key, _arr(0.1))
                len(c), c.stats()
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    def gui():
        try:
            start.wait()
            for i in range(100):
                c.set_max_bytes(int((2 + i % 3) * 10**6))
                c.set_disk_max_bytes(int((i % 2) * 5 * 10**6))
                c.stats()
                if i % 10 == 0:
                    c.clear()
        except Exception as e:  # pragma: no cover - only on regression
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    threads.append(threading.Thread(target=gui))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    # Bookkeeping must be consistent: recompute sizes from the stores.
    assert c.nbytes == sum(item[1] for item in c._store.values())
    assert c.disk_nbytes == sum(item[1] for item in c._disk_store.values())
    assert c.nbytes <= c.stats()["max_bytes"]
    c.close()


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
