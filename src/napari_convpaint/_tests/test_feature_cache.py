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


def test_invalidate_predicate():
    c = FeatureCache(max_bytes=100 * 10**6, headroom_frac=0.0)
    c.put(("imgA", 0, "sig"), _arr(1))
    c.put(("imgA", 1, "sig"), _arr(1))
    c.put(("imgB", 0, "sig"), _arr(1))
    c.invalidate(lambda key: key[0] == "imgA")  # drop all imgA slices
    assert c.get(("imgA", 0, "sig")) is None
    assert c.get(("imgA", 1, "sig")) is None
    assert c.get(("imgB", 0, "sig")) is not None
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
