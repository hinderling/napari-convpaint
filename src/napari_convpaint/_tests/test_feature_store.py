"""Tests for the persistent feature store (storage logic; model integration is tested below)."""
import os

import numpy as np
import pytest

from napari_convpaint.feature_store import FeatureStore, _MARKER


def _payload(rng, num_planes=1, was_torch=False):
    """A payload like the FE protocol produces: 2 scales, 2 arrays each, [F, Z, h, w]."""
    scales = []
    for h in (16, 8):
        arrays = [rng.random((5, num_planes, h, h), dtype=np.float32) for _ in range(2)]
        scales.append((arrays, (3, num_planes, h, h), (3, num_planes, h, h)))
    return {"scales": scales, "was_torch": was_torch}


def _equal(p1, p2):
    if p1["was_torch"] != p2["was_torch"] or len(p1["scales"]) != len(p2["scales"]):
        return False
    for (a1, pre1, red1), (a2, pre2, red2) in zip(p1["scales"], p2["scales"]):
        if tuple(pre1) != tuple(pre2) or tuple(red1) != tuple(red2) or len(a1) != len(a2):
            return False
        if not all(np.array_equal(x, y) for x, y in zip(a1, a2)):
            return False
    return True


def test_store_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    store = FeatureStore(tmp_path / "store")
    key = ("abc123", (("fe_name", "gaussian_features"), ("fe_scalings", (1, 2))))
    assert key not in store
    assert store.get(key) is None
    payload = _payload(rng, was_torch=True)
    store.put(key, payload)
    assert key in store and len(store) == 1
    got = store.get(key)
    assert _equal(got, payload)
    assert store.stats()["hits"] == 1 and store.stats()["misses"] == 1
    assert store.nbytes > 0
    # Memory-mapped arrays must be usable like normal arrays (incl. torch lifting, copy-on-write)
    import torch
    t = torch.from_numpy(got["scales"][0][0][0])
    assert t.shape == (5, 1, 16, 16)


def test_store_keys_and_idempotent_put(tmp_path):
    rng = np.random.default_rng(1)
    store = FeatureStore(tmp_path / "store")
    p1, p2 = _payload(rng), _payload(rng)
    key1 = ("data1", ("sig",))
    key2 = ("data1", ("other_sig",))  # same data, other FE settings -> other entry
    store.put(key1, p1)
    store.put(key2, p2)
    assert len(store) == 2
    assert _equal(store.get(key1), p1) and _equal(store.get(key2), p2)
    store.put(key1, p2)  # already present -> not overwritten
    assert _equal(store.get(key1), p1)


def test_store_clear_and_reopen(tmp_path):
    rng = np.random.default_rng(2)
    folder = tmp_path / "store"
    store = FeatureStore(folder)
    store.put(("d", ("s",)), _payload(rng))
    # Reopening an existing store finds the entries
    store2 = FeatureStore(folder)
    assert ("d", ("s",)) in store2 and len(store2) == 1
    store2.clear()
    assert len(store) == 0 and os.path.isfile(folder / _MARKER)


def test_store_size_cap(tmp_path):
    rng = np.random.default_rng(3)
    p1 = _payload(rng)
    one = sum(a.nbytes for arrays, _, _ in p1["scales"] for a in arrays)
    store = FeatureStore(tmp_path / "store", max_bytes=int(2.5 * one))
    store.put(("a", ("s",)), p1)
    store.put(("b", ("s",)), _payload(rng))
    assert len(store) == 2 and store.nbytes >= 2 * one
    with pytest.warns(UserWarning, match="size cap"):
        store.put(("c", ("s",)), _payload(rng))       # would exceed the cap -> not stored
    assert len(store) == 2 and ("c", ("s",)) not in store
    store.clear()
    assert store.nbytes == 0
    store.put(("c", ("s",)), _payload(rng))           # room again after clearing
    assert len(store) == 1
    # A reopened store knows its size
    assert FeatureStore(tmp_path / "store").nbytes == store.nbytes


def test_store_refuses_foreign_folder(tmp_path):
    (tmp_path / "somefile.txt").write_text("not a store")
    with pytest.raises(ValueError):
        FeatureStore(tmp_path)
    # An empty folder is fine
    FeatureStore(tmp_path / "empty")


# --- integration with the model --------------------------------------------

def _stack_and_annot():
    rng = np.random.RandomState(0)
    stack = rng.rand(3, 64, 64).astype(np.float32)   # [Z, H, W], single channel
    annot = np.zeros((3, 64, 64), dtype=np.uint8)
    annot[1, :10, :10] = 1
    annot[1, -10:, -10:] = 2
    return stack, annot


def test_model_store_serves_across_instances(tmp_path):
    """Features extracted with the store on are reused by another model instance with the
    same folder (no cache), bit-identical; annotation tiles are never stored."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    stack, annot = _stack_and_annot()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp_off = ConvpaintModel('gaussian')
        cp_off.set_params(tile_annotations=False, normalize=1)
        cp_off.train(stack, annot)
        seg_off = cp_off.segment(stack)

        cp1 = ConvpaintModel('gaussian')
        cp1.set_params(tile_annotations=True, normalize=1)
        store = cp1.enable_feature_store(tmp_path / "store")
        cp1.train(stack, annot)                        # annotation tiles -> nothing stored
        assert len(store) == 0
        cp1.set_params(tile_annotations=False)
        cp1.train(stack, annot)                        # the annotated plane is stored
        assert len(store) == 1
        seg1 = cp1.segment(stack)                      # 1 plane reused, 2 stored
        assert len(store) == 3 and store.stats()['hits'] == 1

        cp2 = ConvpaintModel('gaussian')               # a new model (e.g. a new session)
        cp2.set_params(tile_annotations=False, normalize=1)
        store2 = cp2.enable_feature_store(tmp_path / "store")
        cp2.train(stack, annot)
        seg2 = cp2.segment(stack)
        assert store2.stats()['hits'] == 4 and store2.stats()['misses'] == 0
        cp2.disable_feature_store()
        assert cp2._feature_store is None and len(store2) == 3   # files are kept
    assert np.array_equal(seg1, seg_off) and np.array_equal(seg2, seg_off)


def test_model_cache_and_store_together(tmp_path):
    """With both enabled, the cache is consulted first and misses are kept in both."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    stack, _ = _stack_and_annot()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp = ConvpaintModel('gaussian')
        cp.set_params(normalize=1)
        fc = cp.enable_feature_cache()
        store = cp.enable_feature_store(tmp_path / "store")
        f1 = cp.get_feature_image(stack)
        assert len(fc) == 3 and len(store) == 3
        f2 = cp.get_feature_image(stack)
        assert fc.stats()['hits'] == 3 and store.stats()['hits'] == 0
        cp.disable_feature_cache()
        f3 = cp.get_feature_image(stack)
        assert store.stats()['hits'] == 3
    assert np.array_equal(f1, f2) and np.array_equal(f1, f3)


def test_store_features_prepares_a_stack(tmp_path):
    """store_features fills the store plane by plane; train and predict then only reuse."""
    import warnings as _w
    from napari_convpaint.convpaint_model import ConvpaintModel
    stack, annot = _stack_and_annot()
    with _w.catch_warnings():
        _w.simplefilter('ignore')
        cp_off = ConvpaintModel('gaussian')
        cp_off.set_params(tile_annotations=False)
        cp_off.train(stack, annot)
        seg_off = cp_off.segment(stack)

        cp = ConvpaintModel('gaussian')
        cp.set_params(tile_annotations=False)          # default normalization (over the stack)
        store = cp.enable_feature_store(tmp_path / "store")
        with pytest.raises(ValueError):
            ConvpaintModel('gaussian').store_features(stack)   # no store enabled
        cp.store_features(stack)
        assert len(store) == 3 and store.stats()['misses'] == 3
        cp.store_features(stack)                       # already stored -> nothing new
        assert len(store) == 3 and store.stats()['hits'] == 3
        cp.train(stack, annot)
        seg = cp.segment(stack)
        assert store.stats()['misses'] == 3            # no extraction anymore
    assert np.array_equal(seg, seg_off)


# --- widget ------------------------------------------------------------------

def test_widget_store_controls(make_napari_viewer, tmp_path, monkeypatch):
    """Checkbox connects/disconnects the store on the model; train+predict fill it;
    delete (confirmed) empties it but keeps it active; unchecking keeps the files."""
    from qtpy.QtWidgets import QMessageBox
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    from napari_convpaint.testing_data import generate_synthetic_square, generate_synthetic_circle_annotation
    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    assert w.cp_model._feature_store is None and not w.check_use_store.isChecked()
    assert w.store_folder.endswith('feature_store')            # default folder in the user cache dir

    w.store_folder = str(tmp_path / 'store')
    w.check_use_store.setChecked(True)
    fs = w.cp_model._feature_store
    assert fs is not None and fs.folder == w.store_folder and w.btn_store_delete.isEnabled()

    im, _ = generate_synthetic_square(im_dims=(252, 252), square_dims=(70, 70))
    im_annot = generate_synthetic_circle_annotation(im_dims=(252, 252), circle1_xy=(125, 70), circle2_xy=(125, 125))
    viewer.add_image(im)
    w._on_add_annot_layer()
    viewer.layers['annotations'].data[...] = im_annot
    w.cp_model.set_params(channel_mode='rgb')
    w._on_train()
    w._on_predict()
    assert len(fs) >= 1 and 'Stored features: 1 planes' in w.store_size_label.text()

    monkeypatch.setattr(QMessageBox, 'question', lambda *a, **k: QMessageBox.Yes)
    w._on_delete_stored_features()
    assert len(fs) == 0 and w.cp_model._feature_store is fs   # emptied, still active

    w.check_use_store.setChecked(False)
    assert w.cp_model._feature_store is None and not w.btn_store_delete.isEnabled()
    assert (tmp_path / 'store' / 'convpaint_feature_store.json').is_file()


def test_widget_store_features_of_stack(make_napari_viewer, tmp_path):
    """The 'Store features' action stores every plane of the selected stack; a following
    stack prediction only reuses."""
    from napari_convpaint.convpaint_widget import ConvpaintWidget
    from napari_convpaint.testing_data import generate_synthetic_square, generate_synthetic_circle_annotation
    viewer = make_napari_viewer()
    w = ConvpaintWidget(viewer)
    w.ensure_init()
    w.store_folder = str(tmp_path / 'store')
    w.check_use_store.setChecked(True)
    fs = w.cp_model._feature_store

    im, _ = generate_synthetic_square(im_dims=(252, 252), square_dims=(70, 70))
    stack = np.stack([im, im[::-1], im[:, ::-1]])           # 3 different planes [Z, H, W, 3] (RGB)
    viewer.add_image(stack, rgb=True)
    w._on_select_layer()                                    # (the changed-signal handler is QTimer-delayed)
    assert w.cp_model.get_param('channel_mode') == 'rgb'
    w._on_store_features()
    assert len(fs) == 3 and fs.stats()['misses'] == 3

    im_annot = generate_synthetic_circle_annotation(im_dims=(252, 252), circle1_xy=(125, 70), circle2_xy=(125, 125))
    w._on_add_annot_layer()
    viewer.layers['annotations'].data[0] = im_annot
    w.check_tile_annotations.setChecked(False)
    w._on_train()
    assert len(fs) == 3 and fs.stats()['misses'] == 3, fs.stats()   # training reused the stored plane
    w._on_predict_all()
    assert len(fs) == 3 and fs.stats()['misses'] == 3, fs.stats()   # nothing extracted anymore
