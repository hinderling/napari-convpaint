"""Persistent feature store for stacks and movies (the disk counterpart of feature_cache.py).

Keeps the native (pre-rescale) features of every processed plane as .npy files in a
user-chosen folder, without eviction: extract a stack once, then train and predict from
the stored features (also across sessions). Entries use the same key as the RAM cache
(content hash of the prepared plane + FE signature), so the store is self-invalidating:
a changed image or setting simply misses. Hits are memory-mapped, so only the planes
actually used are read from disk.
"""
import hashlib
import json
import os
import shutil
import warnings

import numpy as np


_MARKER = "convpaint_feature_store.json" # Marks a folder as a feature store (only such folders are cleared)
_META = "meta.json" # Per-entry description (shapes, native form)
_DISK_HEADROOM_BYTES = 2 * 1024 ** 3 # Never fill the disk below this much free space


class FeatureStore:
    """Feature store in a folder: one sub-folder per entry (plane), no eviction.

    Parameters
    ----------
    folder : str or Path
        Folder to store the features in. Created if it does not exist; an existing
        folder must be empty or a feature store (marked by a marker file).
    """

    def __init__(self, folder):
        self.folder = str(folder)
        os.makedirs(self.folder, exist_ok=True)
        marker = os.path.join(self.folder, _MARKER)
        if not os.path.isfile(marker):
            if os.listdir(self.folder):
                raise ValueError(f"'{self.folder}' is not empty and not a feature store. "
                                 "Choose an empty (or not yet existing) folder.")
            with open(marker, 'w') as f:
                json.dump({"format": 1}, f)
        self.hits = 0
        self.misses = 0
        self._warned_full = False

    # -- keys ---------------------------------------------------------------

    @staticmethod
    def _entry_name(key):
        """Folder name of an entry: <data signature>_<hash of the FE signature>."""
        data_sig, fe_sig = key
        fe_hash = hashlib.blake2b(repr(fe_sig).encode(), digest_size=8).hexdigest()
        return f"{data_sig}_{fe_hash}"

    def _entry_dir(self, key):
        return os.path.join(self.folder, self._entry_name(key))

    def _entry_dirs(self):
        return [os.path.join(self.folder, n) for n in os.listdir(self.folder)
                if os.path.isdir(os.path.join(self.folder, n))]

    # -- public API ---------------------------------------------------------

    def __contains__(self, key):
        return os.path.isfile(os.path.join(self._entry_dir(key), _META))

    def get(self, key):
        """Return the stored payload for `key` (arrays memory-mapped, copy-on-write), or None."""
        entry_dir = self._entry_dir(key)
        meta_path = os.path.join(entry_dir, _META)
        if not os.path.isfile(meta_path):
            self.misses += 1
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        scales = []
        for i, scale in enumerate(meta["scales"]):
            arrays = [np.load(os.path.join(entry_dir, f"s{i}_{j}.npy"), mmap_mode='c')
                      for j in range(scale["n_arrays"])]
            scales.append((arrays, tuple(scale["pre_shape"]), tuple(scale["reduced_shape"])))
        self.hits += 1
        return {"scales": scales, "was_torch": meta["was_torch"]}

    def put(self, key, payload):
        """Store `payload` under `key` (no-op if present or if the disk would get too full)."""
        if key in self:
            return
        nbytes = sum(a.nbytes for arrays, _, _ in payload["scales"] for a in arrays)
        if shutil.disk_usage(self.folder).free - nbytes < _DISK_HEADROOM_BYTES:
            if not self._warned_full:
                warnings.warn(f"Feature store '{self.folder}': disk (almost) full, features are not stored anymore.")
                self._warned_full = True
            return
        # Write into a temporary folder and rename it, so that readers never see a partial entry
        entry_dir = self._entry_dir(key)
        tmp_dir = entry_dir + ".tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir)
        meta = {"was_torch": bool(payload["was_torch"]), "scales": []}
        for i, (arrays, pre_shape, reduced_shape) in enumerate(payload["scales"]):
            for j, a in enumerate(arrays):
                np.save(os.path.join(tmp_dir, f"s{i}_{j}.npy"), np.ascontiguousarray(a))
            meta["scales"].append({"n_arrays": len(arrays),
                                   "pre_shape": list(pre_shape),
                                   "reduced_shape": list(reduced_shape)})
        with open(os.path.join(tmp_dir, _META), 'w') as f:
            json.dump(meta, f)
        os.rename(tmp_dir, entry_dir)

    def clear(self):
        """Delete all entries (keeps the folder and its marker)."""
        for d in self._entry_dirs():
            shutil.rmtree(d, ignore_errors=True)

    def __len__(self):
        return sum(os.path.isfile(os.path.join(d, _META)) for d in self._entry_dirs())

    @property
    def nbytes(self):
        """Total size of the stored features on disk."""
        return sum(os.path.getsize(os.path.join(d, n))
                   for d in self._entry_dirs() for n in os.listdir(d))

    def stats(self):
        return {
            "entries": len(self),
            "bytes": self.nbytes,
            "hits": self.hits,
            "misses": self.misses,
            "folder": self.folder,
        }
