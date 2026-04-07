"""Compatibility helpers for loading old pickles with renamed modules.

This module provides a very small, non-invasive compatibility layer:

- `safe_load(path_or_file)`: try normal ``pickle.load`` then fall back to a
  mapping-based Unpickler that remaps old module names to new ones.
- `migrate_pickle(src, dst=None)`: load (using compatibility) and re-dump
  the object using the current module paths so future loads do not need the
  compatibility layer.

Keep this file small and importable; do not modify existing code paths. Call
these helpers from migration scripts or GUI load handlers when needed.
"""
from __future__ import annotations

import os
import pickle
from typing import BinaryIO, Mapping

# Map old module names (as found in old pickles) -> current module name
# Add entries here for any renames you make in the codebase.
MODULE_RENAMES: Mapping[str, str] = {
    "napari_convpaint.conv_paint_param": "napari_convpaint.param",
}


class CompatUnpickler(pickle.Unpickler):
    """Unpickler that remaps module names during unpickling.

    Only remaps module names present in ``MODULE_RENAMES``. All other
    names are resolved normally.
    """

    def find_class(self, module: str, name: str):
        mapped = MODULE_RENAMES.get(module, module)
        return super().find_class(mapped, name)


def _open_file(path_or_file: "str | os.PathLike | BinaryIO") -> BinaryIO:
    if hasattr(path_or_file, "read"):
        return path_or_file  # file-like
    return open(path_or_file, "rb")


def safe_load(path_or_file: "str | os.PathLike | BinaryIO"):
    """Load a pickle with a safe fallback remapping old module names.

    Parameters
    - path_or_file: path or binary file-like object to read from.

    Behavior
    - First attempts a normal ``pickle.load``.
    - If that fails (commonly due to ModuleNotFoundError), it retries using
      ``CompatUnpickler`` which remaps module names from
      ``MODULE_RENAMES``.
    """
    opened = False
    f = _open_file(path_or_file)
    try:
        # If we were given a path, _open_file opened it for us.
        if not hasattr(path_or_file, "read"):
            opened = True
        try:
            f.seek(0)
        except Exception:
            pass
        try:
            return pickle.load(f)
        except Exception:
            try:
                f.seek(0)
            except Exception:
                # If we can't seek, re-open if path provided
                if opened:
                    f.close()
                    f = open(path_or_file, "rb")
                    opened = True
            return CompatUnpickler(f).load()
    finally:
        if opened and not f.closed:
            f.close()


def migrate_pickle(src: "str | os.PathLike | BinaryIO", dst: "str | os.PathLike | None" = None):
    """Load a pickle using compatibility rules and re-save it with current names.

    If ``dst`` is None the original file is replaced atomically (a temporary
    file is written and then moved over the original). Returns the output
    path written.
    """
    # Load using compatibility loader (works for normal files too)
    obj = safe_load(src)

    # Determine output path
    if dst is None:
        if hasattr(src, "read"):
            raise ValueError("dst must be provided when src is a file-like object")
        dst = os.fspath(src)

    dst_path = os.fspath(dst)
    # If replacing original, write to temp then replace
    if hasattr(src, "read") or os.path.abspath(dst_path) != os.path.abspath(os.fspath(src)):
        # writing to a different path
        with open(dst_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        tmp = dst_path + ".migrating"
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, dst_path)

    return dst_path


__all__ = ["safe_load", "migrate_pickle", "MODULE_RENAMES"]
