import sys

if sys.platform == "win32":
    import torch  # must load before Qt/Napari on Windows

import gc
import torch
import pytest

# Fix for MPS memory leaks in tests:
@pytest.fixture(autouse=True)
def cleanup_mps_after_test():
    yield
    # delete any local refs in test code if possible, then:
    gc.collect()
    # empty PyTorch MPS cache if available
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass