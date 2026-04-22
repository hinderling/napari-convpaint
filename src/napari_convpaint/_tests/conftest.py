import sys

if sys.platform == "win32":
    import torch  # must load before Qt/Napari on Windows

# Run long-running widget operations (train / predict / predict_all) on the
# calling thread in tests. The test assertions check layer state immediately
# after _on_train() / _on_predict(), which assumes synchronous execution.
from napari_convpaint.convpaint_widget import ConvpaintWidget  # noqa: E402
ConvpaintWidget._sync_workers = True