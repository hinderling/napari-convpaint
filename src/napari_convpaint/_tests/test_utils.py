import numpy as np
import pytest

from napari_convpaint import convpaint_model
from napari_convpaint.feature_extractors import Hookmodel
from napari_convpaint.param import Param
from napari_convpaint.utils import scale_img
import skimage

def test_hook_model():
    
    model = Hookmodel(model_name='vgg16')
    layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
          'features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
         ]
    model.register_hooks(selected_layers=layers)
    
    model.features_per_layer
    assert isinstance(model.features_per_layer, list), "Expect list of number of features"
    assert model.features_per_layer[0] == 64, f'Number of features expected from first layer is 64 but got {model.features_per_layer[0]}'
    assert model.features_per_layer[1] == 256, f'Number of features expected from first layer is 256 but got {model.features_per_layer[1]}'

def test_filter_image():

    param = Param()
    param.fe_name = 'vgg16'
    param.fe_layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
            'features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
            ]
    param.fe_use_min_features = False
    param.fe_order = 0
    param.fe_scalings = [1]
    param.tile_annotations = True
    param.image_downsample = 1

    model = convpaint_model.ConvpaintModel(param=param)

    image = skimage.data.cells3d()
    image = image[30,1]
    image = image[60:188, 0:128]

    features = model.get_feature_image(data=image)
    assert features.shape[0] == 320, f'Expecting 320 features but got {features.shape[1]}'
    assert features.shape[1] == 128, f'Expecting 128 annotated pixels but got {features.shape[0]}'

    #disable annotations tiling should lead to the same results
    model.set_param("tile_annotations", False)
    features = model.get_feature_image(data=image)
    assert features.shape[0] == 320, f'Expecting 320 features but got {features.shape[1]}'
    assert features.shape[1] == 128, f'Expecting 128 annotated pixels but got {features.shape[0]}'


@pytest.mark.parametrize("upscale", [False, True], ids=["down", "up"])
@pytest.mark.parametrize("factor", [2, 3, 5, 7])
@pytest.mark.parametrize("H,W", [(256, 256), (255, 257), (100, 101)])
def test_scale_img_image_and_labels_shape_match(factor, H, W, upscale):
    """Image and labels paths must produce the same spatial shape after scale_img.

    Regression: previously the downsample image path centre-cropped (floor(H/f))
    while the labels path padded (ceil(H/f)), so non-aligned `image_downsample`
    crashed `get_features_targets` (boolean-mask shape mismatch). After the fix,
    both paths pre-pad to a multiple of f and reduce, so shapes always agree.
    """
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, H, W)).astype(np.float32)
    lbl = rng.integers(0, 3, size=(1, H, W), dtype=np.uint16)

    img_out = scale_img(img, factor, upscale=upscale, input_type="img")
    lbl_out = scale_img(lbl, factor, upscale=upscale, input_type="labels")

    assert img_out.shape[-2:] == lbl_out.shape[-2:], (
        f"shape mismatch at factor={factor}, upscale={upscale}, (H,W)=({H},{W}): "
        f"img={img_out.shape[-2:]}  lbl={lbl_out.shape[-2:]}"
    )


def test_guided_model_download_is_atomic(tmp_path, monkeypatch):
    """A complete download lands under the final name; a truncated one raises and leaves no
    (partial) file behind, so the next call downloads again instead of using a corrupt file."""
    import os
    from napari_convpaint.utils import guided_model_download

    payload = b'x' * 1000

    class FakeResponse:
        def __init__(self, data, content_length):
            self.data, self.headers = data, {'content-length': str(content_length)}
        def raise_for_status(self): pass
        def iter_content(self, chunk_size):
            for i in range(0, len(self.data), chunk_size):
                yield self.data[i:i + chunk_size]
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_get(url, stream=True):
        return FakeResponse(payload if 'ok' in url else payload[:600], len(payload))
    monkeypatch.setattr('napari_convpaint.utils.requests.get', fake_get)

    path = guided_model_download('weights.pth', 'https://example.org/ok', model_dir=str(tmp_path))
    assert os.path.getsize(path) == 1000 and os.listdir(tmp_path) == ['weights.pth']

    with pytest.raises(RuntimeError, match='Incomplete download'):
        guided_model_download('truncated.pth', 'https://example.org/bad', model_dir=str(tmp_path))
    assert os.listdir(tmp_path) == ['weights.pth']   # no partial file left behind

