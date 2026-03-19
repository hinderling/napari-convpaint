import torch
import numpy as np
import skimage
from .conv_paint_utils import get_device_from_torch_model

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    models = None
    CELLPOSE_AVAILABLE = False

AVAILABLE_MODELS = ['cellpose_backbone'] if CELLPOSE_AVAILABLE else []
IMPORT_ERROR_MESSAGE = (
    "Cellpose is not installed and is not available as feature extractor.\n"
    "Run 'pip install napari-convpaint[cellpose]' to install it."
)

from .conv_paint_feature_extractor import FeatureExtractor

class CellposeFeatures(FeatureExtractor):

    def __init__(self, model_name='cellpose_backbone', model=None, **kwargs):

        super().__init__(model_name=model_name, model=model)
        self.patch_size = 8
        self.num_input_channels = [2]
        self.norm_mode = "percentile"

        self.device = get_device_from_torch_model(self.model) if self.model is not None else torch.device("cpu")

    @staticmethod
    def create_model(model_name):
        # Load the cellpose model
        model_cellpose = models.CellposeModel(model_type='tissuenet_cp3',
                                              gpu=False) # We will move the model to the appropriate device at feature extraction time
        return model_cellpose

    def get_description(self):
        return "Model specialized in cell segmentation."
    
    def gives_patched_features(self) -> bool:
        # Requires image divisible by 8x8 patches as input, but returns non-patched features
        return False

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_layers = None
        param.fe_scalings = [1]
        param.fe_order = 0
        param.tile_annotations = False
        return param
    
    def move_model_to_device(self, use_device="auto"):
        """
        Move cellpose feature extractor model to the resolved runtime device.
        """
        device = self.resolve_device(use_device)

        if self.model is None:
            self.device = device
            return device

        current_device = next(self.model.net.parameters()).device
        if current_device != device:
            self.model.net = self.model.net.to(device)
            self.model.device = device
            self.model.gpu = device.type in ("cuda", "mps")
            if hasattr(self.model, "eval"):
                self.model.eval()

        self.device = device
        return device
    
    def supports_gpu(self):
        return (self.model is not None and
                hasattr(self.model, "net")
        )

    def get_features_from_plane(self, image, use_device='auto'):

        self.move_model_to_device(use_device)
        net = getattr(self.model, "net", self.model)
        image_expanded = np.expand_dims(image, axis=0)
        tensor = torch.from_numpy(image_expanded).float()
        tensor = tensor.to(self.device)
        use_mkldnn = getattr(net, "mkldnn", False) and self.device.type == "cpu"

        with torch.no_grad():
            if use_mkldnn:
                tensor = tensor.to_mkldnn()
            T0 = net.downsample(tensor)
            if use_mkldnn:
                style = net.make_style(T0[-1].to_dense())
            else:
                style = net.make_style(T0[-1])
            if not net.style_on:
                style = style * 0
            T1 = net.upsample(style, T0, use_mkldnn)
            T1 = net.output(T1)
            if use_mkldnn:
                T0 = [t0.to_dense() for t0 in T0]
                T1 = T1.to_dense()

        w_img,h_img = image.shape[-2:]
        out_t = []
        #append the output tensors from T0
        for t in T0[:3]:
            # Put to cpu, detach, and convert to numpy
            t = t.detach().cpu().numpy()[0]
            # Resize if necessary
            f,w,h = t.shape[-3:]
            if (w,h) != (w_img,h_img):
                t = skimage.transform.resize(
                        image=t,
                        output_shape=(f, w_img, h_img),
                        preserve_range=True, order=0)
            out_t.append(t)

        #append the output tensor from T1 (gradients and cell probability)
        t = T1.detach().cpu().numpy()[0]
        f,w,h = t.shape[-3:]
        if (w,h) != (w_img,h_img):
            t = skimage.transform.resize(
                    image=t,
                    output_shape=(f, w_img, h_img),
                    preserve_range=True, order=0)
        out_t.append(t)

        #append the original image
        out_t.append(image)
        
        #combine the tensors
        out_t = np.concatenate(out_t, axis=0)

        return out_t
