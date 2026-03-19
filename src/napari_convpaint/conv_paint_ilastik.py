import itertools
import numpy as np

try:
    from ilastik.napari.filters import (FilterSet,
                                        Gaussian,
                                        LaplacianOfGaussian,
                                        GaussianGradientMagnitude,
                                        DifferenceOfGaussians,
                                        StructureTensorEigenvalues,
                                        HessianOfGaussianEigenvalues)
    ILASTIK_AVAILABLE = True
except ImportError as e:
    FilterSet = None
    Gaussian = None
    LaplacianOfGaussian = None
    GaussianGradientMagnitude = None
    DifferenceOfGaussians = None
    StructureTensorEigenvalues = None
    HessianOfGaussianEigenvalues = None
    ILASTIK_AVAILABLE = False

AVAILABLE_MODELS = ['ilastik_2d'] if ILASTIK_AVAILABLE else []
IMPORT_ERROR_MESSAGE = (
    "Ilastik is not installed and is not available as feature extractor.\n"
    "Run 'pip install napari-convpaint[ilastik]' to install it.\n"
    "Make sure to also have fastfilters installed ('conda install -c ilastik-forge fastfilters')."
)

# Define the filter set and scales
if not ILASTIK_AVAILABLE:
    FILTERS = tuple()
    FILTER_SET = None
else:
    FILTER_LIST = (Gaussian,
                LaplacianOfGaussian,
                GaussianGradientMagnitude,
                DifferenceOfGaussians,
                StructureTensorEigenvalues,
                HessianOfGaussianEigenvalues)
    SCALE_LIST = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
    # Generate all combinations of FILTER_LIST and SCALE_LIST
    ALL_FILTER_SCALING_COMBOS = list(itertools.product(range(len(FILTER_LIST)), range(len(SCALE_LIST))))
    FILTERS = tuple(FILTER_LIST[row](SCALE_LIST[col]) for row, col in sorted(ALL_FILTER_SCALING_COMBOS))
    FILTER_SET = FilterSet(filters=FILTERS)

from .conv_paint_feature_extractor import FeatureExtractor

class IlastikFeatures(FeatureExtractor):
    def __init__(self, model_name='ilastik_2d', **kwargs):
        super().__init__(model_name=model_name)
        self.padding = FILTER_SET.kernel_size // 2

    def get_description(self):
        return "Extraction of image features using the full filter set of the popular segmentation tool Ilastik."
        
    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_layers = None
        return param

    def get_features_from_plane(self, image, use_device='auto', filter_set=None):
        if filter_set is None:
            filter_set = FILTER_SET
        
        # Given that we get single-channel images as input:
        features = filter_set.transform(image[0]) # Ilastik wants 2D
        features = np.moveaxis(features, -1, 0)

        return features