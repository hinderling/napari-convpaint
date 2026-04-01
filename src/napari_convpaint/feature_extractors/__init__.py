from napari_convpaint.feature_extractors.dino import DINOFeatures
from napari_convpaint.feature_extractors.dino_jafar import DinoJafarFeatures
from napari_convpaint.feature_extractors.gaussian import GaussianFeatures
from napari_convpaint.feature_extractors.nnlayers import Hookmodel
from napari_convpaint.feature_extractors.combo_fe import ComboFeatures

# Optional imports
try:
    from napari_convpaint.feature_extractors.ilastik import IlastikFeatures
except ImportError as e:
    print(f"Ilastik feature extractor not available.")
    pass
try:
    from napari_convpaint.feature_extractors.cellpose import CellposeFeatures
except ImportError as e:
    print(f"Cellpose feature extractor not available.")
    pass