from .dino import DinoFeatures
from .dino_jafar import DinoJafarFeatures
from .gaussian import GaussianFeatures
from .nnlayers import Hookmodel
from .combo_fe import ComboFeatures

# Optional imports
try:
    from .ilastik import IlastikFeatures
except ImportError as e:
    print(f"Ilastik feature extractor not available.")
    pass
try:
    from .cellpose import CellposeFeatures
except ImportError as e:
    print(f"Cellpose feature extractor not available.")
    pass