import timm
from ..utils import guided_model_download
from .dino import Dinov2Features

# Weights are pulled from the open timm-mirrored HuggingFace repos. To add a
# variant, register it here and in AVAILABLE_MODELS. Requires timm >= 1.0.20.
def _hf_url(timm_name):
    return f"https://huggingface.co/timm/{timm_name}/resolve/main/model.safetensors"

DINOV3_MODELS = {
    'dinov3_small-plus': {
        'timm_name': 'vit_small_plus_patch16_dinov3.lvd1689m',
        'patch_size': 16,
        'embed_dim': 384,
    },
    # 'dinov3_vits16':     {'timm_name': 'vit_small_patch16_dinov3.lvd1689m',       'patch_size': 16, 'embed_dim': 384},
    # 'dinov3_vitb16':     {'timm_name': 'vit_base_patch16_dinov3.lvd1689m',        'patch_size': 16, 'embed_dim': 768},
    # 'dinov3_vitl16':     {'timm_name': 'vit_large_patch16_dinov3.lvd1689m',       'patch_size': 16, 'embed_dim': 1024},
    # 'dinov3_vith16plus': {'timm_name': 'vit_huge_plus_patch16_dinov3.lvd1689m',   'patch_size': 16, 'embed_dim': 1280},
}

AVAILABLE_MODELS = ['dinov3_small-plus']

STD_MODELS = {
    "dinov3": {"fe_name": "dinov3_small-plus"},
}


class Dinov3Features(Dinov2Features):
    """Feature extractor using DINOv3, a self-supervised vision transformer model from Meta AI Research.

    Loaded via the timm library (timm-mirrored HuggingFace weights, no gating).
    Shares the whole extraction pipeline with Dinov2Features; only model
    creation and the patch-token slicing differ.
    """

    MODELS = DINOV3_MODELS

    def __init__(self, model_name='dinov3_small-plus', **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        # CLS + register tokens prefix the patch tokens in forward_features output
        self.num_prefix_tokens = getattr(self.model, 'num_prefix_tokens', 5)

    @staticmethod
    def create_model(model_name):
        spec = DINOV3_MODELS[model_name]
        # Pre-fetch with guided progress (mirrors DINOv2 UX), then point timm at the
        # local file instead of letting it download silently via huggingface_hub.
        weights_filename = f"{spec['timm_name'].replace('.', '_')}.safetensors"
        local_path = guided_model_download(weights_filename, _hf_url(spec['timm_name']))
        model = timm.create_model(
            spec['timm_name'],
            pretrained=False,
            num_classes=0,  # feature-extraction mode, no classifier head
            checkpoint_path=local_path,
        )
        model.eval()
        return model

    def get_description(self):
        desc = "Foundational ViT model (DINOv3). Extracts long range, semantic features."
        desc += "\nGood for: natural images (living beings, objects), histology etc."
        desc += "\n(The ViT-S+ version is used, with 4 register tokens and patch size 16x16.)"
        return desc

    def _get_patch_tokens(self, features_out):
        # timm returns [B, num_prefix_tokens + N_patches, D] — drop the prefix tokens.
        return features_out[:, self.num_prefix_tokens:, :]
