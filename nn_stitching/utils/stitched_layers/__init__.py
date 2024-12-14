from typing import List


def get_stitched_layers(model_name: str) -> List[str]:
    if model_name.startswith("resnet"):
        from .resnet_layers import get_stitched_layers
        return get_stitched_layers(model_name)
    if model_name.startswith("vit_"):
        from .vit_layers import get_stitched_layers
        return get_stitched_layers(model_name)
