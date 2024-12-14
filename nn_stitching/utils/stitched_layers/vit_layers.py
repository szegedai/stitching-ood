from typing import List


def _get_vit_stitched_layers(depth: int) -> List[str]:
    return [f"blocks.{d}" for d in range(depth)]


def get_stitched_layers(model_name: str) -> List[str]:
    if model_name.startswith("vit_tiny_"):
        return _get_vit_stitched_layers(12)
    if model_name.startswith("vit_small_"):
        return _get_vit_stitched_layers(12)
    if model_name.startswith("vit_medium_"):
        return _get_vit_stitched_layers(12)
    if model_name.startswith("vit_base_"):
        return _get_vit_stitched_layers(12)
    if model_name.startswith("vit_large_"):
        return _get_vit_stitched_layers(24)
    if model_name.startswith("vit_huge_"):
        return _get_vit_stitched_layers(32)
    if model_name.startswith("vit_giant_"):
        return _get_vit_stitched_layers(40)
    if model_name.startswith("vit_gigantic_"):
        return _get_vit_stitched_layers(48)
