from typing import List


def _get_resnet_stitched_layers(block_depths: List[int]) -> List[str]:
    layers = []
    for idx, depth in enumerate(block_depths):
        layers.extend([f"layer{idx + 1}.{d}" for d in range(depth)])
    return layers


def get_stitched_layers(model_name: str) -> List[str]:
    if "resnet18" in model_name:
        return _get_resnet_stitched_layers([2, 2, 2, 2])
    if "resnet34" in model_name:
        return _get_resnet_stitched_layers([3, 4, 6, 3])
    if "resnet50" in model_name:
        return _get_resnet_stitched_layers([3, 4, 6, 3])
    if "resnet101" in model_name:
        return _get_resnet_stitched_layers([3, 4, 23, 3])
    if "resnet152" in model_name:
        return _get_resnet_stitched_layers([3, 8, 36, 3])
