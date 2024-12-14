import timm
from torch import nn

from .vision_transformer import *


def get_model(model_name: str, n_classes: int, n_channels: int) -> nn.Module:
    if model_name == "vit_tiny_patch4_32":
        return vit_tiny_patch4_32(n_classes, n_channels)
    if model_name == "vit_small_patch4_32":
        return vit_small_patch4_32(n_classes, n_channels)
    if model_name == "vit_medium_patch4_32":
        return vit_medium_patch4_32(n_classes, n_channels)
    if model_name == "vit_base_patch4_32":
        return vit_base_patch4_32(n_classes, n_channels)
    if model_name == "vit_large_patch4_32":
        return vit_large_patch4_32(n_classes, n_channels)
    if model_name == "vit_huge_patch4_32":
        return vit_huge_patch4_32(n_classes, n_channels)
    if model_name == "vit_giant_patch4_32":
        return vit_giant_patch4_32(n_classes, n_channels)
    if model_name == "vit_gigantic_patch4_32":
        return vit_gigantic_patch4_32(n_classes, n_channels)
    if model_name == "vit_tiny_patch16_224":
        return vit_tiny_patch16_224(n_classes, n_channels)
    if model_name == "vit_small_patch16_224":
        return vit_small_patch16_224(n_classes, n_channels)
    if model_name == "vit_medium_patch16_224":
        return vit_medium_patch16_224(n_classes, n_channels)
    if model_name == "vit_base_patch16_224":
        return vit_base_patch16_224(n_classes, n_channels)
    if model_name == "vit_large_patch16_224":
        return vit_large_patch16_224(n_classes, n_channels)
    if model_name == "vit_huge_patch16_224":
        return vit_huge_patch16_224(n_classes, n_channels)
    if model_name == "vit_giant_patch16_224":
        return vit_giant_patch16_224(n_classes, n_channels)
    if model_name == "vit_gigantic_patch16_224":
        return vit_gigantic_patch16_224(n_classes, n_channels)

    return getattr(timm.models.vision_transformer, model_name)(num_classes=n_classes)
