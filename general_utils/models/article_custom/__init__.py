from typing import Tuple
from functools import partial

import torch
from torch import nn

from general_utils.general import get_layer


def get_model(model_name: str, n_classes: int, n_channels: int, *args, **kwargs) -> nn.Module:
    if model_name == "liu2020_wrn":
        from .liu2020_wrn import WideResNet
        return WideResNet(depth=40,
                          num_classes=n_classes,
                          widen_factor=2,
                          dropRate=0.0)
    if model_name.startswith("resnet18_"):
        from .custom_resnet import resnet18
        return resnet18(n_classes=n_classes,
                        n_channels=n_channels,
                        same_width="_samewidth" in model_name,
                        same_size="_samesize" in model_name)
    if model_name == "repr_cls_resnet18":
        from .repr_cls_resnet import resnet18
        return resnet18(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_resnet34":
        from .repr_cls_resnet import resnet34
        return resnet34(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_resnet50":
        from .repr_cls_resnet import resnet50
        return resnet50(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_resnet101":
        from .repr_cls_resnet import resnet101
        return resnet101(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_resnet152":
        from .repr_cls_resnet import resnet152
        return resnet152(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_tiny":
        from .repr_cls_vit import repr_cls_vit_tiny
        return repr_cls_vit_tiny(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_small":
        from .repr_cls_vit import repr_cls_vit_small
        return repr_cls_vit_small(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_medium":
        from .repr_cls_vit import repr_cls_vit_medium
        return repr_cls_vit_medium(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_base":
        from .repr_cls_vit import repr_cls_vit_base
        return repr_cls_vit_base(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_large":
        from .repr_cls_vit import repr_cls_vit_large
        return repr_cls_vit_large(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_huge":
        from .repr_cls_vit import repr_cls_vit_huge
        return repr_cls_vit_huge(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_giant":
        from .repr_cls_vit import repr_cls_vit_giant
        return repr_cls_vit_giant(n_classes, n_channels, *args, **kwargs)
    if model_name == "repr_cls_vit_gigantic":
        from .repr_cls_vit import repr_cls_vit_gigantic
        return repr_cls_vit_gigantic(n_classes, n_channels, *args, **kwargs)
    if model_name == "small_cnn":
        from .small_cnn import SmallCNN
        return SmallCNN(n_classes, n_channels, *args, **kwargs)
