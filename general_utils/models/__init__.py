import re

from torch import nn


def get_model(model_name: str, n_classes: int, n_channels: int, *args, **kwargs) -> nn.Module:
    if model_name.startswith("repr_cls_"):
        from .article_custom import get_model
        return get_model(model_name, n_classes, n_channels, *args, **kwargs)
    if re.match(r"resnet.*_.*", model_name):
        from .article_custom import get_model
        return get_model(model_name, n_classes, n_channels)
    if "resnet" in model_name:
        from .resnet import get_model
        return get_model(model_name, n_classes, n_channels)
    if "vit" in model_name:
        from .vit import get_model
        return get_model(model_name, n_classes, n_channels)
    if "liu2020_wrn" in model_name:
        from .article_custom import get_model
        return get_model(model_name, n_classes, n_channels)
    if "small_cnn" in model_name:
        from .article_custom import get_model
        return get_model(model_name, n_classes, n_channels)
