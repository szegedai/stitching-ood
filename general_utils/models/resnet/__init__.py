from torch import nn
from torchvision.models import resnet as tv_resnet


from .preact_resnet import (PreActResNet, Swish, preact_resnet18,
                            preact_resnet34)
from .resnet import (ResNet, resnet18, resnet34, resnet50, resnet101,
                     resnet152, resnext50_32x4d, resnext101_32x8d,
                     wide_resnet50_2, wide_resnet101_2)
from .wide_resnet import WideResNet, wide_resnet_28_10, wide_resnet_34_10


def get_model(model_name: str, n_classes: int, n_channels: int) -> nn.Module:
    if model_name == "resnet18":
        return resnet18(n_classes, n_channels)
    if model_name == "resnet34":
        return resnet34(n_classes, n_channels)
    if model_name == "resnet50":
        return resnet50(n_classes, n_channels)
    if model_name == "resnet101":
        return resnet101(n_classes, n_channels)
    if model_name == "resnet152":
        return resnet152(n_classes, n_channels)
    if model_name == "resnext50_32x4d":
        return resnext50_32x4d(n_classes, n_channels)
    if model_name == "resnext101_32x8d":
        return resnext101_32x8d(n_classes, n_channels)
    if model_name == "wide_resnet50_2":
        return wide_resnet50_2(n_classes, n_channels)
    if model_name == "wide_resnet101_2":
        return wide_resnet101_2(n_classes, n_channels)
    if model_name == "preact_resnet18":
        return preact_resnet18(n_classes, n_channels)
    if model_name == "preact_resnet34":
        return preact_resnet34(n_classes, n_channels)
    if model_name == "wide_resnet_28_10":
        return wide_resnet_28_10(n_classes, n_channels)
    if model_name == "wide_resnet_34_10":
        return wide_resnet_34_10(n_classes, n_channels)
    if model_name == "tv_resnet18":
        return tv_resnet.resnet18(num_classes=n_classes)
    if model_name == "tv_resnet34":
        return tv_resnet.resnet34(num_classes=n_classes)
    if model_name == "tv_resnet50":
        return tv_resnet.resnet50(num_classes=n_classes)
    if model_name == "tv_resnet101":
        return tv_resnet.resnet101(num_classes=n_classes)
    if model_name == "tv_resnet152":
        return tv_resnet.resnet152(num_classes=n_classes)
