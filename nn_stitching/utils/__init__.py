from typing import Optional

import torch
from torch import nn


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer
    return None


def get_internal_activation(data: torch.Tensor,
                            target: torch.Tensor,
                            model: nn.Module,
                            layer_name: str,
                            transform: Optional[nn.Module] = None,
) -> torch.Tensor:
    """"""

    layer = get_layer(model, layer_name)
    activation = None

    def act_store_hook(m, i, o):
        nonlocal activation
        activation = o

    hook_handler = layer.register_forward_hook(act_store_hook)
    model(data)
    hook_handler.remove()

    if transform is not None:
        activation = transform(activation)

    return activation
