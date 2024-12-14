import random
from typing import Union

import torch
import numpy as np
from torch import nn


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def parse_float_type(val: Union[str, int, float]):
    try:
        return float(val)
    except ValueError:
        return eval(val)


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    for name, layer in model.named_modules():
        if name == layer_name:
            return layer
    return None

